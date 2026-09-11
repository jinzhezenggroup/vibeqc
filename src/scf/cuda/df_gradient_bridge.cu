#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/df_derivatives.cuh"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_df_gradient.hpp"

namespace vibeqc::scf {
namespace {
struct CudaFailure {
  cudaError_t status;
};
void check(cudaError_t status) {
  if (status != cudaSuccess) throw CudaFailure{status};
}
/** Restore thread-local selection only after the arena has drained its stream. */
struct DeviceGuard {
  int previous{};
  DeviceGuard() { check(cudaGetDevice(&previous)); }
  ~DeviceGuard() { (void)cudaSetDevice(previous); }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
};
/** Compact metadata independent of Direct quartet queues and derivative tensors. */
struct HostBasis {
  std::vector<std::int32_t> shell_atoms, ao_shells;
  std::vector<std::int64_t> primitive_offsets;
  std::vector<std::uint8_t> term_counts, term_angular;
  std::vector<double> term_coefficients, exponents, coefficients;
};
HostBasis pack(const core::System& system) {
  HostBasis h;
  h.primitive_offsets.push_back(0);
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > 3 || shell.atom_index >= system.atoms.size())
      throw std::invalid_argument(
          "generated DF gradients require valid orbital/auxiliary s/p/d/f shells");
    const auto si = static_cast<std::int32_t>(h.shell_atoms.size());
    h.shell_atoms.push_back(shell.atom_index);
    for (const auto& p : shell.primitives) {
      h.exponents.push_back(p.exponent);
      h.coefficients.push_back(p.coefficient);
    }
    h.primitive_offsets.push_back(h.exponents.size());
    for (const auto& expansion :
         molecule::ao_expansions(shell.angular_momentum, system.basis_representation)) {
      h.ao_shells.push_back(si);
      h.term_counts.push_back(expansion.size());
      for (std::size_t term = 0; term < molecule::kMaximumAoExpansionTerms; ++term) {
        if (term < expansion.size()) {
          const auto& item = expansion[term];
          for (auto power : item.component) h.term_angular.push_back(power);
          h.term_coefficients.push_back(
              item.coefficient * molecule::cartesian_component_normalization(item.component));
        } else {
          h.term_angular.insert(h.term_angular.end(), 3, 0);
          h.term_coefficients.push_back(0.0);
        }
      }
    }
  }
  return h;
}
/** The stream is drained before device buffers and host D2H destinations expire. */
struct Arena {
  cudaStream_t stream{};
  bool completed{};
  bool owns_stream{true};
  std::size_t budget;
  DfGradientResources stats;
  std::vector<void*> pointers;
  explicit Arena(std::size_t maximum) : budget(maximum) {}
  ~Arena() {
    if (stream && !completed) (void)cudaStreamSynchronize(stream);
    for (auto p : pointers) (void)runtime::resource_cuda_free(p);
    if (stream && owns_stream) (void)cudaStreamDestroy(stream);
  }
  void* allocate(std::size_t bytes) {
    if (bytes > budget - stats.device_bytes) throw std::bad_alloc();
    void* p = nullptr;
    check(runtime::resource_cuda_malloc(&p, bytes));
    try {
      pointers.push_back(p);
    } catch (...) {
      (void)runtime::resource_cuda_free(p);
      throw;
    }
    stats.device_bytes += bytes;
    return p;
  }
  template <class T>
  const T* upload(const std::vector<T>& data) {
    stats.host_bytes += data.capacity() * sizeof(T);
    if (data.empty()) return nullptr;
    auto* p = static_cast<T*>(allocate(data.size() * sizeof(T)));
    check(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    stats.host_to_device_bytes += data.size() * sizeof(T);
    ++stats.uploads;
    return p;
  }
  DfDerivativeBasisView upload(const HostBasis& b) {
    return {b.ao_shells.size(),          upload(b.shell_atoms), upload(b.ao_shells),
            upload(b.primitive_offsets), upload(b.term_counts), upload(b.term_angular),
            upload(b.term_coefficients), upload(b.exponents),   upload(b.coefficients)};
  }
};
}  // namespace
vibeqc_status execute_cuda_df_gradient(int device, const core::System& orbital,
                                       const core::System& auxiliary, std::span<const double> bar_a,
                                       std::span<const double> bar_m, unsigned schedule,
                                       std::size_t maximum_bytes, std::size_t maximum_tile_elements,
                                       std::vector<double>& gradient, std::string& detail,
                                       DfGradientResources* resources) {
  detail.clear();
  if (resources) *resources = {};
  const auto n = molecule::ao_count(orbital), a = molecule::ao_count(auxiliary),
             atoms = orbital.atoms.size();
  const auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(double);
  const auto index_limit = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  if (device < 0 || schedule > 1 || !maximum_bytes || !n || !a || !atoms || n > index_limit ||
      a > index_limit || atoms > index_limit / 3 || n > maximum / n || a > maximum / a ||
      n * n > maximum / a || atoms != auxiliary.atoms.size()) {
    detail = "invalid generated DF gradient dimensions or budget";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t i = 0; i < atoms; ++i)
    if (orbital.atoms[i].position != auxiliary.atoms[i].position) {
      detail = "DF orbital/auxiliary bases must share physical atom coordinates";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  if ((!bar_a.empty() && bar_a.size() != n * n * a) || (!bar_m.empty() && bar_m.size() != a * a)) {
    detail = "DF response weights require full A[mu,nu,P] and M[P,Q] layouts";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (auto weights : {bar_a, bar_m})
    if (!std::all_of(weights.begin(), weights.end(), [](double x) { return std::isfinite(x); })) {
      detail = "DF response weights must be finite";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  long double primitives = 0;
  for (const auto* system : {&orbital, &auxiliary})
    for (const auto& shell : system->shells) primitives += shell.primitives.size();
  // Conservative geometric-capacity bound before creating any metadata vectors.
  constexpr long double per_ao =
      2 * sizeof(std::int32_t) + sizeof(std::int64_t) + sizeof(std::uint8_t) +
      molecule::kMaximumAoExpansionTerms * (3 * sizeof(std::uint8_t) + sizeof(double));
  const long double host_bound = 2 * (per_ao * (n + a) + 2 * sizeof(double) * primitives +
                                      6 * sizeof(double) * atoms + 2 * sizeof(std::int64_t));
  if (host_bound > maximum_bytes) {
    detail = "generated DF host staging exceeds maximum_bytes";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  try {
    const auto host_o = pack(orbital), host_a = pack(auxiliary);
    std::vector<double> positions(3 * atoms), result(3 * atoms);
    for (std::size_t atom = 0; atom < atoms; ++atom)
      std::copy(orbital.atoms[atom].position.begin(), orbital.atoms[atom].position.end(),
                positions.begin() + 3 * atom);
    DeviceGuard device_guard;
    check(cudaSetDevice(device));
    Arena arena(maximum_bytes);
    check(cudaStreamCreateWithFlags(&arena.stream, cudaStreamNonBlocking));
    const auto o = arena.upload(host_o), x = arena.upload(host_a);
    const auto* r = arena.upload(positions);
    auto* output = static_cast<double*>(arena.allocate(result.size() * sizeof(double)));
    arena.stats.host_bytes += result.capacity() * sizeof(double);
    // Validate retained capacities as well as the geometric preflight bound;
    // vector growth policies must never permit a successful over-budget call.
    if (arena.stats.host_bytes > maximum_bytes) throw std::bad_alloc();
    check(cudaMemsetAsync(output, 0, result.size() * sizeof(double), arena.stream));
    const auto available = (maximum_bytes - arena.stats.device_bytes) / sizeof(double);
    const auto requested = maximum_tile_elements ? maximum_tile_elements : 65536U;
    const auto tile = std::min({available, requested, std::max(bar_a.size(), bar_m.size())});
    if ((!bar_a.empty() || !bar_m.empty()) && !tile) throw std::bad_alloc();
    auto* weights = tile ? static_cast<double*>(arena.allocate(tile * sizeof(double))) : nullptr;
    arena.stats.weight_tile_elements = tile;
    for (unsigned kind = 0; kind < 2; ++kind) {
      const auto source = kind ? bar_m : bar_a;
      for (std::size_t begin = 0; begin < source.size(); begin += tile) {
        const auto count = std::min(tile, source.size() - begin);
        check(cudaMemcpyAsync(weights, source.data() + begin, count * sizeof(double),
                              cudaMemcpyHostToDevice, arena.stream));
        arena.stats.host_to_device_bytes += count * sizeof(double);
        ++arena.stats.uploads;
        check(launch_df_derivative_tile(o, x, r, kind, {begin, 1, 1, 1}, count, weights, schedule,
                                        output, arena.stream));
        ++arena.stats.tiles;
      }
    }
    check(cudaMemcpyAsync(result.data(), output, result.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, arena.stream));
    arena.stats.device_to_host_bytes = result.size() * sizeof(double);
    check(cudaStreamSynchronize(arena.stream));
    arena.completed = true;
    arena.stats.stream_synchronizations = 1;
    if (!std::all_of(result.begin(), result.end(), [](double x) { return std::isfinite(x); })) {
      detail = "nonfinite generated DF gradient";
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    gradient.swap(result);
    if (resources) *resources = arena.stats;
    return VIBEQC_STATUS_SUCCESS;
  } catch (const CudaFailure& error) {
    detail = std::string("generated DF CUDA failure: ") + cudaGetErrorString(error.status);
    return error.status == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                     : VIBEQC_STATUS_CUDA_ERROR;
  } catch (const std::bad_alloc&) {
    detail = "generated DF gradient exceeded its allocation budget";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
}

vibeqc_status execute_cuda_df_hf_gradient(
    int device, void* stream_handle, CudaDensityFittingIntegralSource* source,
    std::size_t source_index, const core::System& orbital, const core::System& auxiliary,
    std::span<const double> raw_a, const std::vector<double>& metric,
    const std::vector<double>& inverse, std::span<const DensityFittingDensityResponse> terms,
    double relative_threshold, unsigned schedule, std::size_t maximum_bytes,
    std::size_t maximum_auxiliary_tile, std::vector<double>& gradient, std::string& detail,
    DfGradientResources* resources) {
  detail.clear();
  if (resources) *resources = {};
  const auto n = molecule::ao_count(orbital), a = molecule::ao_count(auxiliary),
             atoms = orbital.atoms.size();
  const auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(double);
  const auto index_limit = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  if (n > index_limit || a > index_limit || atoms > index_limit / 3 || device < 0 ||
      !stream_handle || schedule > 1 || !maximum_bytes || !n || !a || !atoms || n > maximum / n ||
      a > maximum / a || n * n > maximum / a || atoms > maximum / 3 ||
      atoms != auxiliary.atoms.size() || (!source && raw_a.size() != n * n * a) ||
      metric.size() != a * a || inverse.size() != a * a) {
    detail = "invalid generated DF-HF response dimensions or budget";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t atom = 0; atom < atoms; ++atom)
    if (orbital.atoms[atom].position != auxiliary.atoms[atom].position) {
      detail = "DF-HF orbital/auxiliary bases must share physical atoms";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  long double primitives = 0;
  for (const auto* system : {&orbital, &auxiliary})
    for (const auto& shell : system->shells) primitives += shell.primitives.size();
  constexpr long double per_ao =
      2 * sizeof(std::int32_t) + sizeof(std::int64_t) + sizeof(std::uint8_t) +
      molecule::kMaximumAoExpansionTerms * (3 * sizeof(std::uint8_t) + sizeof(double));
  const long double host_bound = 2 * (per_ao * (n + a) + 2 * sizeof(double) * primitives +
                                      6 * sizeof(double) * atoms + 2 * sizeof(std::int64_t));
  if (host_bound > maximum_bytes) {
    detail = "generated DF-HF metadata exceeds maximum_bytes";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  try {
    const auto host_o = pack(orbital), host_a = pack(auxiliary);
    std::vector<double> positions(3 * atoms), result(3 * atoms);
    for (std::size_t atom = 0; atom < atoms; ++atom)
      std::copy(orbital.atoms[atom].position.begin(), orbital.atoms[atom].position.end(),
                positions.begin() + 3 * atom);
    DeviceGuard device_guard;
    check(cudaSetDevice(device));
    Arena arena(maximum_bytes);
    arena.stream = reinterpret_cast<cudaStream_t>(stream_handle);
    arena.owns_stream = false;
    const auto o = arena.upload(host_o), x = arena.upload(host_a);
    const auto* r = arena.upload(positions);
    auto* output = static_cast<double*>(arena.allocate(result.size() * sizeof(double)));
    auto* device_values =
        source ? static_cast<double*>(arena.allocate(n * n * sizeof(double))) : nullptr;
    arena.stats.host_bytes += result.capacity() * sizeof(double);
    if (arena.stats.host_bytes >= maximum_bytes) throw std::bad_alloc();
    check(cudaMemsetAsync(output, 0, result.size() * sizeof(double), arena.stream));
    const auto weight_tile =
        std::min({std::size_t{65536}, std::max(n * n * a, a * a),
                  (maximum_bytes - arena.stats.device_bytes) / sizeof(double)});
    if (!weight_tile) throw std::bad_alloc();
    auto* weights = static_cast<double*>(arena.allocate(weight_tile * sizeof(double)));
    arena.stats.weight_tile_elements = weight_tile;
    const auto weight_stats = contract_density_fitting_response_weights(
        n, a, metric, inverse, terms, relative_threshold, maximum_bytes - arena.stats.host_bytes,
        maximum_auxiliary_tile,
        [&](std::size_t p, std::span<double> values) {
          bool drained = false;
          auto drain = [&] {
            if (!drained) (void)cudaStreamSynchronize(arena.stream);
          };
          runtime::ResourceScopeExit drain_before_host_reuse(drain);
          if (source) {
            const auto status = generate_cuda_density_fitting_raw_tile(
                source, source_index, 0, n * n, p, 1, -1, stream_handle, device_values, detail);
            if (status != VIBEQC_STATUS_SUCCESS) {
              if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
              throw std::runtime_error(detail);
            }
            check(cudaMemcpyAsync(values.data(), device_values, values.size_bytes(),
                                  cudaMemcpyDeviceToHost, arena.stream));
            check(cudaStreamSynchronize(arena.stream));
            arena.stats.device_to_host_bytes += values.size_bytes();
            ++arena.stats.stream_synchronizations;
          } else {
            for (std::size_t ij = 0; ij < n * n; ++ij) values[ij] = raw_a[ij * a + p];
          }
          drained = true;
        },
        [&](unsigned kind, runtime::StridedRange range, std::span<const double> host_weights) {
          bool drained = false;
          auto drain = [&] {
            if (!drained) (void)cudaStreamSynchronize(arena.stream);
          };
          runtime::ResourceScopeExit drain_before_host_reuse(drain);
          for (std::size_t begin = 0; begin < host_weights.size(); begin += weight_tile) {
            const auto count = std::min(weight_tile, host_weights.size() - begin);
            check(cudaMemcpyAsync(weights, host_weights.data() + begin, count * sizeof(double),
                                  cudaMemcpyHostToDevice, arena.stream));
            arena.stats.host_to_device_bytes += count * sizeof(double);
            ++arena.stats.uploads;
            check(launch_df_derivative_tile(o, x, r, kind, range, count, weights, schedule, output,
                                            arena.stream, begin));
            ++arena.stats.tiles;
          }
          // The adapter reuses this host span after returning. Drain all its
          // uploads before that mutation, even on pageable-memory CUDA paths.
          check(cudaStreamSynchronize(arena.stream));
          ++arena.stats.stream_synchronizations;
          drained = true;
        });
    arena.stats.host_bytes += weight_stats.host_peak_bytes;
    arena.stats.value_slices = weight_stats.value_slices;
    arena.stats.auxiliary_weight_tile = weight_stats.auxiliary_tile;
    check(cudaMemcpyAsync(result.data(), output, result.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, arena.stream));
    arena.stats.device_to_host_bytes += result.size() * sizeof(double);
    check(cudaStreamSynchronize(arena.stream));
    ++arena.stats.stream_synchronizations;
    arena.completed = true;
    if (!std::all_of(result.begin(), result.end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("nonfinite generated DF-HF gradient");
    gradient.swap(result);
    if (resources) *resources = arena.stats;
    return VIBEQC_STATUS_SUCCESS;
  } catch (const CudaFailure& error) {
    detail = std::string("generated DF-HF CUDA failure: ") + cudaGetErrorString(error.status);
    return error.status == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                     : VIBEQC_STATUS_CUDA_ERROR;
  } catch (const std::bad_alloc&) {
    detail = "generated DF-HF response exceeded its allocation budget";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    detail = error.what();
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
}
}  // namespace vibeqc::scf

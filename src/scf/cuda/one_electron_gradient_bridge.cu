#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "molecule/basis.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/one_electron_derivatives.cuh"
#include "scf/cuda_one_electron_gradient.hpp"

namespace vibeqc::scf {
namespace {
struct CudaFailure {
  cudaError_t status;
};
void check(cudaError_t status) {
  if (status != cudaSuccess) throw CudaFailure{status};
}

/** Restore thread-local device selection after the owning arena drains. */
struct DeviceGuard {
  int previous{};
  DeviceGuard() { check(cudaGetDevice(&previous)); }
  ~DeviceGuard() { (void)cudaSetDevice(previous); }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
};

/** Compact topology only: no Direct-HF quartet queues or integral tensors. */
struct HostView {
  std::vector<std::int64_t> atom_offsets, ao_offsets, primitive_offsets;
  std::vector<std::int32_t> atomic_numbers, shell_atoms, ao_shells, shell_first, shell_second,
      pair_first, pair_second;
  std::vector<std::uint8_t> term_counts, term_angular;
  std::vector<double> positions, term_coefficients, exponents, coefficients;
};

HostView pack(const core::System& system, unsigned schedule) {
  HostView host;
  host.atom_offsets = {0, static_cast<std::int64_t>(system.atoms.size())};
  for (const auto& atom : system.atoms) {
    host.atomic_numbers.push_back(atom.atomic_number);
    host.positions.insert(host.positions.end(), atom.position.begin(), atom.position.end());
  }
  host.ao_offsets.push_back(0);
  host.primitive_offsets.push_back(0);
  for (const auto& shell : system.shells) {
    const auto si = static_cast<std::int32_t>(host.shell_atoms.size());
    if (shell.angular_momentum > 3 || shell.atom_index >= system.atoms.size())
      throw std::invalid_argument("generated one-electron gradients require valid s/p/d/f shells");
    host.shell_atoms.push_back(shell.atom_index);
    for (const auto& primitive : shell.primitives) {
      host.exponents.push_back(primitive.exponent);
      host.coefficients.push_back(primitive.coefficient);
    }
    host.primitive_offsets.push_back(host.exponents.size());
    for (const auto& expansion :
         molecule::ao_expansions(shell.angular_momentum, system.basis_representation)) {
      host.ao_shells.push_back(si);
      host.term_counts.push_back(expansion.size());
      for (std::size_t term = 0; term < molecule::kMaximumAoExpansionTerms; ++term) {
        if (term < expansion.size()) {
          const auto& item = expansion[term];
          for (auto power : item.component) host.term_angular.push_back(power);
          host.term_coefficients.push_back(
              item.coefficient * molecule::cartesian_component_normalization(item.component));
        } else {
          host.term_angular.insert(host.term_angular.end(), 3, 0);
          host.term_coefficients.push_back(0.0);
        }
      }
    }
    host.ao_offsets.push_back(host.ao_shells.size());
    if (schedule == 1)
      for (std::int32_t sj = 0; sj <= si; ++sj) {
        host.shell_first.push_back(si);
        host.shell_second.push_back(sj);
      }
  }
  if (schedule == 0)
    for (std::int32_t i = 0; i < static_cast<std::int32_t>(host.ao_shells.size()); ++i)
      for (std::int32_t j = 0; j <= i; ++j) {
        host.pair_first.push_back(i);
        host.pair_second.push_back(j);
      }
  return host;
}

/** Resource-tracked allocations remain owned until all queued work completes. */
struct Arena {
  cudaStream_t stream{};
  bool completed{};
  std::vector<void*> allocations;
  std::size_t budget;
  OneElectronGradientResources stats;

  explicit Arena(std::size_t maximum) : budget(maximum) {}
  ~Arena() {
    if (stream && !completed) (void)cudaStreamSynchronize(stream);
    for (void* pointer : allocations) (void)runtime::resource_cuda_free(pointer);
    if (stream) (void)cudaStreamDestroy(stream);
  }
  void* allocate(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes > budget - stats.device_bytes) throw std::bad_alloc();
    void* pointer = nullptr;
    check(runtime::resource_cuda_malloc(&pointer, bytes));
    try {
      allocations.push_back(pointer);
    } catch (...) {
      (void)runtime::resource_cuda_free(pointer);
      throw;
    }
    stats.device_bytes += bytes;
    return pointer;
  }
  template <class T>
  const T* upload(std::span<const T> source) {
    if (source.empty()) return nullptr;
    auto* destination = static_cast<T*>(allocate(source.size_bytes()));
    check(cudaMemcpy(destination, source.data(), source.size_bytes(), cudaMemcpyHostToDevice));
    stats.host_to_device_bytes += source.size_bytes();
    ++stats.synchronous_uploads;
    return destination;
  }
  template <class T>
  const T* upload(const std::vector<T>& source) {
    stats.host_numeric_bytes += source.capacity() * sizeof(T);
    return upload<T>(std::span<const T>(source));
  }
};
}  // namespace

vibeqc_status execute_cuda_one_electron_gradient(
    int device_id, const core::System& system, std::span<const double> ws,
    std::span<const double> wt, std::span<const double> wv, unsigned schedule,
    std::size_t maximum_bytes, std::vector<double>& gradient, std::string& detail,
    OneElectronGradientResources* resources, double overlap_scale) {
  if (resources) *resources = {};
  const std::size_t n = molecule::ao_count(system), atoms = system.atoms.size();
  if (device_id < 0 || schedule > 2 || !maximum_bytes || !n || !atoms ||
      !std::isfinite(overlap_scale) ||
      atoms > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      n > std::numeric_limits<std::size_t>::max() / n || system.shells.size() > n) {
    detail = "invalid generated one-electron gradient dimensions or budget";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (auto weights : {ws, wt, wv})
    if ((!weights.empty() && weights.size() != n * n) ||
        !std::all_of(weights.begin(), weights.end(),
                     [](double value) { return std::isfinite(value); })) {
      detail = "one-electron weights must be finite full public-AO matrices";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  // Geometric vector growth may retain nearly twice the initialized numeric
  // entries. Bound that before creating any pair lists or metadata vectors.
  long double primitives = 0;
  for (const auto& shell : system.shells) primitives += shell.primitives.size();
  // Shell count is bounded by AO count. Include shell/primitive offsets,
  // expansion storage, atom data, output, and the larger of the pair lists.
  constexpr long double per_ao =
      2 * sizeof(std::int32_t) + 2 * sizeof(std::int64_t) + sizeof(std::uint8_t) +
      molecule::kMaximumAoExpansionTerms * (3 * sizeof(std::uint8_t) + sizeof(double));
  constexpr long double per_atom = sizeof(std::int32_t) + 6 * sizeof(double);
  const long double host_bound =
      2 * (per_ao * n + per_atom * atoms + 2 * sizeof(double) * primitives +
           sizeof(std::int32_t) * static_cast<long double>(n) * (n + 1) + 4 * sizeof(std::int64_t));
  if (host_bound > maximum_bytes) {
    detail = "generated one-electron host staging exceeds maximum_bytes";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  try {
    const auto host = pack(system, schedule);
    // Keep the D2H destination alive through Arena's failure-path stream drain.
    // A caller's old output capacity is outside this operation's budget, and
    // its input weights may alias that vector, so replace it only on success.
    std::vector<double> result(3 * atoms);
    DeviceGuard device_guard;
    check(cudaSetDevice(device_id));
    Arena arena(maximum_bytes);
    check(cudaStreamCreateWithFlags(&arena.stream, cudaStreamNonBlocking));
    OneElectronDeviceView view{1,
                               static_cast<std::int32_t>(n),
                               host.shell_first.size(),
                               arena.upload(host.atom_offsets),
                               arena.upload(host.atomic_numbers),
                               arena.upload(host.positions),
                               arena.upload(host.shell_atoms),
                               arena.upload(host.ao_offsets),
                               arena.upload(host.primitive_offsets),
                               arena.upload(host.shell_first),
                               arena.upload(host.shell_second),
                               arena.upload(host.ao_shells),
                               arena.upload(host.term_counts),
                               arena.upload(host.term_angular),
                               arena.upload(host.term_coefficients),
                               arena.upload(host.exponents),
                               arena.upload(host.coefficients)};
    const auto* first = arena.upload(host.pair_first);
    const auto* second = arena.upload(host.pair_second);
    OneElectronWeightView weights{arena.upload(ws), arena.upload(wt), nullptr};
    weights.overlap_scale = overlap_scale;
    weights.attraction =
        wv.data() == wt.data() && wv.size() == wt.size() ? weights.kinetic : arena.upload(wv);
    auto* output = static_cast<double*>(arena.allocate(3 * atoms * sizeof(double)));
    check(cudaMemsetAsync(output, 0, 3 * atoms * sizeof(double), arena.stream));
    check(launch_generated_one_electron_gradient(view, first, second, n * (n + 1) / 2, weights,
                                                 nullptr, schedule, 1.0, output, arena.stream));
    arena.stats.host_numeric_bytes += result.capacity() * sizeof(double);
    check(cudaMemcpyAsync(result.data(), output, 3 * atoms * sizeof(double), cudaMemcpyDeviceToHost,
                          arena.stream));
    arena.stats.device_to_host_bytes = 3 * atoms * sizeof(double);
    check(cudaStreamSynchronize(arena.stream));
    arena.completed = true;
    arena.stats.stream_synchronizations = 1;
    if (!std::all_of(result.begin(), result.end(), [](double x) { return std::isfinite(x); })) {
      detail = "nonfinite generated one-electron gradient";
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    gradient.swap(result);
    if (resources) *resources = arena.stats;
    return VIBEQC_STATUS_SUCCESS;
  } catch (const CudaFailure& failure) {
    detail =
        std::string("generated one-electron CUDA failure: ") + cudaGetErrorString(failure.status);
    return failure.status == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                       : VIBEQC_STATUS_NUMERICAL_FAILURE;
  } catch (const std::bad_alloc&) {
    detail = "generated one-electron gradient exceeded its allocation budget";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
}
}  // namespace vibeqc::scf

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "molecule/basis.hpp"
#include "runtime/cuda_component_trace.hpp"
#include "runtime/host_component_trace.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/df_derivatives.cuh"
#include "scf/cuda/df_packed_values.hpp"
#include "scf/cuda/df_response_weights.cuh"
#include "scf/cuda/df_shell_derivatives.cuh"
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
/** Diagnostic staging only; declare before Arena so queued reads drain first. */
struct PinnedResponseSlice {
  double* data{};
  ~PinnedResponseSlice() {
    if (data) (void)cudaFreeHost(data);
  }
};
/** Two bounded pinned panels, each protected by its last queued H2D read.
 * Declare before Arena: even an exceptional exit drains GPU reads before the
 * host allocation is released. Events mark copies, rather than later response
 * work, so recycling one panel need not drain the entire response stream.
 */
struct PinnedResponsePanels {
  struct Slot {
    cudaEvent_t last_read{};
    std::size_t begin{}, count{};
  } slots[2];
  double* data{};
  std::size_t matrix{}, auxiliary{}, columns{}, stride{}, next{};
  ~PinnedResponsePanels() {
    for (const auto& slot : slots)
      if (slot.last_read) (void)cudaEventDestroy(slot.last_read);
    if (data) (void)cudaFreeHost(data);
  }
  std::size_t initialize(std::size_t pairs, std::size_t naux, std::size_t available) {
    matrix = pairs;
    auxiliary = naux;
    // Offset adjacent column cache sets by one line. An unpadded AO-square
    // stride aliases cache sets for the 192/384 sentinels during the transpose.
    stride = ((matrix + 511) / 512) * 512 + 8;
    columns = std::min({std::size_t{16}, auxiliary, available / sizeof(double) / stride / 2});
    if (!columns) throw std::bad_alloc();
    const auto bytes = 2 * columns * stride * sizeof(double);
    check(cudaMallocHost(reinterpret_cast<void**>(&data), bytes));
    for (auto& slot : slots)
      check(cudaEventCreateWithFlags(&slot.last_read, cudaEventDisableTiming));
    return bytes;
  }
  unsigned prepare(std::size_t p, std::span<const double> raw, cudaStream_t stream,
                   std::size_t nbf) {
    for (unsigned i = 0; i < 2; ++i)
      if (p >= slots[i].begin && p - slots[i].begin < slots[i].count) {
        runtime::cuda_trace::trace_counter("raw_panel_host_cache_hits", 1);
        return i;
      }
    const auto i = static_cast<unsigned>(next++ % 2);
    auto& slot = slots[i];
    if (slot.count) {
      runtime::cuda_trace::TraceRegion reuse("raw_panel_host_reuse_wait", stream);
      check(cudaEventSynchronize(slot.last_read));
      runtime::cuda_trace::trace_counter("raw_panel_event_synchronizations", 1);
    }
    slot.begin = p / columns * columns;
    slot.count = std::min(columns, auxiliary - slot.begin);
    runtime::cuda_trace::TraceRegion gather("raw_panel_host_gather", stream);
    runtime::host_trace::Region host_gather("df_raw_panel_host_gather", nbf);
    auto* output = data + i * columns * stride;
    // Traverse a bounded row group one output column at a time. Contiguous
    // stores avoid cycling through sixteen distant destination pages for
    // every AO pair, while the group's source cache lines stay reusable by
    // the adjacent columns. This changes only byte-copy order, not storage.
    constexpr std::size_t gather_rows = 128;
    for (std::size_t first = 0; first < matrix; first += gather_rows)
      for (std::size_t q = 0; q < slot.count; ++q)
        for (std::size_t ij = first; ij < std::min(matrix, first + gather_rows); ++ij)
          output[q * stride + ij] = raw[ij * auxiliary + slot.begin + q];
    runtime::cuda_trace::trace_counter("raw_panel_host_gather_elements", matrix * slot.count);
    return i;
  }
  const double* column(unsigned slot, std::size_t p) const {
    return data + (slot * columns + p - slots[slot].begin) * stride;
  }
  void record(unsigned slot, cudaStream_t stream) {
    check(cudaEventRecord(slots[slot].last_read, stream));
  }
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

/** Optional shell traversal metadata, charged to the response owner's budget.
 * Class lists retain original shell order, so clipping a panel needs only two
 * binary searches per angular class, including panels cutting through a shell.
 */
struct ShellMetadata {
  struct SignatureGroup {
    unsigned angular{};
    std::size_t primitives{}, begin{}, count{};
  };
  std::vector<std::int32_t> ids, signature_ids;
  std::vector<std::int64_t> offsets;
  std::vector<SignatureGroup> signature_groups;
  DfShellBasisView view, signature_view;
  ShellMetadata(const core::System& system, const HostBasis& host, DfDerivativeBasisView basis,
                Arena& arena, bool signatures) {
    offsets.reserve(system.shells.size() + 1);
    for (std::size_t shell = 0; shell < system.shells.size(); ++shell)
      offsets.push_back(std::lower_bound(host.ao_shells.begin(), host.ao_shells.end(), shell) -
                        host.ao_shells.begin());
    offsets.push_back(host.ao_shells.size());
    ids.reserve(system.shells.size());
    if (signatures) signature_ids.reserve(system.shells.size());
    for (unsigned l = 0; l < 4; ++l) {
      view.begin[l] = ids.size();
      for (std::size_t shell = 0; shell < system.shells.size(); ++shell)
        if (system.shells[shell].angular_momentum == l) ids.push_back(shell);
      view.count[l] = ids.size() - view.begin[l];

      if (!signatures) continue;
      // Preserve public shell order within each signature: panel clipping then
      // remains a binary search, including a panel that splits an auxiliary shell.
      std::vector<std::size_t> primitive_counts;
      for (auto shell : std::span(ids).subspan(view.begin[l], view.count[l]))
        primitive_counts.push_back(system.shells[shell].primitives.size());
      std::sort(primitive_counts.begin(), primitive_counts.end());
      primitive_counts.erase(std::unique(primitive_counts.begin(), primitive_counts.end()),
                             primitive_counts.end());
      for (auto primitives : primitive_counts) {
        const auto begin = signature_ids.size();
        for (auto shell : std::span(ids).subspan(view.begin[l], view.count[l]))
          if (system.shells[shell].primitives.size() == primitives) signature_ids.push_back(shell);
        signature_groups.push_back({l, primitives, begin, signature_ids.size() - begin});
      }
    }
    view.basis = basis;
    view.shell_ids = arena.upload(ids);
    view.ao_offsets = arena.upload(offsets);
    signature_view.basis = basis;
    if (signatures) signature_view.shell_ids = arena.upload(signature_ids);
    signature_view.ao_offsets = view.ao_offsets;
    if (arena.stats.host_bytes > arena.budget) throw std::bad_alloc();
  }
  DfShellBasisView panel(std::size_t begin, std::size_t count) const {
    auto result = view;
    for (unsigned l = 0; l < 4; ++l) {
      const auto first = ids.begin() + view.begin[l], end = first + view.count[l];
      const auto low =
          std::partition_point(first, end, [&](auto shell) { return offsets[shell + 1] <= begin; });
      const auto high = std::partition_point(
          low, end, [&](auto shell) { return offsets[shell] < begin + count; });
      result.begin[l] = low - ids.begin();
      result.count[l] = high - low;
    }
    return result;
  }
  DfShellBasisView signature_group(std::size_t group, std::size_t panel_begin = 0,
                                   std::size_t panel_count = 0) const {
    DfShellBasisView result = signature_view;
    const auto& g = signature_groups.at(group);
    result.begin[g.angular] = g.begin;
    result.count[g.angular] = g.count;
    result.primitives = g.primitives;
    if (panel_count) {
      const auto first = signature_ids.begin() + g.begin, end = first + g.count;
      const auto low = std::partition_point(
          first, end, [&](auto shell) { return offsets[shell + 1] <= panel_begin; });
      const auto high = std::partition_point(
          low, end, [&](auto shell) { return offsets[shell] < panel_begin + panel_count; });
      result.begin[g.angular] = low - signature_ids.begin();
      result.count[g.angular] = high - low;
    }
    return result;
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
        arena.stats.response_host_to_device_bytes += count * sizeof(double);
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

vibeqc_status execute_cuda_df_gradient_tile(int device, const core::System& orbital,
                                            const core::System& auxiliary, unsigned kind,
                                            runtime::StridedRange range,
                                            std::span<const double> weights, unsigned schedule,
                                            std::size_t maximum_bytes,
                                            std::vector<double>& gradient, std::string& detail,
                                            DfGradientResources* resources) {
  detail.clear();
  if (resources) *resources = {};
  const auto n = molecule::ao_count(orbital), a = molecule::ao_count(auxiliary),
             atoms = orbital.atoms.size();
  const auto index_limit = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  const auto element_limit = std::numeric_limits<std::size_t>::max() / sizeof(double);
  if (device < 0 || kind > 1 || schedule > 1 || !maximum_bytes || weights.empty() || !n || !a ||
      !atoms || n > index_limit || a > index_limit || atoms > index_limit / 3 ||
      atoms != auxiliary.atoms.size() || !range.row_length || !range.row_stride ||
      !range.column_stride || n > element_limit / n || a > element_limit / a ||
      n * n > element_limit / a || weights.size() > element_limit)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  for (std::size_t atom = 0; atom < atoms; ++atom)
    if (orbital.atoms[atom].position != auxiliary.atoms[atom].position) {
      detail = "DF orbital/auxiliary bases must share physical atom coordinates";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  if (!std::all_of(weights.begin(), weights.end(),
                   [](double value) { return std::isfinite(value); })) {
    detail = "DF response weight tile must be finite";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto maximum = std::numeric_limits<std::size_t>::max();
  const auto row = (weights.size() - 1) / range.row_length;
  const auto column = (weights.size() - 1) % range.row_length;
  if (row > maximum / range.row_stride || column > maximum / range.column_stride ||
      range.offset > maximum - row * range.row_stride ||
      range.offset + row * range.row_stride > maximum - column * range.column_stride) {
    detail = "DF response weight tile range overflows size_t";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto last = range.index(weights.size() - 1);
  const auto total = kind ? a * a : n * n * a;
  if (last >= total) {
    detail = "DF response weight tile exceeds its full tensor";
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
    detail = "generated DF tile host staging exceeds maximum_bytes";
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
    if (arena.stats.host_bytes > maximum_bytes) throw std::bad_alloc();
    check(cudaMemsetAsync(output, 0, result.size() * sizeof(double), arena.stream));
    auto* device_weights = static_cast<double*>(arena.allocate(weights.size() * sizeof(double)));
    check(cudaMemcpyAsync(device_weights, weights.data(), weights.size() * sizeof(double),
                          cudaMemcpyHostToDevice, arena.stream));
    arena.stats.host_to_device_bytes += weights.size() * sizeof(double);
    arena.stats.response_host_to_device_bytes += weights.size() * sizeof(double);
    arena.stats.weight_tile_elements = weights.size();
    arena.stats.uploads += 1;
    check(launch_df_derivative_tile(o, x, r, kind, range, weights.size(), device_weights, schedule,
                                    output, arena.stream));
    arena.stats.tiles = 1;
    check(cudaMemcpyAsync(result.data(), output, result.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, arena.stream));
    arena.stats.device_to_host_bytes = result.size() * sizeof(double);
    check(cudaStreamSynchronize(arena.stream));
    arena.completed = true;
    arena.stats.stream_synchronizations = 1;
    if (!std::all_of(result.begin(), result.end(),
                     [](double value) { return std::isfinite(value); })) {
      detail = "nonfinite generated DF tile gradient";
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    gradient.swap(result);
    if (resources) *resources = arena.stats;
    return VIBEQC_STATUS_SUCCESS;
  } catch (const CudaFailure& error) {
    detail = std::string("generated DF tile CUDA failure: ") + cudaGetErrorString(error.status);
    return error.status == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                     : VIBEQC_STATUS_CUDA_ERROR;
  } catch (const std::bad_alloc&) {
    detail = "generated DF tile gradient exceeded its allocation budget";
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
    DfGradientResources* resources, const CudaDfMetricView* device_metric, void* blas_handle,
    const CudaDfResponseBuffers* borrowed, const CudaDfPackedRawTensorView* packed_raw) {
  detail.clear();
  // Validate even when the selected execution path retains strict evaluation.
  double target = 0;
  const char* screen_control = std::getenv("VIBEQC_DF_FORCE_SCREEN_ABS");
  if (screen_control && std::string_view(screen_control) != "off") {
    char* end = nullptr;
    target = std::strtod(screen_control, &end);
    if (end == screen_control || *end || !std::isfinite(target) || target < 0) {
      detail = "VIBEQC_DF_FORCE_SCREEN_ABS requires off or a finite nonnegative force budget";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }

  if (resources) *resources = {};
  const auto n = molecule::ao_count(orbital), a = molecule::ao_count(auxiliary),
             atoms = orbital.atoms.size();
  const auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(double);
  const auto index_limit = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  if (n > index_limit || a > index_limit || atoms > index_limit / 3 || device < 0 ||
      !stream_handle || schedule > 1 || !maximum_bytes || !n || !a || !atoms || n > maximum / n ||
      a > maximum / a || n * n > maximum / a || atoms > maximum / 3 ||
      atoms != auxiliary.atoms.size() || (source && !device_metric) ||
      (!source && !packed_raw && raw_a.size() != n * n * a) ||
      (device_metric && (!blas_handle || n * n > index_limit)) ||
      (device_metric ? (!device_metric->inverse_square_root || !device_metric->eigenvectors ||
                        !device_metric->eigenvalues)
                     : (metric.size() != a * a || inverse.size() != a * a)) ||
      terms.empty() || !std::isfinite(relative_threshold) || relative_threshold <= 0 ||
      relative_threshold >= 1) {
    detail = "invalid generated DF-HF response dimensions or budget";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto metric_matches = [&](const CudaDfMetricView& view) {
    return device_metric && view.inverse_square_root == device_metric->inverse_square_root &&
           view.eigenvectors == device_metric->eigenvectors &&
           view.eigenvalues == device_metric->eigenvalues &&
           view.relative_threshold == device_metric->relative_threshold;
  };
  if (packed_raw && (!packed_raw->data || !packed_raw->owner_identity || packed_raw->nbf != n ||
                     packed_raw->naux != a || packed_raw->pair_count != n * (n + 1) / 2 ||
                     !metric_matches(packed_raw->metric))) {
    detail = "packed raw DF view differs from response shape or metric owner";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  // A packed resident borrow is admitted only for the occupied algorithm.
  // Dense/rejected factors keep the bounded loader below; they must not enter
  // the old all-Q routine with capacities sized for one auxiliary panel.
  const bool packed_borrow =
      borrowed && packed_raw && borrowed->occupied_response &&
      borrowed->resident_packed_raw.data == packed_raw->data &&
      borrowed->resident_packed_raw.owner_identity == packed_raw->owner_identity &&
      borrowed->resident_packed_raw.nbf == n && borrowed->resident_packed_raw.naux == a &&
      borrowed->resident_packed_raw.pair_count == packed_raw->pair_count &&
      metric_matches(borrowed->resident_packed_raw.metric) &&
      packed_raw->data != borrowed->staging_weights &&
      packed_raw->data != borrowed->raw_auxiliary_major &&
      packed_raw->data != borrowed->exchange_response && !borrowed->resident_raw.data;
  const auto minimum_scratch = packed_borrow ? n * n : n * n * a;
  if (borrowed &&
      (!device_metric ||
       ((source || packed_raw || borrowed->resident_packed_raw.data) && !packed_borrow) ||
       n * n * a > maximum / 3 || borrowed->staging_capacity() < minimum_scratch ||
       borrowed->raw_capacity() < minimum_scratch ||
       borrowed->exchange_capacity() < minimum_scratch ||
       borrowed->staging_capacity() > maximum / 3 || borrowed->raw_capacity() > maximum / 3 ||
       borrowed->exchange_capacity() > maximum / 3 || !borrowed->staging_weights ||
       !borrowed->raw_auxiliary_major || !borrowed->exchange_response ||
       borrowed->staging_weights == borrowed->raw_auxiliary_major ||
       borrowed->staging_weights == borrowed->exchange_response ||
       borrowed->raw_auxiliary_major == borrowed->exchange_response)) {
    detail = "invalid borrowed resident DF response tensors";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (borrowed && borrowed->resident_raw.data) {
    const auto& raw = borrowed->resident_raw;
    if (!raw.owner_identity || raw.data != borrowed->raw_auxiliary_major || raw.nbf != n ||
        raw.naux != a || raw.auxiliary_stride != n * n || raw.row_stride != n ||
        raw.column_stride != 1 ||
        raw.metric.inverse_square_root != device_metric->inverse_square_root ||
        raw.metric.eigenvectors != device_metric->eigenvectors ||
        raw.metric.eigenvalues != device_metric->eigenvalues ||
        raw.metric.relative_threshold != device_metric->relative_threshold) {
      detail = "resident raw DF view differs from the response shape/layout/metric owner";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }
  if (borrowed && borrowed->occupied_response) {
    if (terms.size() > borrowed->occupied_factors.size()) {
      detail = "too many occupied DF response descriptors";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    std::size_t projected = 0;
    for (std::size_t t = 0; t < terms.size(); ++t) {
      const auto& factor = borrowed->occupied_factors[t];
      if (factor.rank > n || (factor.rank && !factor.coefficients) ||
          !std::isfinite(factor.density_scale) || factor.density_scale < 0 ||
          factor.rank * factor.rank > borrowed->staging_capacity() / a - projected ||
          factor.rank * factor.rank > borrowed->exchange_capacity() / a) {
        detail = "invalid occupied DF response factor or borrowed capacity";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
      projected += factor.rank * factor.rank;
    }
  }
  for (const auto& term : terms) {
    if (term.density.size() != n * n || !std::isfinite(term.coulomb_coefficient) ||
        !std::isfinite(term.exchange_coefficient)) {
      detail = "invalid HF density response term";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        if (!std::isfinite(term.density[i * n + j]) ||
            std::abs(term.density[i * n + j] - term.density[j * n + i]) > 1e-10) {
          detail = "HF response requires finite symmetric densities";
          return VIBEQC_STATUS_INVALID_ARGUMENT;
        }
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
    runtime::cuda_trace::TraceOperation trace("force_response",
                                              reinterpret_cast<cudaStream_t>(stream_handle),
                                              {1, n, a, source != nullptr, true, source_index});
    runtime::cuda_trace::TraceRegion preparation("response_allocation_and_uploads",
                                                 reinterpret_cast<cudaStream_t>(stream_handle));
    PinnedResponseSlice packed_slice;
    PinnedResponsePanels raw_panels;
    // The host destination must outlive Arena's exceptional-path stream drain.
    std::array<unsigned long long, 6> observed_shell_work{};
    std::array<unsigned long long, 3> observed_screen_work{};
    unsigned long long* screen_counters = nullptr;
    std::vector<unsigned long long> detailed_shell_work_host;
    DfShellDiagnostics detailed_shell_work;
    DfShellDiagnostics* shell_diagnostics = nullptr;
    unsigned long long* shell_counters = nullptr;
    Arena arena(maximum_bytes);
    arena.stream = reinterpret_cast<cudaStream_t>(stream_handle);
    arena.owns_stream = false;
    const auto o = arena.upload(host_o), x = arena.upload(host_a);
    // Clean complete-force endpoints qualify this combined default only for
    // resident 192--384-AO sm_120 execution and the qualified 768/768-AO
    // borrowed response. Small UHF regresses from launch and pinned-allocation
    // overhead; other backends and source-backed regimes need their own
    // endpoint evidence. Explicit selectors stay usable everywhere.
    // Attribution probes retain the original comparison defaults.
    bool promoted_default = false, packed_default = false, signature_device = false;
    const char* upload_diagnostic = std::getenv("VIBEQC_DF_RESPONSE_UPLOAD_PROBE");
    const char* scatter_diagnostic = std::getenv("VIBEQC_DF_RESPONSE_SCATTER_PROBE");
    const char* serial_diagnostic = std::getenv("VIBEQC_DF_SERIAL_RESPONSE_DOT");
    if (device_metric && schedule == 0 && (!source || packed_raw) &&
        ((n >= 192 && n <= 384) || (borrowed && n == 768 && a == 768)) &&
        !(upload_diagnostic && *upload_diagnostic) &&
        !(scatter_diagnostic && *scatter_diagnostic) &&
        !(serial_diagnostic && std::string_view(serial_diagnostic) == "1")) {
      cudaDeviceProp properties{};
      check(cudaGetDeviceProperties(&properties, device));
      promoted_default = properties.major == 12 && properties.minor == 0;
      signature_device =
          promoted_default && std::string_view(properties.name) == "NVIDIA GeForce RTX 5090";
      // Packed production is qualified on the same trusted 768-AO occupied
      // response as #381. Smaller defaults keep the faster dense response
      // producer and consume its shell pairs symmetrically.
      packed_default = promoted_default && borrowed && borrowed->occupied_response && n == 768 &&
                       a == 768 && terms.size() == 1 && borrowed->occupied_factors[0].rank == 160 &&
                       std::string_view(properties.name) == "NVIDIA GeForce RTX 5090";
    }
    const char* execution_control = std::getenv("VIBEQC_DF_WEIGHTED_EXECUTION");
    const std::string_view execution =
        execution_control ? execution_control : (promoted_default ? "shell" : "generic");
    if (execution != "generic" && execution != "shell-sp" && execution != "shell")
      throw std::invalid_argument("unknown DF weighted execution (use generic, shell-sp or shell)");
    const bool shell_execution = execution != "generic" && device_metric && schedule == 0;
    const bool full_shell_domain = execution == "shell";
    const char* pair_control = std::getenv("VIBEQC_DF_DERIVATIVE_PAIRS");
    const std::string_view pair_policy = pair_control ? pair_control : "auto";
    if (pair_policy != "auto" && pair_policy != "full" && pair_policy != "symmetric" &&
        pair_policy != "packed")
      throw std::invalid_argument(
          "unknown DF derivative pairs (use auto, full, symmetric or packed)");
    // Only the trusted occupied producer supplies folded packed AO weights.
    // Unsupported/corrected states retain the dense response, folding its two
    // ordered adjoints when a generated shell consumer is available.
    const bool packed_pairs =
        (pair_policy == "packed" || (pair_policy == "auto" && packed_default)) && shell_execution &&
        full_shell_domain && borrowed && borrowed->occupied_response;
    const auto derivative_pairs = packed_pairs ? DfDerivativePairs::packed
                                  : pair_policy == "symmetric" || pair_policy == "packed" ||
                                          (pair_policy == "auto" && promoted_default)
                                      ? DfDerivativePairs::symmetric
                                      : DfDerivativePairs::full;
    const auto response_pair_stride = packed_pairs ? n * (n + 1) / 2 : n * n;
    const char* block_control = std::getenv("VIBEQC_DF_PACKED_AO_BLOCK_ROWS");
    // The 64-row experiment halved weight storage but paid for many small
    // GEMMs. 256 rows recover the complete endpoint while still skipping all
    // upper off-diagonal blocks; diagnostic sizes preserve that experiment.
    const std::string_view block_policy = block_control ? block_control : "256";
    if (block_policy != "64" && block_policy != "128" && block_policy != "256" &&
        block_policy != "384")
      throw std::invalid_argument("unknown DF packed AO block size (use 64, 128, 256 or 384)");
    const std::size_t packed_block_rows = block_policy == "64"    ? 64
                                          : block_policy == "128" ? 128
                                          : block_policy == "256" ? 256
                                                                  : 384;
    const char* shell_schedule_control = std::getenv("VIBEQC_DF_SHELL_SCHEDULE");
    const std::string_view shell_schedule =
        shell_schedule_control ? shell_schedule_control : (promoted_default ? "compact" : "warp");
    if (shell_schedule != "auto" && shell_schedule != "warp" && shell_schedule != "packed" &&
        shell_schedule != "compact")
      throw std::invalid_argument("unknown DF shell schedule (use auto, warp, packed or compact)");
    const unsigned shell_variant = shell_schedule == "auto"     ? (promoted_default ? 2 : 0)
                                   : shell_schedule == "warp"   ? 0
                                   : shell_schedule == "packed" ? 1
                                                                : 2;
    const char* primitive_bucket_control = std::getenv("VIBEQC_DF_PRIMITIVE_BUCKETS");
    const std::string_view primitive_bucket_policy =
        primitive_bucket_control ? primitive_bucket_control : "auto";
    if (primitive_bucket_policy != "auto" && primitive_bucket_policy != "off" &&
        primitive_bucket_policy != "on" && primitive_bucket_policy != "packet")
      throw std::invalid_argument("unknown DF primitive buckets (use auto, off, on or packet)");
    // Automatic selection is an exact measured workload profile, not an AO-size
    // guess: both public bases must match the spherical def2-SVP water shell
    // histogram. Other signatures, representations and schedules stay opt-in.
    const auto signature_profile = [n](const core::System& system) {
      if (system.basis_representation != VIBEQC_BASIS_SPHERICAL) return false;
      std::array<std::size_t, 6> counts{};
      for (const auto& shell : system.shells) {
        const auto l = shell.angular_momentum;
        const auto p = shell.primitives.size();
        const int index = l == 0 && p == 1   ? 0
                          : l == 0 && p == 3 ? 1
                          : l == 0 && p == 5 ? 2
                          : l == 1 && p == 1 ? 3
                          : l == 1 && p == 3 ? 4
                          : l == 2 && p == 1 ? 5
                                             : -1;
        if (index < 0) return false;
        ++counts[index];
      }
      return counts == std::array<std::size_t, 6>{n / 6, n / 12, n / 24, n / 8, n / 24, n / 24};
    };
    const bool automatic_packets =
        primitive_bucket_policy == "auto" && signature_device && full_shell_domain &&
        shell_variant == 2 && a == n && terms.size() == 1 &&
        ((n == 384 && derivative_pairs == DfDerivativePairs::symmetric) ||
         (n == 768 && packed_default && packed_pairs)) &&
        signature_profile(orbital) && signature_profile(auxiliary);
    const bool signature_packets = primitive_bucket_policy == "packet" || automatic_packets;
    const bool primitive_buckets =
        shell_execution && (primitive_bucket_policy == "on" || signature_packets);
    runtime::cuda_trace::trace_counter("shell_primitive_signature_policy", !primitive_buckets  ? 0
                                                                           : signature_packets ? 2
                                                                                               : 1);
    std::unique_ptr<ShellMetadata> shell_o, shell_x;
    if (shell_execution) {
      {
        runtime::host_trace::Region metadata("df_shell_metadata", n);
        runtime::cuda_trace::TraceRegion metadata_device("shell_metadata_preparation",
                                                         arena.stream);
        shell_o = std::make_unique<ShellMetadata>(orbital, host_o, o, arena, primitive_buckets);
        shell_x = std::make_unique<ShellMetadata>(auxiliary, host_a, x, arena, primitive_buckets);
        runtime::cuda_trace::trace_counter(
            "shell_signature_id_bytes",
            (shell_o->signature_ids.size() + shell_x->signature_ids.size()) * sizeof(std::int32_t));
        runtime::cuda_trace::trace_counter(
            "shell_signature_groups",
            shell_o->signature_groups.size() + shell_x->signature_groups.size());
        runtime::cuda_trace::trace_counter(
            "shell_signature_host_range_bytes",
            (shell_o->signature_groups.capacity() + shell_x->signature_groups.capacity()) *
                sizeof(ShellMetadata::SignatureGroup));
      }
      // Count distinct orbital shell pairs once, independently of auxiliary
      // classes and panel clipping (which can revisit a partial auxiliary shell).
      std::size_t logical_pairs = 0;
      for (unsigned la = 0; la < 4; ++la)
        for (unsigned lb = 0; lb < 4; ++lb) {
          if ((!full_shell_domain && (la > 1 || lb > 1)) ||
              (!full_shell_domain && la + lb == 0 && !shell_x->view.count[1]) ||
              (derivative_pairs != DfDerivativePairs::full && la < lb))
            continue;
          const auto na = shell_o->view.count[la], nb = shell_o->view.count[lb];
          logical_pairs +=
              derivative_pairs != DfDerivativePairs::full && la == lb ? na * (na + 1) / 2 : na * nb;
        }
      runtime::cuda_trace::trace_counter("shell_pairs_logical", logical_pairs);
      const char* counter_control = std::getenv("VIBEQC_DF_SHELL_COUNTERS");
      if (counter_control && std::string_view(counter_control) == "1") {
        shell_counters =
            static_cast<unsigned long long*>(arena.allocate(sizeof(observed_shell_work)));
        check(cudaMemsetAsync(shell_counters, 0, sizeof(observed_shell_work), arena.stream));
        arena.stats.host_bytes += sizeof(observed_shell_work);
      }
      {
        if (target > 0 && full_shell_domain) {
          // S auxiliary shells have one AO and intersect exactly one response
          // panel. Ordered orbital pairs bound full/symmetric/packed work;
          // summing all primitive allowances therefore bounds every final
          // force component independently of response weights and cancellation.
          long double orbital_primitives = 0, auxiliary_primitives = 0;
          for (const auto& shell : orbital.shells)
            if (!shell.angular_momentum) orbital_primitives += shell.primitives.size();
          for (const auto& shell : auxiliary.shells)
            if (!shell.angular_momentum) auxiliary_primitives += shell.primitives.size();
          const auto capacity = orbital_primitives * orbital_primitives * auxiliary_primitives;
          if (capacity > 0 && capacity <= std::numeric_limits<unsigned long long>::max()) {
            const double budget = std::nextafter(static_cast<double>(target / capacity), 0.0);
            shell_o->view.force_screen_budget = shell_o->signature_view.force_screen_budget =
                budget;
            runtime::cuda_trace::trace_counter("screening_000_primitive_capacity",
                                               static_cast<unsigned long long>(capacity));
            runtime::cuda_trace::trace_counter("screening_000_enabled", budget > 0);
            if (shell_counters && budget > 0) {
              screen_counters =
                  static_cast<unsigned long long*>(arena.allocate(sizeof(observed_screen_work)));
              check(
                  cudaMemsetAsync(screen_counters, 0, sizeof(observed_screen_work), arena.stream));
              arena.stats.host_bytes += sizeof(observed_screen_work);
              shell_o->view.force_screen_counts = shell_o->signature_view.force_screen_counts =
                  screen_counters;
            }
          }
        }
      }
      const char* work_control = std::getenv("VIBEQC_DF_SHELL_WORK");
      if (work_control && std::string_view(work_control) == "1") {
        // One fixed packet buffer is reused after each explicitly intrusive
        // readback. Keep its host destination alive through exceptional drains.
        detailed_shell_work_host.resize(DfShellDiagnostics::elements);
        const auto bytes = detailed_shell_work_host.size() * sizeof(unsigned long long);
        detailed_shell_work.device = static_cast<unsigned long long*>(arena.allocate(bytes));
        detailed_shell_work.host = detailed_shell_work_host.data();
        shell_diagnostics = &detailed_shell_work;
        arena.stats.host_bytes += bytes;
        runtime::cuda_trace::trace_counter("shell_work_diagnostic_bytes", bytes);
        runtime::cuda_trace::trace_counter("shell_work_diagnostics_enabled", 1);
        runtime::cuda_trace::trace_counter("shell_work_pair_mode",
                                           static_cast<unsigned>(derivative_pairs));
      }
    }
    const auto* r = arena.upload(positions);
    auto* output = static_cast<double*>(arena.allocate(result.size() * sizeof(double)));
    arena.stats.host_bytes += result.capacity() * sizeof(double);
    if (arena.stats.host_bytes >= maximum_bytes) throw std::bad_alloc();
    check(cudaMemsetAsync(output, 0, result.size() * sizeof(double), arena.stream));
    if (device_metric) {
      // Choose the auxiliary block from the remaining *device* budget, after
      // basis metadata, coordinates and final output have been charged. Every
      // new response allocation is owned here; launch wrappers allocate
      // nothing. Full borrowed J/K tensors stay charged to the value plan.
      const long double fixed_elements =
          4.0L * a * a + (3.0L + terms.size()) * n * n + 2.0L * terms.size() * a;
      const auto available = maximum_bytes - arena.stats.device_bytes;
      if ((fixed_elements + 2.0L * n * n) * sizeof(double) > available) throw std::bad_alloc();
      const auto capacity =
          static_cast<std::size_t>((available / sizeof(double) - fixed_elements) / (2.0L * n * n));
      const auto tile =
          std::min({a, capacity, maximum_auxiliary_tile ? maximum_auxiliary_tile : a});
      // Occupied projection scratch is dead before derivative consumption.
      // Borrow at most 64 auxiliary AO matrices from it: this bounds W
      // independently of Naux and avoids fragmenting generated shell work.
      const auto consume_tile =
          borrowed ? std::min({a, borrowed->occupied_response ? std::size_t{64} : a,
                               borrowed->exchange_capacity() / (n * n),
                               maximum_auxiliary_tile ? maximum_auxiliary_tile : a})
                   : tile;
      auto* densities = static_cast<double*>(arena.allocate(terms.size() * n * n * sizeof(double)));
      for (std::size_t t = 0; t < terms.size(); ++t) {
        check(cudaMemcpyAsync(densities + t * n * n, terms[t].density.data(),
                              n * n * sizeof(double), cudaMemcpyHostToDevice, arena.stream));
        arena.stats.host_to_device_bytes += n * n * sizeof(double);
        arena.stats.density_host_to_device_bytes += n * n * sizeof(double);
        ++arena.stats.uploads;
      }
      auto* workspace = static_cast<double*>(arena.allocate(
          cuda_df_response_workspace_elements(n, a, terms.size(),
                                              borrowed && borrowed->occupied_response ? 0 : tile) *
          sizeof(double)));
      arena.stats.device_response = true;
      arena.stats.occupied_response = borrowed && borrowed->occupied_response;
      arena.stats.auxiliary_weight_tile = consume_tile;
      arena.stats.weight_tile_elements = consume_tile * response_pair_stride;
      if (borrowed) {
        // These allocations remain owned and charged by the value plan. Keep
        // their capacity visible without double-counting it as new response
        // scratch or silently widening the caller's private force allowance.
        arena.stats.borrowed_device_bytes =
            (borrowed->staging_capacity() + borrowed->raw_capacity() +
             borrowed->exchange_capacity()) *
            sizeof(double);
        runtime::cuda_trace::trace_counter("response_borrowed_jk_bytes",
                                           arena.stats.borrowed_device_bytes);
        runtime::cuda_trace::trace_counter("response_resident_auxiliary_tile", consume_tile);
      }
      runtime::cuda_trace::trace_counter("response_scratch_bytes", arena.stats.device_bytes);
      runtime::cuda_trace::trace_counter("density_upload_bytes",
                                         arena.stats.density_host_to_device_bytes);
      // Causal attribution controls, not production schedule candidates. Both
      // drain prior work before a raw read; packed additionally exposes the
      // CPU gather and a contiguous pinned H2D copy as separate intervals.
      // Probes retain the original pageable strided submission by default.
      const char* upload_probe = std::getenv("VIBEQC_DF_RESPONSE_UPLOAD_PROBE");
      const std::string_view probe = upload_probe ? upload_probe : "";
      if (!probe.empty() && probe != "drain" && probe != "packed")
        throw std::invalid_argument("unknown DF response upload probe (use drain or packed)");
      if (!probe.empty() && source)
        throw std::invalid_argument("DF response upload probe requires resident host raw values");
      if (borrowed && !probe.empty())
        throw std::invalid_argument("resident JK scratch cannot be combined with upload probes");
      const char* staging_control = std::getenv("VIBEQC_DF_RAW_STAGING");
      const std::string_view staging =
          staging_control ? staging_control : (promoted_default ? "pinned-panels" : "pageable");
      if (staging != "pageable" && staging != "pinned-panels")
        throw std::invalid_argument("unknown DF raw staging (use pageable or pinned-panels)");
      if (staging == "pinned-panels" && !source && !borrowed) {
        if (!probe.empty())
          throw std::invalid_argument("pinned panel staging cannot be combined with upload probes");
        const auto bytes = raw_panels.initialize(n * n, a, maximum_bytes - arena.stats.host_bytes);
        arena.stats.host_bytes += bytes;
        runtime::cuda_trace::trace_counter("raw_panel_pinned_host_bytes", bytes);
        runtime::cuda_trace::trace_counter("raw_panel_columns", raw_panels.columns);
      }
      if (probe == "packed") {
        const auto bytes = n * n * sizeof(double);
        if (bytes > maximum_bytes - arena.stats.host_bytes) throw std::bad_alloc();
        check(cudaMallocHost(reinterpret_cast<void**>(&packed_slice.data), bytes));
        arena.stats.host_bytes += bytes;
        runtime::cuda_trace::trace_counter("raw_probe_pinned_host_bytes", bytes);
      }
      // Keep the original auxiliary tile and response scratch unchanged. This
      // diagnostic reserves only unused budget headroom; insufficient room is
      // an error, never a silent tile/traffic change that confounds attribution.
      const char* sink_policy = std::getenv("VIBEQC_DF_RESPONSE_SCATTER_PROBE");
      const std::string_view sink = sink_policy ? sink_policy : "";
      if (!sink.empty() && sink != "sharded")
        throw std::invalid_argument("unknown DF response scatter probe (use sharded)");
      const unsigned gradient_copies = sink.empty() ? 1 : 128;
      if (shell_execution && gradient_copies != 1)
        throw std::invalid_argument("shell execution cannot be combined with the scatter probe");
      auto* derivative_output = output;
      const double* ones = nullptr;
      if (gradient_copies > 1) {
        const auto bytes = result.size() * gradient_copies * sizeof(double);
        if (gradient_copies * sizeof(double) > maximum_bytes - arena.stats.host_bytes)
          throw std::bad_alloc();
        derivative_output = static_cast<double*>(arena.allocate(bytes));
        ones = arena.upload(std::vector<double>(gradient_copies, 1.0));
        check(cudaMemsetAsync(derivative_output, 0, bytes, arena.stream));
        runtime::cuda_trace::trace_counter("derivative_probe_gradient_copies", gradient_copies);
        runtime::cuda_trace::trace_counter("derivative_probe_scratch_bytes",
                                           bytes + gradient_copies * sizeof(double));
      }
      runtime::cuda_trace::trace_counter("response_probe_total_device_bytes",
                                         arena.stats.device_bytes);
      preparation.finish();
      runtime::cuda_trace::TraceRegion response_weights("response_weights", arena.stream);
      const char* dot_policy = std::getenv("VIBEQC_DF_SERIAL_RESPONSE_DOT");
      const bool serial_dot = dot_policy && dot_policy[0] == '1' && dot_policy[1] == '\0';
      const char* algebra_control = std::getenv("VIBEQC_DF_RESPONSE_ALGEBRA");
      const std::string_view algebra =
          algebra_control ? algebra_control : (borrowed || promoted_default ? "blas" : "scalar");
      if (algebra != "scalar" && algebra != "blas")
        throw std::invalid_argument("unknown DF response algebra (use scalar or blas)");
      if (borrowed && (algebra != "blas" || serial_dot || gradient_copies != 1))
        throw std::invalid_argument(
            "resident JK scratch requires BLAS response without serial/scatter probes");
      if (borrowed && !borrowed->resident_raw.data && !packed_borrow) {
        arena.stats.host_to_device_bytes += raw_a.size_bytes();
        arena.stats.tensor_host_to_device_bytes += raw_a.size_bytes();
        arena.stats.value_slices += a;
        ++arena.stats.uploads;
      }
      if (borrowed && borrowed->resident_raw.data) {
        runtime::cuda_trace::trace_counter("raw_value_upload_bytes", 0);
        runtime::cuda_trace::trace_counter("raw_value_bulk_uploads", 0);
        runtime::cuda_trace::trace_counter("raw_value_reused_bytes", n * n * a * sizeof(double));
        runtime::cuda_trace::trace_counter("raw_value_owner_identity",
                                           borrowed->resident_raw.owner_identity);
      }
      if (packed_raw) {
        runtime::cuda_trace::trace_counter("raw_packed_value_reused_bytes",
                                           packed_raw->pair_count * a * sizeof(double));
        runtime::cuda_trace::trace_counter("raw_value_owner_identity", packed_raw->owner_identity);
      }
      check(contract_cuda_df_response_weights(
          n, a, terms, densities, *device_metric, consume_tile, workspace, arena.stream,
          reinterpret_cast<cublasHandle_t>(blas_handle), serial_dot, algebra == "blas",
          [&](std::size_t p, double* values) {
            if (packed_raw) {
              cuda_df::launch_unpack_df_values(arena.stream, n, a, 0, n, p, 1, true,
                                               packed_raw->data, values);
              check(cudaGetLastError());
              runtime::cuda_trace::trace_counter("raw_packed_unpacked_elements", n * n);
            } else if (source) {
              const auto status = generate_cuda_density_fitting_raw_tile(
                  source, source_index, 0, n * n, p, 1, -1, stream_handle, values, detail);
              if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
              if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
              arena.stats.recomputed_value_bytes += n * n * sizeof(double);
            } else {
              const unsigned raw_panel =
                  raw_panels.data ? raw_panels.prepare(p, raw_a, arena.stream, n) : 0;
              if (!probe.empty()) {
                runtime::cuda_trace::TraceRegion wait("raw_value_prior_stream_wait", arena.stream);
                check(cudaStreamSynchronize(arena.stream));
                ++arena.stats.stream_synchronizations;
                runtime::cuda_trace::trace_counter("raw_probe_prior_stream_drains", 1);
              }
              if (packed_slice.data) {
                runtime::cuda_trace::TraceRegion gather("raw_value_host_gather", arena.stream);
                runtime::host_trace::Region host_gather("df_raw_value_host_gather", n);
                for (std::size_t ij = 0; ij < n * n; ++ij)
                  packed_slice.data[ij] = raw_a[ij * a + p];
                host_gather.finish();
                runtime::cuda_trace::trace_counter("raw_probe_gather_elements", n * n);
              }
              // raw_a is caller-owned [mu*nu,P], live until the bridge drains
              // its stream. Gather a column directly into existing device
              // scratch; no host response matrix or full raw GPU copy is made.
              runtime::cuda_trace::TraceRegion upload("raw_value_slice_upload", arena.stream);
              if (raw_panels.data) {
                check(cudaMemcpyAsync(values, raw_panels.column(raw_panel, p),
                                      n * n * sizeof(double), cudaMemcpyHostToDevice,
                                      arena.stream));
                raw_panels.record(raw_panel, arena.stream);
              } else if (packed_slice.data)
                check(cudaMemcpyAsync(values, packed_slice.data, n * n * sizeof(double),
                                      cudaMemcpyHostToDevice, arena.stream));
              else
                check(cudaMemcpy2DAsync(values, sizeof(double), raw_a.data() + p,
                                        a * sizeof(double), sizeof(double), n * n,
                                        cudaMemcpyHostToDevice, arena.stream));
              arena.stats.host_to_device_bytes += n * n * sizeof(double);
              arena.stats.tensor_host_to_device_bytes += n * n * sizeof(double);
              ++arena.stats.uploads;
              runtime::cuda_trace::trace_counter("raw_value_upload_bytes", n * n * sizeof(double));
            }
            ++arena.stats.value_slices;
          },
          [&](unsigned kind, runtime::StridedRange range, std::size_t count,
              const double* weights) {
            const bool metric_weights = kind == 1;
            const auto stride = kind == 2 ? n * (n + 1) / 2 : n * n;
            runtime::cuda_trace::TraceRegion derivatives(
                metric_weights ? "metric_center_derivative_contraction"
                               : "three_center_derivative_contraction",
                arena.stream);
            runtime::cuda_trace::trace_counter(
                metric_weights ? "metric_derivative_weights" : "three_center_derivative_weights",
                count);
            runtime::cuda_trace::trace_counter(metric_weights
                                                   ? "metric_derivative_weight_bytes"
                                                   : "three_center_derivative_weight_bytes",
                                               count * sizeof(double));
            if (shell_execution && !metric_weights) {
              const auto panel_count = count / stride;
              if (shell_diagnostics) {
                char panel_name[96];
                std::snprintf(panel_name, sizeof(panel_name), "shell_work_panel_%zu_%zu",
                              range.offset, panel_count);
                runtime::cuda_trace::trace_counter(panel_name, 1);
              }
              if (primitive_buckets) {
                runtime::host_trace::Region grouping("df_shell_signature_dispatch", n);
                // Products of compact shell slices are implicit task queues: no
                // shell-triple descriptors, device prefix/scatter, or readback.
                std::vector<DfShellBasisView> auxiliary_groups;
                auxiliary_groups.reserve(shell_x->signature_groups.size());
                for (std::size_t gc = 0; gc < shell_x->signature_groups.size(); ++gc)
                  auxiliary_groups.push_back(
                      shell_x->signature_group(gc, range.offset, panel_count));
                if (signature_packets) {
                  std::vector<DfShellBasisView> orbital_groups;
                  orbital_groups.reserve(shell_o->signature_groups.size());
                  for (std::size_t g = 0; g < shell_o->signature_groups.size(); ++g)
                    orbital_groups.push_back(shell_o->signature_group(g));
                  runtime::cuda_trace::trace_maximum(
                      "signature_packet_host_view_bytes",
                      (orbital_groups.capacity() + auxiliary_groups.capacity()) *
                          sizeof(DfShellBasisView));
                  check(launch_df_shell_derivative_packets(
                      orbital_groups, auxiliary_groups, r, range.offset, panel_count, weights,
                      derivative_output, shell_counters, arena.stream, full_shell_domain,
                      shell_variant, derivative_pairs, shell_diagnostics));
                } else {
                  std::size_t signature_launches = 0;
                  for (std::size_t ga = 0; ga < shell_o->signature_groups.size(); ++ga) {
                    const auto& a_group = shell_o->signature_groups[ga];
                    for (std::size_t gb = 0; gb < shell_o->signature_groups.size(); ++gb) {
                      const auto& b_group = shell_o->signature_groups[gb];
                      if (derivative_pairs != DfDerivativePairs::full &&
                          (a_group.angular < b_group.angular ||
                           (a_group.angular == b_group.angular && ga < gb)))
                        continue;
                      const auto first = shell_o->signature_group(ga);
                      const auto second = shell_o->signature_group(gb);
                      for (std::size_t gc = 0; gc < shell_x->signature_groups.size(); ++gc) {
                        const auto lc = shell_x->signature_groups[gc].angular;
                        if (!auxiliary_groups[gc].count[lc] ||
                            (!full_shell_domain &&
                             (a_group.angular > 1 || b_group.angular > 1 || lc > 1 ||
                              a_group.angular + b_group.angular + lc == 0)))
                          continue;
                        check(launch_df_shell_derivative_group(
                            first, second, auxiliary_groups[gc], r, range.offset, panel_count,
                            weights, derivative_output, shell_counters, arena.stream,
                            full_shell_domain, shell_variant, derivative_pairs,
                            derivative_pairs != DfDerivativePairs::full && ga == gb,
                            shell_diagnostics));
                        ++signature_launches;
                      }
                    }
                  }
                  runtime::cuda_trace::trace_counter("three_center_primitive_signature_launches",
                                                     signature_launches);
                }
              } else {
                check(launch_df_shell_derivative_panel(
                    shell_o->view, shell_x->panel(range.offset, panel_count), r, range.offset,
                    panel_count, weights, derivative_output, shell_counters, arena.stream,
                    full_shell_domain, shell_variant, derivative_pairs, shell_diagnostics));
              }
              runtime::cuda_trace::trace_counter("three_center_shell_panels", 1);
            }
            if (metric_weights || !shell_execution || !full_shell_domain)
              check(launch_df_derivative_tile(o, x, r, kind, range, count, weights, schedule,
                                              derivative_output, arena.stream, 0, result.size(),
                                              gradient_copies, shell_execution && !metric_weights));
            ++arena.stats.tiles;
            arena.stats.device_response_bytes += count * sizeof(double);
          },
          borrowed, raw_a, packed_pairs,
          packed_pairs ? std::span<const std::int64_t>(shell_x->offsets)
                       : std::span<const std::int64_t>{},
          packed_block_rows));
      if (gradient_copies > 1) {
        runtime::cuda_trace::TraceRegion reduction("gradient_probe_shard_reduction", arena.stream);
        // Each column is already a complete contracted atom gradient, not an
        // AO-element or nuclear-coordinate derivative tensor. The shared BLAS
        // provider sums these bounded copies on the same owning stream.
        const double alpha = 1.0, beta = 0.0;
        const auto status = cublasDgemv(
            reinterpret_cast<cublasHandle_t>(blas_handle), CUBLAS_OP_N,
            static_cast<int>(result.size()), static_cast<int>(gradient_copies), &alpha,
            derivative_output, static_cast<int>(result.size()), ones, 1, &beta, output, 1);
        if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
      }
    } else {
      const auto weight_tile =
          std::min({std::size_t{65536}, std::max(n * n * a, a * a),
                    (maximum_bytes - arena.stats.device_bytes) / sizeof(double)});
      if (!weight_tile) throw std::bad_alloc();
      auto* weights = static_cast<double*>(arena.allocate(weight_tile * sizeof(double)));
      arena.stats.weight_tile_elements = weight_tile;
      preparation.finish();
      runtime::cuda_trace::TraceRegion host_weights("host_response_weights", arena.stream);
      const auto weight_stats = contract_density_fitting_response_weights(
          n, a, metric, inverse, terms, relative_threshold, maximum_bytes - arena.stats.host_bytes,
          maximum_auxiliary_tile,
          [&](std::size_t p, std::span<double> values) {
            // Source-backed execution requires device_metric above. This
            // compatibility adapter can only read caller-owned host values.
            runtime::cuda_trace::TraceRegion gather("host_raw_three_center_gather", arena.stream);
            runtime::cuda_trace::trace_counter("raw_value_cache_hits", 1);
            runtime::cuda_trace::trace_counter("raw_value_reuse_bytes", n * n * sizeof(double));
            for (std::size_t ij = 0; ij < n * n; ++ij) values[ij] = raw_a[ij * a + p];
          },
          [&](unsigned kind, runtime::StridedRange range, std::span<const double> host_weights) {
            runtime::cuda_trace::TraceRegion weight_uploads(
                "host_response_uploads_and_synchronization", arena.stream);
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
              arena.stats.response_host_to_device_bytes += count * sizeof(double);
              ++arena.stats.uploads;
              runtime::cuda_trace::TraceRegion derivatives(
                  kind ? "metric_center_derivative_contraction"
                       : "three_center_derivative_contraction",
                  arena.stream);
              runtime::cuda_trace::trace_counter(
                  kind ? "metric_derivative_weight_bytes" : "three_center_derivative_weight_bytes",
                  count * sizeof(double));
              check(launch_df_derivative_tile(o, x, r, kind, range, count, weights, schedule,
                                              output, arena.stream, begin));
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
    }
    runtime::cuda_trace::TraceRegion output_transfer("response_output_and_synchronization",
                                                     arena.stream);
    arena.stats.device_to_host_bytes += detailed_shell_work.readback_bytes;
    arena.stats.stream_synchronizations += detailed_shell_work.stream_drains;
    if (shell_counters) {
      check(cudaMemcpyAsync(observed_shell_work.data(), shell_counters, sizeof(observed_shell_work),
                            cudaMemcpyDeviceToHost, arena.stream));
      arena.stats.device_to_host_bytes += sizeof(observed_shell_work);
    }
    if (screen_counters) {
      check(cudaMemcpyAsync(observed_screen_work.data(), screen_counters,
                            sizeof(observed_screen_work), cudaMemcpyDeviceToHost, arena.stream));
      arena.stats.device_to_host_bytes += sizeof(observed_screen_work);
    }
    check(cudaMemcpyAsync(result.data(), output, result.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, arena.stream));
    arena.stats.device_to_host_bytes += result.size() * sizeof(double);
    check(cudaStreamSynchronize(arena.stream));
    ++arena.stats.stream_synchronizations;
    arena.completed = true;
    if (shell_counters) {
      constexpr const char* names[]{"shell_triples_visited",
                                    "shell_triples_nonzero",
                                    "shell_public_weights_nonzero",
                                    "shell_primitive_products",
                                    "shell_cartesian_component_products",
                                    "shell_public_weights_consumed"};
      for (unsigned i = 0; i < observed_shell_work.size(); ++i)
        runtime::cuda_trace::trace_counter(names[i], observed_shell_work[i]);
    }
    if (screen_counters) {
      runtime::cuda_trace::trace_counter("screening_000_primitives_considered",
                                         observed_screen_work[0]);
      runtime::cuda_trace::trace_counter("screening_000_primitives_skipped",
                                         observed_screen_work[1]);
      runtime::cuda_trace::trace_counter("screening_000_primitives_executed",
                                         observed_screen_work[0] - observed_screen_work[1]);
      runtime::cuda_trace::trace_counter("screening_000_shell_tasks_skipped",
                                         observed_screen_work[2]);
    }
    runtime::cuda_trace::trace_counter("host_to_device_bytes", arena.stats.host_to_device_bytes);
    runtime::cuda_trace::trace_counter("tensor_host_to_device_bytes",
                                       arena.stats.tensor_host_to_device_bytes);
    runtime::cuda_trace::trace_counter("response_host_to_device_bytes",
                                       arena.stats.response_host_to_device_bytes);
    runtime::cuda_trace::trace_counter("device_to_host_bytes", arena.stats.device_to_host_bytes);
    runtime::cuda_trace::trace_counter("stream_synchronizations",
                                       arena.stats.stream_synchronizations);
    runtime::cuda_trace::trace_counter("atom_coordinates", 3 * atoms);
    if (!std::all_of(result.begin(), result.end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("nonfinite generated DF-HF gradient");
    gradient.swap(result);
    if (resources) *resources = arena.stats;
    return VIBEQC_STATUS_SUCCESS;
  } catch (const CudaFailure& error) {
    detail = std::string("generated DF-HF CUDA failure: ") + cudaGetErrorString(error.status);
    return error.status == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                     : VIBEQC_STATUS_CUDA_ERROR;
  } catch (const CudaDfResponseBlasFailure& error) {
    detail = "generated DF-HF metric response cuBLAS failure (status " +
             std::to_string(static_cast<int>(error.status)) + ")";
    return error.status == CUBLAS_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
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

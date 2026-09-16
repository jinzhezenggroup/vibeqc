#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <type_traits>

#include "generated_df_rys_shell.cuh"
#include "molecule/basis.hpp"
#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_shell_derivatives.cuh"

namespace vibeqc::scf {
namespace {
namespace generated = generated_df_shell;
namespace scalar = generated_df_derivatives;
constexpr unsigned term_capacity = molecule::kMaximumAoExpansionTerms;

/** Publish one aggregated owner count; normal kernels receive a null sink. */
__device__ __forceinline__ void record_work(unsigned long long* work, DfShellWork kind,
                                            unsigned long long value) {
  if (work && value) atomicAdd(work + static_cast<unsigned>(kind), value);
}

cudaError_t clear_work(DfShellDiagnostics* diagnostics, unsigned rows, cudaStream_t stream) {
  if (!diagnostics) return cudaSuccess;
  if (!diagnostics->device || !diagnostics->host || rows > DfShellDiagnostics::packet_capacity)
    return cudaErrorInvalidValue;
  return cudaMemsetAsync(diagnostics->device, 0,
                         rows * DfShellDiagnostics::row_elements * sizeof(unsigned long long),
                         stream);
}

/** Read back only in explicitly intrusive diagnostics; no clean-path drain. */
cudaError_t read_work(DfShellDiagnostics* diagnostics, unsigned rows, cudaStream_t stream) {
  if (!diagnostics) return cudaSuccess;
  const auto bytes = rows * DfShellDiagnostics::row_elements * sizeof(unsigned long long);
  auto error = cudaMemcpyAsync(diagnostics->host, diagnostics->device, bytes,
                               cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error == cudaSuccess) {
    diagnostics->readback_bytes += bytes;
    ++diagnostics->stream_drains;
  }
  return error;
}

template <unsigned A, unsigned B, unsigned C>
void report_work(DfShellDiagnostics* diagnostics, unsigned row, std::size_t pa, std::size_t pb,
                 std::size_t pc) {
  if (!diagnostics) return;
  std::array<unsigned long long, DfShellDiagnostics::metrics> totals{};
  for (unsigned shard = 0; shard < DfShellDiagnostics::shards; ++shard)
    for (unsigned metric = 0; metric < totals.size(); ++metric)
      totals[metric] += diagnostics->host[row * DfShellDiagnostics::row_elements +
                                          shard * DfShellDiagnostics::metrics + metric];
  for (unsigned metric = 0; metric < totals.size(); ++metric) {
    char name[160];
    std::snprintf(name, sizeof(name), "shell_%u%u%u_work_%s", A, B, C, df_shell_work_names[metric]);
    runtime::cuda_trace::trace_counter(name, totals[metric]);
    std::snprintf(name, sizeof(name), "shell_%u%u%u_p%zu_%zu_%zu_work_%s", A, B, C, pa, pb, pc,
                  df_shell_work_names[metric]);
    runtime::cuda_trace::trace_counter(name, totals[metric]);
  }
  if (diagnostics->observer)
    diagnostics->observer(A, B, C, pa, pb, pc, totals, diagnostics->observer_context);
}

/** One generated subgroup owns a shell triple and contracts its public weights.
 * Sparse normalized AO expansions are applied to weights once, before any
 * primitive work. Partial auxiliary shells contribute only in-panel weights.
 * Every subgroup lane rendezvous uses its original mask, including lanes with
 * zero weights or no component. Independent groups may share a physical warp;
 * no barrier waits on another group's primitive count or panel clipping.
 * No derivative array escapes the subgroup.
 */
template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false>
__device__ __forceinline__ void contract_shell_task(
    DfShellBasisView first, DfShellBasisView second, DfShellBasisView auxiliary,
    const double* positions, std::size_t panel_begin, std::size_t panel_count,
    const double* weights, double* gradient, unsigned long long* counters, DfDerivativePairs pairs,
    bool triangle, std::size_t task, std::size_t tasks, unsigned long long* work) {
  using Math = std::conditional_t<Rys, generated::RysShell<A, B, C>, generated::Shell<A, B, C>>;
  using Schedule = generated::Schedule<A, B, C, Variant>;
  constexpr auto lanes = Schedule::lanes, groups = Schedule::groups;
  __shared__ double all_weights[groups][Math::components], all_cache[groups][3 * Math::axis_size];
  __shared__ scalar::Geometry all_geometry[groups];
  const unsigned lane = threadIdx.x % lanes, group = threadIdx.x / lanes;
  const unsigned mask = (0xffffffffU >> (32 - lanes)) << ((threadIdx.x % 32) / lanes * lanes);
  if (work) work += (blockIdx.x % DfShellDiagnostics::shards) * DfShellDiagnostics::metrics;
  auto* cart_weights = all_weights[group];
  auto* cache = all_cache[group];
  auto& geometry = all_geometry[group];
  if (task >= tasks) return;
  const auto sc = auxiliary.shell_ids[auxiliary.begin[C] + task % auxiliary.count[C]];
  const auto pair = task / auxiliary.count[C];
  auto ia_shell = pair / second.count[B], ib_shell = pair % second.count[B];
  if (triangle) {
    // Invert the compact triangular index, correcting roundoff at boundaries.
    ia_shell = static_cast<std::size_t>((sqrt(8.0 * pair + 1) - 1) * .5);
    while (ia_shell * (ia_shell + 1) / 2 > pair) --ia_shell;
    while ((ia_shell + 1) * (ia_shell + 2) / 2 <= pair) ++ia_shell;
    ib_shell = pair - ia_shell * (ia_shell + 1) / 2;
  }
  const auto sb = second.shell_ids[second.begin[B] + ib_shell];
  const auto sa = first.shell_ids[first.begin[A] + ia_shell];
  if (lane == 0 && counters) atomicAdd(counters, 1ULL);
  if (lane == 0) record_work(work, DfShellWork::shell_tasks, 1);
  const auto& o = first.basis;
  const auto& x = auxiliary.basis;
  const auto oa = first.ao_offsets[sa], ob = second.ao_offsets[sb];
  const auto oc = auxiliary.ao_offsets[sc];
  const auto na = first.ao_offsets[sa + 1] - oa;
  const auto nb = second.ao_offsets[sb + 1] - ob;
  const auto nc = auxiliary.ao_offsets[sc + 1] - oc;
  if (oc >= panel_begin + panel_count || oc + nc <= panel_begin) return;
  for (unsigned i = lane; i < Math::components; i += lanes) cart_weights[i] = 0;
  __syncwarp(mask);
  unsigned public_work = 0, public_loads = 0;
  unsigned long long expansion_work = 0, convolution_work = 0;
  for (auto i = std::int64_t{lane}; i < na * nb * nc; i += lanes) {
    const auto ai = oa + i / nb / nc, bi = ob + i / nc % nb, ci = oc + i % nc;
    if (ci < panel_begin || ci - panel_begin >= panel_count) continue;
    double weight;
    if (pairs == DfDerivativePairs::packed) {
      // A diagonal shell shares one physical center; its lower AO triangle
      // carries the complete folded response, including spherical expansions.
      if (sa == sb && ai < bi) continue;
      const auto hi = ai > bi ? ai : bi, lo = ai > bi ? bi : ai;
      weight = weights[(ci - panel_begin) * o.nbf * (o.nbf + 1) / 2 + hi * (hi + 1) / 2 + lo];
      ++public_loads;
    } else {
      const auto offset = (ci - panel_begin) * o.nbf * o.nbf;
      weight = weights[offset + ai * o.nbf + bi];
      ++public_loads;
      if (pairs == DfDerivativePairs::symmetric && sa != sb) {
        // Read both adjoints: exact also for nonsymmetric diagnostic weights.
        // Swapping orbital centers leaves the physical atomic derivative equal.
        weight += weights[offset + bi * o.nbf + ai];
        ++public_loads;
      }
    }
    if (weight == 0) continue;
    ++public_work;
    if (work)
      expansion_work += static_cast<unsigned long long>(o.term_counts[ai]) * o.term_counts[bi] *
                        x.term_counts[ci];
    for (unsigned at = 0; at < o.term_counts[ai]; ++at)
      for (unsigned bt = 0; bt < o.term_counts[bi]; ++bt)
        for (unsigned ct = 0; ct < x.term_counts[ci]; ++ct) {
          const auto ia = ai * term_capacity + at, ib = bi * term_capacity + bt;
          const auto ic = ci * term_capacity + ct;
          const auto ac = generated::cartesian_index(A, o.term_angular + 3 * ia);
          const auto bc = generated::cartesian_index(B, o.term_angular + 3 * ib);
          const auto cc = generated::cartesian_index(C, x.term_angular + 3 * ic);
          atomicAdd(
              cart_weights + (ac * Math::nb + bc) * Math::nc + cc,
              weight * o.term_coefficients[ia] * o.term_coefficients[ib] * x.term_coefficients[ic]);
        }
  }
  __syncwarp(mask);
  unsigned component_work = 0;
  for (unsigned i = lane; i < Math::components; i += lanes)
    if (cart_weights[i] != 0) {
      ++component_work;
      if (work) convolution_work += Math::convolution_work(i);
    }
  const bool active = __any_sync(mask, component_work != 0);
  for (unsigned delta = lanes / 2; delta; delta /= 2) {
    public_work += __shfl_down_sync(mask, public_work, delta, lanes);
    public_loads += __shfl_down_sync(mask, public_loads, delta, lanes);
    component_work += __shfl_down_sync(mask, component_work, delta, lanes);
    if (work) {
      expansion_work += __shfl_down_sync(mask, expansion_work, delta, lanes);
      convolution_work += __shfl_down_sync(mask, convolution_work, delta, lanes);
    }
  }
  if (lane == 0 && counters) {
    atomicAdd(counters + 2, static_cast<unsigned long long>(public_work));
    atomicAdd(counters + 5, static_cast<unsigned long long>(public_loads));
  }
  if (lane == 0) {
    record_work(work, DfShellWork::public_weight_loads, public_loads);
    record_work(work, DfShellWork::public_nonzero_weights, public_work);
    record_work(work, DfShellWork::expansion_term_products, expansion_work);
    record_work(work, DfShellWork::folding_shared_atomics, expansion_work);
    record_work(work, DfShellWork::subgroup_rendezvous, 2);
  }
  if (!active) return;
  const auto atom_a = o.shell_atoms[sa], atom_b = o.shell_atoms[sb], atom_c = x.shell_atoms[sc];
  const auto* ra = positions + 3 * atom_a;
  const auto* rb = positions + 3 * atom_b;
  const auto* rc = positions + 3 * atom_c;
  const scalar::Vec3 center_a{ra[0], ra[1], ra[2]}, center_b{rb[0], rb[1], rb[2]},
      center_c{rc[0], rc[1], rc[2]};
  double output[6]{};
  unsigned long long primitive_work = 0;
  unsigned long long series_iterations = 0, series = 0, small_argument = 0, large_argument = 0;
  // Launch-uniform loop lengths keep compact subgroups on the same primitive
  // iteration. Bounds are runtime parameters; the generated math is unchanged.
  const auto pa0 = o.primitive_offsets[sa], pb0 = o.primitive_offsets[sb];
  const auto pc0 = x.primitive_offsets[sc];
  const auto na_prim = first.primitives ? first.primitives : o.primitive_offsets[sa + 1] - pa0;
  const auto nb_prim = second.primitives ? second.primitives : o.primitive_offsets[sb + 1] - pb0;
  const auto nc_prim =
      auxiliary.primitives ? auxiliary.primitives : x.primitive_offsets[sc + 1] - pc0;
  for (std::size_t ia = 0; ia < na_prim; ++ia)
    for (std::size_t ib = 0; ib < nb_prim; ++ib)
      for (std::size_t ic = 0; ic < nc_prim; ++ic) {
        const auto pa = pa0 + ia, pb = pb0 + ib, pc = pc0 + ic;
        const double alpha = o.exponents[pa], beta = o.exponents[pb], gamma = x.exponents[pc];
        if (lane == 0) {
          if constexpr (Rys) {
            scalar::prepare_geometry_rys<Math::nroots>(alpha, center_a, beta, center_b, gamma,
                                                       center_c, A + B + C, geometry);
          } else if (work) {
            scalar::BoysWork observed;
            scalar::prepare_geometry(alpha, center_a, beta, center_b, gamma, center_c, A + B + C,
                                     geometry, &observed);
            series_iterations += observed.series_iterations;
            series += observed.series;
            small_argument += observed.small_argument;
            large_argument += observed.large_argument;
          } else
            scalar::prepare_geometry(alpha, center_a, beta, center_b, gamma, center_c, A + B + C,
                                     geometry);
          ++primitive_work;
        }
        __syncwarp(mask);
        if constexpr (!Rys) {
          Math::prepare(geometry, cache, lane, lanes);
          __syncwarp(mask);
        }
        for (unsigned i = lane; i < Math::components; i += lanes)
          if (cart_weights[i] != 0)
            Math::accumulate(
                i, alpha, beta, geometry, cache,
                cart_weights[i] * o.coefficients[pa] * o.coefficients[pb] * x.coefficients[pc],
                output);
        // The next primitive must not overwrite moments still being consumed.
        __syncwarp(mask);
      }
  for (unsigned i = 0; i < 6; ++i)
    for (unsigned delta = lanes / 2; delta; delta /= 2)
      output[i] += __shfl_down_sync(mask, output[i], delta, lanes);
  if (lane == 0) {
    const runtime::cuda_gaussian_products::Factor factors[3]{{o, oa}, {o, ob}, {x, oc}};
    runtime::cuda_gaussian_products::scatter(factors, Math::finish(output), 1.0, gradient);
    if (work) {
      record_work(work, DfShellWork::active_shell_tasks, 1);
      record_work(work, DfShellWork::primitive_products, primitive_work);
      record_work(work, DfShellWork::geometry_preparations, primitive_work);
      if constexpr (Rys) {
        record_work(work, DfShellWork::rys_evaluations, primitive_work);
        record_work(work, DfShellWork::rys_roots, Math::nroots * primitive_work);
        record_work(work, DfShellWork::recurrence_states,
                    Math::recurrence_states_per_primitive * primitive_work);
      } else {
        record_work(work, DfShellWork::boys_evaluations, primitive_work);
        record_work(work, DfShellWork::boys_order_sum, primitive_work * (A + B + C + 1));
        record_work(work, DfShellWork::boys_series_iterations, series_iterations);
        record_work(work, DfShellWork::boys_series, series);
        record_work(work, DfShellWork::boys_small_argument, small_argument);
        record_work(work, DfShellWork::boys_large_argument, large_argument);
      }
      record_work(work, DfShellWork::axis_polynomial_calls,
                  primitive_work * Math::polynomial_calls);
      record_work(work, DfShellWork::specialized_prepare_axis_calls,
                  primitive_work * Math::specialized_axis_calls);
      record_work(work, DfShellWork::cache_coefficient_values,
                  primitive_work * Math::cache_coefficient_values);
      record_work(work, DfShellWork::convolution_iterations, primitive_work * convolution_work);
      record_work(work, DfShellWork::active_component_products, primitive_work * component_work);
      record_work(work, DfShellWork::gradient_atomics_a, 3);
      record_work(work, DfShellWork::gradient_atomics_b, 3);
      record_work(work, DfShellWork::gradient_atomics_c, 3);
      const unsigned shared = 3 * (unsigned(atom_a == atom_b || atom_a == atom_c) +
                                   unsigned(atom_b == atom_a || atom_b == atom_c) +
                                   unsigned(atom_c == atom_a || atom_c == atom_b));
      record_work(work, DfShellWork::gradient_atomics_shared_atom, shared);
      record_work(work, DfShellWork::gradient_atomics_distinct_atom, 9 - shared);
      record_work(work, DfShellWork::subgroup_rendezvous, (Rys ? 2 : 3) * primitive_work);
    }
    if (counters) {
      atomicAdd(counters + 1, 1ULL);
      atomicAdd(counters + 3, primitive_work);
      atomicAdd(counters + 4, primitive_work * component_work);
    }
  }
}

/** One bounded descriptor names a homogeneous product, never individual triples. */
struct SignatureSlice {
  const std::int32_t *a_ids{}, *b_ids{}, *c_ids{};
  std::size_t a_begin{}, b_begin{}, c_begin{}, a_count{}, b_count{}, c_count{};
  std::size_t a_primitives{}, b_primitives{}, c_primitives{};
  std::size_t first_block{}, tasks{};
  bool triangle{};
};
struct SignaturePacket {
  static constexpr unsigned capacity = DfShellDiagnostics::packet_capacity;
  SignatureSlice slices[capacity]{};
  unsigned count{}, blocks{};
};
// Stay within the original CUDA 4-KiB kernel-argument limit, including views
// and scalar arguments. Driver-copied parameters need no mutable device queue.
static_assert(sizeof(SignaturePacket) + 2 * sizeof(DfShellBasisView) + 128 <= 4096);

template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false>
__global__ void shell_panel(DfShellBasisView first, DfShellBasisView second,
                            DfShellBasisView auxiliary, const double* positions, std::size_t begin,
                            std::size_t count, const double* weights, double* gradient,
                            unsigned long long* counters, DfDerivativePairs pairs, bool triangle,
                            std::size_t tasks, unsigned long long* work) {
  using Schedule = generated::Schedule<A, B, C, Variant>;
  const auto task = std::size_t{blockIdx.x} * Schedule::groups + threadIdx.x / Schedule::lanes;
  contract_shell_task<A, B, C, Variant, Rys>(first, second, auxiliary, positions, begin, count,
                                             weights, gradient, counters, pairs, triangle, task,
                                             tasks, work);
}

template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false>
__global__ void shell_packet(DfShellBasisView orbital, DfShellBasisView auxiliary,
                             const double* positions, std::size_t begin, std::size_t count,
                             const double* weights, double* gradient, unsigned long long* counters,
                             DfDerivativePairs pairs,
                             const __grid_constant__ SignaturePacket packet,
                             unsigned long long* work) {
  using Schedule = generated::Schedule<A, B, C, Variant>;
  // A block owns only one signature. Prefixing block counts pools small slices
  // in a single grid without reintroducing mixed primitive lengths in a warp.
  unsigned low = 0, high = packet.count;
  while (low + 1 < high) {
    const auto middle = (low + high) / 2;
    if (packet.slices[middle].first_block <= blockIdx.x)
      low = middle;
    else
      high = middle;
  }
  // grid_constant prevents an address-taken packet from becoming a private
  // multi-KiB local copy in every thread. All fields remain read-only parameters.
  const auto& slice = packet.slices[low];
  auto first = orbital, second = orbital, third = auxiliary;
  first.shell_ids = slice.a_ids;
  second.shell_ids = slice.b_ids;
  third.shell_ids = slice.c_ids;
  first.begin[A] = slice.a_begin;
  second.begin[B] = slice.b_begin;
  third.begin[C] = slice.c_begin;
  first.count[A] = slice.a_count;
  second.count[B] = slice.b_count;
  third.count[C] = slice.c_count;
  first.primitives = slice.a_primitives;
  second.primitives = slice.b_primitives;
  third.primitives = slice.c_primitives;
  const auto task = (std::size_t{blockIdx.x} - slice.first_block) * Schedule::groups +
                    threadIdx.x / Schedule::lanes;
  contract_shell_task<A, B, C, Variant, Rys>(
      first, second, third, positions, begin, count, weights, gradient, counters, pairs,
      slice.triangle, task, slice.tasks,
      work ? work + low * DfShellDiagnostics::row_elements : nullptr);
}

/** Resource values are maxima within an operation, not sums over its launches. */
template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false, class Kernel>
cudaError_t profile_resources(Kernel kernel) {
  using Schedule = generated::Schedule<A, B, C, Variant>;
  cudaFuncAttributes attributes{};
  int blocks = 0, device = 0, maximum_threads = 0;
  auto error = cudaFuncGetAttributes(&attributes, kernel);
  if (error != cudaSuccess) return error;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel,
                                                        Schedule::lanes * Schedule::groups, 0);
  if (error != cudaSuccess) return error;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  error = cudaDeviceGetAttribute(&maximum_threads, cudaDevAttrMaxThreadsPerMultiProcessor, device);
  if (error != cudaSuccess) return error;
  const auto resource = [&](const char* suffix, std::size_t value) {
    char name[96];
    std::snprintf(name, sizeof(name), "shell_%u%u%u_%s", A, B, C, suffix);
    runtime::cuda_trace::trace_maximum(name, value);
  };
  runtime::cuda_trace::trace_maximum("shell_resource_values_are_maxima", 1);
  resource("rys_selected", Rys);
  resource("registers", attributes.numRegs);
  resource("static_shared_bytes", attributes.sharedSizeBytes);
  resource("dynamic_shared_bytes", 0);
  resource("resident_block_limit", blocks);
  resource("resident_thread_limit", blocks * Schedule::lanes * Schedule::groups);
  resource("sm_thread_limit", maximum_threads);
  return cudaSuccess;
}

template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false>
cudaError_t launch_packets(std::span<const DfShellBasisView> orbital,
                           std::span<const DfShellBasisView> auxiliary, const double* positions,
                           std::size_t begin, std::size_t count, const double* weights,
                           double* gradient, unsigned long long* counters, cudaStream_t stream,
                           DfDerivativePairs pairs, DfShellDiagnostics* diagnostics) {
  if (pairs != DfDerivativePairs::full && A < B) return cudaSuccess;
  using Schedule = generated::Schedule<A, B, C, Variant>;
  constexpr auto maximum = static_cast<std::size_t>(std::numeric_limits<int>::max());
  const char* trace = std::getenv("VIBEQC_DF_TRACE");
  const bool profiling = trace && *trace;
  using Clock = std::chrono::steady_clock;
  auto preparation = profiling ? Clock::now() : Clock::time_point{};
  SignaturePacket packet;
  const auto flush = [&]() -> cudaError_t {
    if (!packet.count) return cudaSuccess;
    // Start expensive, often small signatures first so their final blocks can
    // overlap the larger cheap slices. Ranking changes scheduling only.
    std::sort(packet.slices, packet.slices + packet.count,
              [](const SignatureSlice& a, const SignatureSlice& b) {
                const long double wa =
                    static_cast<long double>(a.a_primitives) * a.b_primitives * a.c_primitives;
                const long double wb =
                    static_cast<long double>(b.a_primitives) * b.b_primitives * b.c_primitives;
                return wa != wb ? wa > wb : a.first_block < b.first_block;
              });
    packet.blocks = 0;
    for (unsigned i = 0; i < packet.count; ++i) {
      packet.slices[i].first_block = packet.blocks;
      packet.blocks +=
          static_cast<unsigned>((packet.slices[i].tasks + Schedule::groups - 1) / Schedule::groups);
    }
    if (profiling) {
      // Stop the preparation clock before driver submission/backpressure.
      runtime::cuda_trace::trace_counter(
          "signature_packet_preparation_ns",
          std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - preparation).count());
      const auto error =
          profile_resources<A, B, C, Variant, Rys>(shell_packet<A, B, C, Variant, Rys>);
      if (error != cudaSuccess) return error;
    }
    char name[64]{};
    if (profiling) std::snprintf(name, sizeof(name), "shell_%u%u%u_packet", A, B, C);
    auto error = clear_work(diagnostics, packet.count, stream);
    if (error != cudaSuccess) return error;
    {
      runtime::cuda_trace::TraceRegion region(name, stream);
      shell_packet<A, B, C, Variant, Rys>
          <<<packet.blocks, Schedule::lanes * Schedule::groups, 0, stream>>>(
              orbital.front(), auxiliary.front(), positions, begin, count, weights, gradient,
              counters, pairs, packet, diagnostics ? diagnostics->device : nullptr);
    }
    error = cudaPeekAtLastError();
    if (error == cudaSuccess) error = read_work(diagnostics, packet.count, stream);
    if (error != cudaSuccess) return error;
    for (unsigned row = 0; diagnostics && row < packet.count; ++row) {
      const auto& slice = packet.slices[row];
      report_work<A, B, C>(diagnostics, row, slice.a_primitives, slice.b_primitives,
                           slice.c_primitives);
    }
    runtime::cuda_trace::trace_counter("three_center_signature_packet_launches", 1);
    runtime::cuda_trace::trace_counter("signature_packet_slices", packet.count);
    runtime::cuda_trace::trace_counter("signature_packet_argument_bytes", sizeof(packet));
    runtime::cuda_trace::trace_maximum("signature_packet_payload_bytes", sizeof(packet));
    packet.count = packet.blocks = 0;
    if (profiling) preparation = Clock::now();
    return error;
  };
  for (std::size_t ga = 0; ga < orbital.size(); ++ga) {
    const auto& first = orbital[ga];
    if (!first.count[A]) continue;
    for (std::size_t gb = 0; gb < orbital.size(); ++gb) {
      const auto& second = orbital[gb];
      if (!second.count[B] || (pairs != DfDerivativePairs::full && A == B && ga < gb)) continue;
      const bool triangle = pairs != DfDerivativePairs::full && A == B && ga == gb;
      for (const auto& third : auxiliary) {
        if (!third.count[C]) continue;
        if (first.count[A] > maximum / second.count[B] ||
            first.count[A] * second.count[B] > maximum / third.count[C])
          return cudaErrorInvalidValue;
        const auto pair_count =
            triangle ? first.count[A] * (first.count[A] + 1) / 2 : first.count[A] * second.count[B];
        const auto tasks = pair_count * third.count[C];
        const auto blocks = (tasks + Schedule::groups - 1) / Schedule::groups;
        if (packet.count == SignaturePacket::capacity || blocks > maximum - packet.blocks) {
          const auto error = flush();
          if (error != cudaSuccess) return error;
        }
        packet.slices[packet.count++] = {
            first.shell_ids,   second.shell_ids, third.shell_ids, first.begin[A], second.begin[B],
            third.begin[C],    first.count[A],   second.count[B], third.count[C], first.primitives,
            second.primitives, third.primitives, packet.blocks,   tasks,          triangle};
        packet.blocks += static_cast<unsigned>(blocks);
        if (profiling) {
          char name[128];
          std::snprintf(name, sizeof(name), "shell_%u%u%u_p%zu_%zu_%zu", A, B, C, first.primitives,
                        second.primitives, third.primitives);
          runtime::cuda_trace::trace_counter(name, tasks);
        }
      }
    }
  }
  const auto error = flush();
  if (profiling)
    runtime::cuda_trace::trace_counter(
        "signature_packet_preparation_ns",
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - preparation).count());
  return error;
}

template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false>
cudaError_t launch_group(DfShellBasisView first, DfShellBasisView second, DfShellBasisView x,
                         const double* positions, std::size_t begin, std::size_t count,
                         const double* weights, double* gradient, unsigned long long* counters,
                         cudaStream_t stream, DfDerivativePairs pairs, bool triangle,
                         DfShellDiagnostics* diagnostics) {
  if (!first.count[A] || !second.count[B] || !x.count[C]) return cudaSuccess;
  if (triangle &&
      (pairs == DfDerivativePairs::full || A != B || first.count[A] != second.count[B] ||
       first.shell_ids != second.shell_ids || first.begin[A] != second.begin[B]))
    return cudaErrorInvalidValue;
  const auto maximum = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (first.count[A] > maximum / second.count[B] ||
      first.count[A] * second.count[B] > maximum / x.count[C])
    return cudaErrorInvalidValue;
  using Schedule = generated::Schedule<A, B, C, Variant>;
  const auto shell_pairs =
      triangle ? first.count[A] * (first.count[A] + 1) / 2 : first.count[A] * second.count[B];
  const auto tasks = shell_pairs * x.count[C];
  // Class/signature timings and resources are diagnostic only. CUDA event
  // intervals may include stream idle time; use Nsight for GPU activity and
  // a separate uninstrumented endpoint for promotion.
  char profile_name[128]{};
  const char* tracing = std::getenv("VIBEQC_DF_TRACE");
  if (tracing && *tracing) {
    std::snprintf(profile_name, sizeof(profile_name), "shell_%u%u%u_p%zu_%zu_%zu", A, B, C,
                  first.primitives, second.primitives, x.primitives);
    runtime::cuda_trace::trace_counter(profile_name, tasks);
    const auto error = profile_resources<A, B, C, Variant, Rys>(shell_panel<A, B, C, Variant, Rys>);
    if (error != cudaSuccess) return error;
  }
  auto error = clear_work(diagnostics, 1, stream);
  if (error != cudaSuccess) return error;
  runtime::cuda_trace::TraceRegion profile(profile_name, stream);
  if (tasks)
    shell_panel<A, B, C, Variant, Rys>
        <<<static_cast<unsigned>((tasks + Schedule::groups - 1) / Schedule::groups),
           Schedule::lanes * Schedule::groups, 0, stream>>>(
            first, second, x, positions, begin, count, weights, gradient, counters, pairs, triangle,
            tasks, diagnostics ? diagnostics->device : nullptr);
  error = cudaPeekAtLastError();
  profile.finish();
  if (error == cudaSuccess) error = read_work(diagnostics, 1, stream);
  if (error == cudaSuccess)
    report_work<A, B, C>(diagnostics, 0, first.primitives, second.primitives, x.primitives);
  return error;
}

template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false>
cudaError_t launch(DfShellBasisView o, DfShellBasisView x, const double* positions,
                   std::size_t begin, std::size_t count, const double* weights, double* gradient,
                   unsigned long long* counters, cudaStream_t stream, DfDerivativePairs pairs,
                   DfShellDiagnostics* diagnostics) {
  if (pairs != DfDerivativePairs::full && A < B) return cudaSuccess;
  return launch_group<A, B, C, Variant, Rys>(
      o, o, x, positions, begin, count, weights, gradient, counters, stream, pairs,
      pairs != DfDerivativePairs::full && A == B, diagnostics);
}
/** Select only the generated 000 class. Automatic promotion stays disabled
 * until complete endpoint qualification. No weight/scheduling policy changes.
 */
template <unsigned A, unsigned B, unsigned C, class Launch>
cudaError_t dispatch_lowering(Launch&& launch) {
  const char* raw = std::getenv("VIBEQC_DF_SHELL_MATH_000");
  const std::string_view policy = raw ? raw : "auto";
  if (policy != "auto" && policy != "polynomial" && policy != "rys") return cudaErrorInvalidValue;
  if constexpr (generated::rys_available<A, B, C>) {
    if (policy == "rys") return launch.template operator()<true>();
  }
  return launch.template operator()<false>();
}
}  // namespace

cudaError_t launch_df_shell_derivative_panel(DfShellBasisView o, DfShellBasisView x,
                                             const double* positions, std::size_t begin,
                                             std::size_t count, const double* weights,
                                             double* gradient, unsigned long long* counters,
                                             cudaStream_t stream, bool full_domain,
                                             unsigned variant, DfDerivativePairs pairs,
                                             DfShellDiagnostics* diagnostics) {
  if (variant > 2) return cudaErrorInvalidValue;
  cudaError_t status = cudaSuccess;
  generated::for_each_class([&]<unsigned A, unsigned B, unsigned C>() {
    if (status != cudaSuccess || (!full_domain && (A > 1 || B > 1 || C > 1 || A + B + C == 0)))
      return;
    status = dispatch_lowering<A, B, C>([&]<bool Rys>() {
      if (variant == 0)
        return launch<A, B, C, 0, Rys>(o, x, positions, begin, count, weights, gradient, counters,
                                       stream, pairs, diagnostics);
      else if (variant == 1)
        return launch<A, B, C, 1, Rys>(o, x, positions, begin, count, weights, gradient, counters,
                                       stream, pairs, diagnostics);
      else
        return launch<A, B, C, 2, Rys>(o, x, positions, begin, count, weights, gradient, counters,
                                       stream, pairs, diagnostics);
    });
  });
  return status;
}

cudaError_t launch_df_shell_derivative_group(
    DfShellBasisView first, DfShellBasisView second, DfShellBasisView x, const double* positions,
    std::size_t begin, std::size_t count, const double* weights, double* gradient,
    unsigned long long* counters, cudaStream_t stream, bool full_domain, unsigned variant,
    DfDerivativePairs pairs, bool triangle, DfShellDiagnostics* diagnostics) {
  if (variant > 2) return cudaErrorInvalidValue;
  cudaError_t status = cudaSuccess;
  generated::for_each_class([&]<unsigned A, unsigned B, unsigned C>() {
    if (status != cudaSuccess || (!full_domain && (A > 1 || B > 1 || C > 1 || A + B + C == 0)))
      return;
    status = dispatch_lowering<A, B, C>([&]<bool Rys>() {
      if (variant == 0)
        return launch_group<A, B, C, 0, Rys>(first, second, x, positions, begin, count, weights,
                                             gradient, counters, stream, pairs, triangle,
                                             diagnostics);
      else if (variant == 1)
        return launch_group<A, B, C, 1, Rys>(first, second, x, positions, begin, count, weights,
                                             gradient, counters, stream, pairs, triangle,
                                             diagnostics);
      else
        return launch_group<A, B, C, 2, Rys>(first, second, x, positions, begin, count, weights,
                                             gradient, counters, stream, pairs, triangle,
                                             diagnostics);
    });
  });
  return status;
}

cudaError_t launch_df_shell_derivative_packets(
    std::span<const DfShellBasisView> orbital, std::span<const DfShellBasisView> auxiliary,
    const double* positions, std::size_t begin, std::size_t count, const double* weights,
    double* gradient, unsigned long long* counters, cudaStream_t stream, bool full_domain,
    unsigned variant, DfDerivativePairs pairs, DfShellDiagnostics* diagnostics) {
  if (variant > 2) return cudaErrorInvalidValue;
  if (orbital.empty() || auxiliary.empty()) return cudaSuccess;
  cudaError_t status = cudaSuccess;
  generated::for_each_class([&]<unsigned A, unsigned B, unsigned C>() {
    if (status != cudaSuccess || (!full_domain && (A > 1 || B > 1 || C > 1 || A + B + C == 0)))
      return;
    status = dispatch_lowering<A, B, C>([&]<bool Rys>() {
      if (variant == 0)
        return launch_packets<A, B, C, 0, Rys>(orbital, auxiliary, positions, begin, count, weights,
                                               gradient, counters, stream, pairs, diagnostics);
      else if (variant == 1)
        return launch_packets<A, B, C, 1, Rys>(orbital, auxiliary, positions, begin, count, weights,
                                               gradient, counters, stream, pairs, diagnostics);
      else
        return launch_packets<A, B, C, 2, Rys>(orbital, auxiliary, positions, begin, count, weights,
                                               gradient, counters, stream, pairs, diagnostics);
    });
  });
  return status;
}

}  // namespace vibeqc::scf

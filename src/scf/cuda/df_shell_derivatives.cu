#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <type_traits>

#include "generated_df_production.hpp"
#include "molecule/basis.hpp"
#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_shell_derivatives.cuh"
#include "scf/cuda/df_shell_kernel.cuh"

namespace vibeqc::scf {
namespace {
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

/** Resource values are maxima within an operation, not sums over its launches. */
template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false,
          bool Screening = false, class Kernel>
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

template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false,
          bool Screening = false>
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
      const auto error = profile_resources<A, B, C, Variant, Rys, Screening>(
          shell_packet<A, B, C, Variant, Rys, Screening>);
      if (error != cudaSuccess) return error;
    }
    char name[64]{};
    if (profiling) std::snprintf(name, sizeof(name), "shell_%u%u%u_packet", A, B, C);
    auto error = clear_work(diagnostics, packet.count, stream);
    if (error != cudaSuccess) return error;
    {
      runtime::cuda_trace::TraceRegion region(name, stream);
      shell_packet<A, B, C, Variant, Rys, Screening>
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

template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false,
          bool Screening = false>
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
    const auto error = profile_resources<A, B, C, Variant, Rys, Screening>(
        shell_panel<A, B, C, Variant, Rys, Screening>);
    if (error != cudaSuccess) return error;
  }
  auto error = clear_work(diagnostics, 1, stream);
  if (error != cudaSuccess) return error;
  runtime::cuda_trace::TraceRegion profile(profile_name, stream);
  if (tasks)
    shell_panel<A, B, C, Variant, Rys, Screening>
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

template <unsigned A, unsigned B, unsigned C, unsigned Variant, bool Rys = false,
          bool Screening = false>
cudaError_t launch(DfShellBasisView o, DfShellBasisView x, const double* positions,
                   std::size_t begin, std::size_t count, const double* weights, double* gradient,
                   unsigned long long* counters, cudaStream_t stream, DfDerivativePairs pairs,
                   DfShellDiagnostics* diagnostics) {
  if (pairs != DfDerivativePairs::full && A < B) return cudaSuccess;
  return launch_group<A, B, C, Variant, Rys, Screening>(
      o, o, x, positions, begin, count, weights, gradient, counters, stream, pairs,
      pairs != DfDerivativePairs::full && A == B, diagnostics);
}
/** Query a target once per host packet call; no dynamic tuning table is read.
 * Only the measured resident 384/768 domains admit automatic promotion. The
 * explicit candidate control is for complete endpoint qualification builds.
 */
cudaError_t production_target(DfShellBasisView o, DfShellBasisView x, unsigned& architecture) {
  architecture = 0;
  const char* control = std::getenv("VIBEQC_DF_SHELL_POLICY");
  const std::string_view policy = control ? control : "auto";
  if (policy != "auto" && policy != "legacy" && policy != "candidate") return cudaErrorInvalidValue;
  if (!generated::production_policy_available || policy == "legacy") return cudaSuccess;
  if (policy == "auto" &&
      ((o.basis.nbf != 384 && o.basis.nbf != 768) || x.basis.nbf != o.basis.nbf))
    return cudaSuccess;
  int device = 0;
  cudaDeviceProp properties{};
  auto error = cudaGetDevice(&device);
  if (error == cudaSuccess) error = cudaGetDeviceProperties(&properties, device);
  if (error == cudaSuccess) architecture = 10 * properties.major + properties.minor;
  return error;
}

template <unsigned A, unsigned B, unsigned C, bool Rys, class Launch>
cudaError_t dispatch_screening(double budget, Launch&& launch) {
  if constexpr (A + B + C == 0)
    if (budget > 0) return launch.template operator()<Rys, true>();
  return launch.template operator()<Rys, false>();
}

template <unsigned A, unsigned B, unsigned C, class Launch>
cudaError_t dispatch_lowering(unsigned architecture, unsigned& variant, double budget,
                              Launch&& launch) {
  const auto choice = generated::DfProductionPolicy<A, B, C>::select(architecture);
  const char* mapping = std::getenv("VIBEQC_DF_SHELL_POLICY");
  const bool use_choice =
      choice.available &&
      (choice.qualified || (mapping && std::string_view(mapping) == "candidate"));
  const char* schedule = std::getenv("VIBEQC_DF_SHELL_SCHEDULE");
  if (use_choice && (!schedule || std::string_view(schedule) == "auto")) variant = choice.variant;
  const char* raw = std::getenv("VIBEQC_DF_SHELL_MATH_000");
  const std::string_view policy = raw ? raw : "auto";
  if (policy != "auto" && policy != "polynomial" && policy != "rys") return cudaErrorInvalidValue;
  if constexpr (generated::rys_available<A, B, C>) {
    if (policy == "rys" || (policy == "auto" && use_choice && choice.rys))
      return dispatch_screening<A, B, C, true>(budget, launch);
  }
  return dispatch_screening<A, B, C, false>(budget, launch);
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
  unsigned architecture = 0;
  auto status = production_target(o, x, architecture);
  if (status != cudaSuccess) return status;
  generated::for_each_class([&]<unsigned A, unsigned B, unsigned C>() {
    if (status != cudaSuccess || (!full_domain && (A > 1 || B > 1 || C > 1 || A + B + C == 0)))
      return;
    unsigned class_variant = variant;
    status = dispatch_lowering<A, B, C>(
        architecture, class_variant, o.force_screen_budget, [&]<bool Rys, bool Screening>() {
          if (class_variant == 0)
            return launch<A, B, C, 0, Rys, Screening>(o, x, positions, begin, count, weights,
                                                      gradient, counters, stream, pairs,
                                                      diagnostics);
          else if (class_variant == 1)
            return launch<A, B, C, 1, Rys, Screening>(o, x, positions, begin, count, weights,
                                                      gradient, counters, stream, pairs,
                                                      diagnostics);
          else
            return launch<A, B, C, 2, Rys, Screening>(o, x, positions, begin, count, weights,
                                                      gradient, counters, stream, pairs,
                                                      diagnostics);
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
  unsigned architecture = 0;
  auto status = production_target(first, x, architecture);
  if (status != cudaSuccess) return status;
  generated::for_each_class([&]<unsigned A, unsigned B, unsigned C>() {
    if (status != cudaSuccess || (!full_domain && (A > 1 || B > 1 || C > 1 || A + B + C == 0)))
      return;
    unsigned class_variant = variant;
    status = dispatch_lowering<A, B, C>(
        architecture, class_variant, first.force_screen_budget, [&]<bool Rys, bool Screening>() {
          if (class_variant == 0)
            return launch_group<A, B, C, 0, Rys, Screening>(first, second, x, positions, begin,
                                                            count, weights, gradient, counters,
                                                            stream, pairs, triangle, diagnostics);
          else if (class_variant == 1)
            return launch_group<A, B, C, 1, Rys, Screening>(first, second, x, positions, begin,
                                                            count, weights, gradient, counters,
                                                            stream, pairs, triangle, diagnostics);
          else
            return launch_group<A, B, C, 2, Rys, Screening>(first, second, x, positions, begin,
                                                            count, weights, gradient, counters,
                                                            stream, pairs, triangle, diagnostics);
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
  unsigned architecture = 0;
  auto status = production_target(orbital.front(), auxiliary.front(), architecture);
  if (status != cudaSuccess) return status;
  generated::for_each_class([&]<unsigned A, unsigned B, unsigned C>() {
    if (status != cudaSuccess || (!full_domain && (A > 1 || B > 1 || C > 1 || A + B + C == 0)))
      return;
    unsigned class_variant = variant;
    status = dispatch_lowering<A, B, C>(
        architecture, class_variant, orbital.front().force_screen_budget,
        [&]<bool Rys, bool Screening>() {
          if (class_variant == 0)
            return launch_packets<A, B, C, 0, Rys, Screening>(orbital, auxiliary, positions, begin,
                                                              count, weights, gradient, counters,
                                                              stream, pairs, diagnostics);
          else if (class_variant == 1)
            return launch_packets<A, B, C, 1, Rys, Screening>(orbital, auxiliary, positions, begin,
                                                              count, weights, gradient, counters,
                                                              stream, pairs, diagnostics);
          else
            return launch_packets<A, B, C, 2, Rys, Screening>(orbital, auxiliary, positions, begin,
                                                              count, weights, gradient, counters,
                                                              stream, pairs, diagnostics);
        });
  });
  return status;
}

}  // namespace vibeqc::scf

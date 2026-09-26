#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <type_traits>

#include "molecule/basis.hpp"
#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_shell_derivatives.cuh"
#include "scf/cuda/df_shell_dispatch.cuh"
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
    // Start expensive homogeneous signatures first. The shared runtime packet
    // planner owns stable profitability ordering and bounded block prefixes;
    // DF owns only the legal signature descriptors and their primitive work.
    runtime::order_homogeneous_task_packet(packet, [](const SignatureSlice& slice) {
      return static_cast<long double>(slice.a_primitives) * slice.b_primitives * slice.c_primitives;
    });
    if (!runtime::finalize_homogeneous_task_packet(packet, Schedule::groups, maximum))
      return cudaErrorInvalidValue;
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
template <unsigned A, unsigned B, unsigned C, bool Rys, class Launch>
cudaError_t dispatch_screening(double budget, Launch&& launch) {
  if constexpr (A + B + C == 0)
    if (budget > 0) return launch.template operator()<Rys, true>();
  return launch.template operator()<Rys, false>();
}

/** Instantiate all supported schedules, independent of production-policy bytes. */
template <unsigned A, unsigned B, unsigned C, class Launch>
cudaError_t dispatch_class(const DfShellLaunch& context, double budget, Launch&& launch) {
  auto variant = [&]<bool Rys, bool Screening>() -> cudaError_t {
    if (context.variant == 0) return launch.template operator()<0, Rys, Screening>();
    if (context.variant == 1) return launch.template operator()<1, Rys, Screening>();
    if (context.variant == 2) return launch.template operator()<2, Rys, Screening>();
    return cudaErrorInvalidValue;
  };
  if constexpr (generated::rys_available<A, B, C>)
    if (context.rys) return dispatch_screening<A, B, C, true>(budget, variant);
  return dispatch_screening<A, B, C, false>(budget, variant);
}

template <unsigned A, unsigned B, unsigned C>
cudaError_t launch_panel_class(DfShellBasisView o, DfShellBasisView x, const DfShellLaunch& c) {
  return dispatch_class<A, B, C>(
      c, o.force_screen_budget, [&]<unsigned Variant, bool Rys, bool Screening>() {
        return launch<A, B, C, Variant, Rys, Screening>(o, x, c.positions, c.begin, c.count,
                                                        c.weights, c.gradient, c.counters, c.stream,
                                                        c.pairs, c.diagnostics);
      });
}

template <unsigned A, unsigned B, unsigned C>
cudaError_t launch_group_class(DfShellBasisView first, DfShellBasisView second, DfShellBasisView x,
                               bool triangle, const DfShellLaunch& c) {
  return dispatch_class<A, B, C>(
      c, first.force_screen_budget, [&]<unsigned Variant, bool Rys, bool Screening>() {
        return launch_group<A, B, C, Variant, Rys, Screening>(
            first, second, x, c.positions, c.begin, c.count, c.weights, c.gradient, c.counters,
            c.stream, c.pairs, triangle, c.diagnostics);
      });
}

template <unsigned A, unsigned B, unsigned C>
cudaError_t launch_packets_class(std::span<const DfShellBasisView> o,
                                 std::span<const DfShellBasisView> x, const DfShellLaunch& c) {
  return dispatch_class<A, B, C>(
      c, o.front().force_screen_budget, [&]<unsigned Variant, bool Rys, bool Screening>() {
        return launch_packets<A, B, C, Variant, Rys, Screening>(o, x, c.positions, c.begin, c.count,
                                                                c.weights, c.gradient, c.counters,
                                                                c.stream, c.pairs, c.diagnostics);
      });
}
}  // namespace
}  // namespace vibeqc::scf

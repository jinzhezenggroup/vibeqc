// Validation adapter only: supplied orbital frames come from independent fixtures.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>

#include "cc/df_source.hpp"
#include "generated_df_cc_source_cpu.hpp"
#include "posthf/raw_source.hpp"

extern "C" int df_cc_source_probe(void* opaque, const double* coefficients, std::size_t occupied,
                                  std::size_t budget, double* output, std::size_t elements,
                                  std::size_t* counts, double* values, char* error,
                                  std::size_t error_size) noexcept {
  try {
    if (!opaque || !coefficients || !output || !counts || !values)
      throw std::invalid_argument("invalid DF source probe pointers");
    const auto& source = *static_cast<generativeqc::posthf::RawSource*>(opaque);
    const auto n = source.nbf(), q = source.naux();
    if (!occupied || occupied >= n)
      throw std::invalid_argument("invalid DF source probe occupation");
    const auto layout =
        generativeqc::cc::generated::df_source::source_layout(occupied, n - occupied, q);
    if (elements != layout.output_values)
      throw std::invalid_argument("invalid DF source probe output size");
    generativeqc::hf::PhysicalReference ref;
    ref.nbf = n;
    ref.nocc = occupied;
    ref.coefficients.assign(coefficients, coefficients + n * n);
    const auto result = generativeqc::cc::build_df_source_cuda(source.orbital(), source.auxiliary(),
                                                               ref, budget, 1e-10, 0);
    auto* cursor = output;
    for (const auto* block : {&result.boo, &result.bov, &result.bvv, &result.ovov, &result.ovvo,
                              &result.oovv, &result.ovoo, &result.oooo})
      cursor = std::copy(block->begin(), block->end(), cursor);
    const std::size_t c[] = {
        result.numeric_capacity_bytes, result.host_output_bytes,    result.source_rows,
        result.source_values,          result.transform_gemms,      result.block_gemms,
        result.transform_summands,     result.block_summands,       result.coefficient_h2d_bytes,
        result.factor_block_d2h_bytes, result.metric_staging_bytes, result.metric_rank};
    std::copy(std::begin(c), std::end(c), counts);
    const double t[] = {result.metric_absolute_threshold,
                        result.metric_condition_number,
                        result.source_seconds,
                        result.metric_seconds,
                        result.transform_seconds,
                        result.block_seconds,
                        result.total_seconds};
    std::copy(std::begin(t), std::end(t), values);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}

// Geometry -> native conventional CUDA RHF -> native DF source -> native CCSD.
// This internal qualification entry never substitutes a supplied/oracle frame.
#include "methods/rccsd_method.hpp"
#include "runtime/execution_context.hpp"

extern "C" int df_cc_molecular_probe(void* opaque, std::size_t budget, double* values, char* error,
                                     std::size_t error_size) noexcept {
  try {
    if (!opaque || !values) throw std::invalid_argument("invalid native molecular probe");
    const auto& source = *static_cast<generativeqc::posthf::RawSource*>(opaque);
    generativeqc::core::ContextState context;
    context.requested_backend = GENERATIVEQC_BACKEND_CUDA;
    generativeqc::runtime::ExecutionContext execution(context);
    generativeqc_method_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.method = GENERATIVEQC_METHOD_RCCSD;
    descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
    descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
    descriptor.ccsd_diis_history = 6;
    descriptor.ccsd_energy_tolerance = 1e-12;
    descriptor.ccsd_residual_tolerance = 1e-10;
    descriptor.correlation_memory_budget_bytes = budget;
    const auto result = generativeqc::methods::detail::run_rccsd_native_state(
        execution, source.orbital(), descriptor, nullptr, nullptr, nullptr, 0, &source.auxiliary());
    if (!result.solved.converged())
      throw std::runtime_error("native molecular DF CCSD did not converge");
    const double data[] = {result.reference->energy,
                           result.solved.correlation_energy,
                           result.solved.total_energy,
                           static_cast<double>(result.solved.diagnostic.iterations),
                           result.solved.diagnostic.replay_r1_max,
                           result.solved.diagnostic.replay_r2_max,
                           static_cast<double>(result.diagnostic.numeric_capacity_bytes),
                           result.performance.reference_seconds,
                           result.performance.problem_seconds,
                           result.performance.solver_seconds,
                           static_cast<double>(result.performance.source_values),
                           static_cast<double>(result.problem.naux),
                           static_cast<double>(result.solved.diagnostic.df_contraction_terms),
                           static_cast<double>(result.solved.diagnostic.iteration_graph_calls),
                           static_cast<double>(result.solved.diagnostic.replay_graph_calls),
                           static_cast<double>(result.solved.diagnostic.df_auxiliary_slices),
                           static_cast<double>(result.performance.transform_fmas),
                           static_cast<double>(result.performance.source_reads),
                           static_cast<double>(result.diagnostic.mo_transfer_bytes),
                           static_cast<double>(result.diagnostic.correlation_owned_device_bytes)};
    std::copy(std::begin(data), std::end(data), values);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}

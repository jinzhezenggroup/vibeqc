#include "methods/dft_method.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "api/handles.hpp"
#include "dft/ao_grid.hpp"
#include "dft/dispersion/d4_runtime.hpp"
#include "dft/grid.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "dft/semilocal_family.hpp"
#include "generated_method_parameters.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/types.hpp"
#include "vibeqc/vibeqc.hpp"

#if VIBEQC_HAS_CUDA
#include "dft/cuda_ks.hpp"
#include "scf/cuda_direct_jk.hpp"
#endif

namespace vibeqc::methods::detail {
namespace {

std::uint64_t next_cpu_ks_owner() {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  do {
    if (value == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CPU KS owner identity exhausted");
  } while (!next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
  return value;
}

struct NativeKsExecutionPlan {
  std::uint32_t spin_channels{1};
  dft::SemilocalFamily semilocal_family{dft::SemilocalFamily::Lda};
  bool compiler_resolved{};
  bool d4_correction{};
  bool nonlocal_correlation{};
  dft::nlc::Vv10Parameters nonlocal_parameters{};
  std::uint64_t nonlocal_maximum_bytes{};
  bool range_exchange{};
  double short_range_exchange{};
  double long_range_exchange{};
  double range_omega{};
};

std::optional<NativeKsExecutionPlan> legacy_ks_execution_plan(vibeqc_method method) noexcept {
  switch (method) {
    case VIBEQC_METHOD_LDA_RKS:
      return NativeKsExecutionPlan{1, dft::SemilocalFamily::Lda, false};
    case VIBEQC_METHOD_LDA_UKS:
      return NativeKsExecutionPlan{2, dft::SemilocalFamily::Lda, false};
    case VIBEQC_METHOD_PBE_D4_RKS:
      return NativeKsExecutionPlan{1, dft::SemilocalFamily::Pbe, false, true};
    case VIBEQC_METHOD_PBE_RKS:
    case VIBEQC_METHOD_PBE0_RKS:
      return NativeKsExecutionPlan{1, dft::SemilocalFamily::Pbe, false};
    case VIBEQC_METHOD_PBE_UKS:
    case VIBEQC_METHOD_PBE0_UKS:
      return NativeKsExecutionPlan{2, dft::SemilocalFamily::Pbe, false};
    case VIBEQC_METHOD_B3LYP_RKS:
      return NativeKsExecutionPlan{1, dft::SemilocalFamily::B3lyp, false};
    case VIBEQC_METHOD_B3LYP_UKS:
      return NativeKsExecutionPlan{2, dft::SemilocalFamily::B3lyp, false};
    case VIBEQC_METHOD_WB97M_V:
      return NativeKsExecutionPlan{1, dft::SemilocalFamily::Wb97mv, false};
    case VIBEQC_METHOD_WB97M_V_UKS:
      return NativeKsExecutionPlan{2, dft::SemilocalFamily::Wb97mv, false};
    case VIBEQC_METHOD_R2SCAN_RKS:
      return NativeKsExecutionPlan{1, dft::SemilocalFamily::R2scan, false};
    case VIBEQC_METHOD_R2SCAN_UKS:
      return NativeKsExecutionPlan{2, dft::SemilocalFamily::R2scan, false};
    default:
      return std::nullopt;
  }
}

bool unrestricted(const NativeKsExecutionPlan& plan) noexcept { return plan.spin_channels == 2; }

std::uint32_t scf_domain_version(const NativeKsExecutionPlan& plan) noexcept {
  return dft::semilocal_family_domain_version(plan.semilocal_family);
}

const char* semilocal_family_name(const NativeKsExecutionPlan& plan) noexcept {
  return dft::semilocal_family_name(plan.semilocal_family);
}

std::optional<double> semilocal_component(const vibeqc_ks_options& input,
                                          std::string_view component_id) {
  std::optional<double> value;
  for (std::uint32_t i = 0; i < input.semilocal_component_count; ++i) {
    const auto& term = input.semilocal_components[i];
    if (!term.component_id || !*term.component_id || !std::isfinite(term.coefficient) ||
        term.coefficient < 0.0)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "invalid KS semilocal component");
    if (std::string_view(term.component_id) == component_id) {
      if (value)
        throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "duplicate KS semilocal component");
      value = term.coefficient;
    }
  }
  return value;
}

bool has_only_semilocal_components(const vibeqc_ks_options& input,
                                   std::initializer_list<std::string_view> ids) {
  if (input.semilocal_component_count != ids.size()) return false;
  for (const auto id : ids)
    if (!semilocal_component(input, id)) return false;
  return true;
}

struct SemilocalAdmission {
  dft::SemilocalFamily family{dft::SemilocalFamily::Lda};
  double exchange_scale{1.0};
  double correlation_scale{1.0};
};

SemilocalAdmission admit_semilocal(const vibeqc_ks_options& input) {
  if (!input.semilocal_components || !input.semilocal_component_count)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "KS execution plan requires semilocal components");
  if (!std::isfinite(input.semilocal_range_omega) || input.semilocal_range_omega < 0.0)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "invalid semilocal range parameter");

  if (has_only_semilocal_components(input, {"LDA_X", "LDA_C_PW"}) &&
      *semilocal_component(input, "LDA_X") == 1.0 &&
      *semilocal_component(input, "LDA_C_PW") == 1.0 && input.semilocal_range_omega == 0.0)
    return {dft::SemilocalFamily::Lda, 1.0, 1.0};

  if (has_only_semilocal_components(input, {"GGA_X_PBE", "GGA_C_PBE"}) &&
      input.semilocal_range_omega == 0.0)
    return {dft::SemilocalFamily::Pbe, *semilocal_component(input, "GGA_X_PBE"),
            *semilocal_component(input, "GGA_C_PBE")};

  if (has_only_semilocal_components(input, {"MGGA_X_R2SCAN", "MGGA_C_R2SCAN"}) &&
      *semilocal_component(input, "MGGA_X_R2SCAN") == 1.0 &&
      *semilocal_component(input, "MGGA_C_R2SCAN") == 1.0 && input.semilocal_range_omega == 0.0)
    return {dft::SemilocalFamily::R2scan, 1.0, 1.0};

  if (has_only_semilocal_components(input, {"LDA_X", "GGA_X_B88", "LDA_C_VWN_RPA", "GGA_C_LYP"}) &&
      *semilocal_component(input, "LDA_X") == 0.08 &&
      *semilocal_component(input, "GGA_X_B88") == 0.72 &&
      *semilocal_component(input, "LDA_C_VWN_RPA") == 0.19 &&
      *semilocal_component(input, "GGA_C_LYP") == 0.81 && input.semilocal_range_omega == 0.0)
    return {dft::SemilocalFamily::B3lyp, 1.0, 1.0};

  if (has_only_semilocal_components(input, {"MGGA_X_WB97M_V", "MGGA_C_WB97M_V"}) &&
      *semilocal_component(input, "MGGA_X_WB97M_V") == 1.0 &&
      *semilocal_component(input, "MGGA_C_WB97M_V") == 1.0 && input.semilocal_range_omega == 0.3)
    return {dft::SemilocalFamily::Wb97mv, 1.0, 1.0};

  throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                    "KS semilocal primitive graph has no qualified native lowerer");
}

std::string_view expected_scf_domain(const NativeKsExecutionPlan& plan) noexcept {
  if (plan.semilocal_family == dft::SemilocalFamily::Wb97mv)
    return "libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16";
  if (plan.semilocal_family == dft::SemilocalFamily::B3lyp)
    return "b3lyp-vwn-rpa-tail-v1/density-vacuum-1e-18";
  return "semilocal-scaled-v1/pbe-spin-c2-1e-18";
}

scf::ScfOptions dft_options(const vibeqc_method_descriptor& descriptor, vibeqc_backend backend,
                            NativeKsExecutionPlan& execution_plan) {
  if (!std::isfinite(descriptor.energy_tolerance) || !std::isfinite(descriptor.density_tolerance) ||
      !std::isfinite(descriptor.screening_tolerance))
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "DFT tolerances must be finite");
  scf::ScfOptions options;
  options.max_iterations = descriptor.max_iterations == 0 ? 100 : descriptor.max_iterations;
  options.diis_history = descriptor.diis_history == 0 ? 8 : descriptor.diis_history;
  options.energy_tolerance =
      descriptor.energy_tolerance > 0.0 ? descriptor.energy_tolerance : 1.0e-10;
  options.density_tolerance =
      descriptor.density_tolerance > 0.0 ? descriptor.density_tolerance : 1.0e-8;
  options.screening_tolerance =
      descriptor.screening_tolerance > 0.0 ? descriptor.screening_tolerance : 1.0e-12;

  const auto legacy_plan = legacy_ks_execution_plan(descriptor.method);
  const vibeqc_ks_options* ks_input = nullptr;
  SemilocalAdmission semilocal;
  if (descriptor.ks_options) {
    ks_input = descriptor.ks_options;
    if (ks_input->struct_size < sizeof(vibeqc_ks_options) ||
        ks_input->abi_version != VIBEQC_ABI_VERSION)
      throw MethodError(VIBEQC_STATUS_ABI_MISMATCH, "KS execution-plan ABI mismatch");
    if ((ks_input->spin_channels != 1 && ks_input->spin_channels != 2) ||
        (ks_input->exchange_terms == nullptr) != (ks_input->exchange_term_count == 0))
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "invalid compiler KS execution plan");
    semilocal = admit_semilocal(*ks_input);
    execution_plan = {ks_input->spin_channels, semilocal.family, true,
                      descriptor.method == VIBEQC_METHOD_PBE_D4_RKS};
  } else {
    if (!legacy_plan)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "DFT execution requires a compiler-resolved KS plan");
    if (descriptor.method == VIBEQC_METHOD_PBE0_RKS ||
        descriptor.method == VIBEQC_METHOD_PBE0_UKS ||
        descriptor.method == VIBEQC_METHOD_B3LYP_RKS ||
        descriptor.method == VIBEQC_METHOD_B3LYP_UKS)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "hybrid DFT requires a compiler-resolved KS plan");
    execution_plan = *legacy_plan;
    semilocal = {execution_plan.semilocal_family, 1.0, 1.0};
  }

  const auto mode = descriptor.density_fitting_mode;
  if (mode != VIBEQC_DENSITY_FITTING_NONE && mode != VIBEQC_DENSITY_FITTING_CPU_REFERENCE &&
      mode != VIBEQC_DENSITY_FITTING_CUDA && mode != VIBEQC_DENSITY_FITTING_AUTO)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown density-fitting execution mode");
  options.density_fitting_mode = mode;
  if ((mode == VIBEQC_DENSITY_FITTING_CPU_REFERENCE && backend != VIBEQC_BACKEND_CPU_REFERENCE) ||
      (mode == VIBEQC_DENSITY_FITTING_CUDA && backend != VIBEQC_BACKEND_CUDA))
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "DFT density-fitting backend must match the calculation backend");
  if (descriptor.density_fitting_auxiliary_basis != nullptr &&
      options.density_fitting_mode == VIBEQC_DENSITY_FITTING_NONE)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                      "DFT auxiliary basis requires density fitting");
  if (descriptor.density_fitting_relative_threshold != 0.0)
    options.density_fitting_relative_threshold = descriptor.density_fitting_relative_threshold;
  options.density_fitting_memory_budget_bytes = descriptor.density_fitting_memory_budget_bytes;
  if (descriptor.precision_mode != VIBEQC_PRECISION_FP64 &&
      descriptor.precision_mode != VIBEQC_PRECISION_AUTO)
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown floating-point precision mode");
  if (descriptor.precision_mode == VIBEQC_PRECISION_AUTO && backend != VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "DFT automatic precision currently requires CUDA");
  if (descriptor.precision_mode == VIBEQC_PRECISION_AUTO && execution_plan.d4_correction)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "PBE-D4 currently requires strict FP64");
  if (descriptor.precision_mode == VIBEQC_PRECISION_AUTO &&
      (execution_plan.semilocal_family == dft::SemilocalFamily::R2scan ||
       execution_plan.semilocal_family == dft::SemilocalFamily::Wb97mv))
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "meta-GGA DFT currently requires strict FP64");
  options.precision_mode = descriptor.precision_mode;

  scf::FockBuildSpec fock;
  fock.spin =
      unrestricted(execution_plan) ? scf::FockSpin::Unrestricted : scf::FockSpin::Restricted;
  fock.derivative_order = 0;
  fock.exchange.present = false;
  options.semilocal_exchange_scale = semilocal.exchange_scale;
  options.semilocal_correlation_scale = semilocal.correlation_scale;

  const vibeqc_ks_exchange_term* full_range = nullptr;
  const vibeqc_ks_exchange_term* short_range = nullptr;
  const vibeqc_ks_exchange_term* long_range = nullptr;
  if (ks_input) {
    const double divisor = unrestricted(execution_plan) ? 1.0 : 2.0;
    for (std::uint32_t i = 0; i < ks_input->exchange_term_count; ++i) {
      const auto& term = ks_input->exchange_terms[i];
      if (!std::isfinite(term.coefficient) || term.coefficient < 0.0 ||
          !std::isfinite(term.omega) || term.omega < 0.0 || !std::isfinite(term.fock_coefficient) ||
          term.fock_coefficient != -term.coefficient / divisor)
        throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "invalid KS exact-exchange contribution");
      switch (term.operator_kind) {
        case VIBEQC_KS_EXCHANGE_FULL_RANGE:
          if (full_range || term.omega != 0.0)
            throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "invalid full-range exchange plan");
          full_range = &term;
          break;
        case VIBEQC_KS_EXCHANGE_SHORT_RANGE:
          if (short_range || term.omega <= 0.0)
            throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "invalid short-range exchange plan");
          short_range = &term;
          break;
        case VIBEQC_KS_EXCHANGE_LONG_RANGE:
          if (long_range || term.omega <= 0.0)
            throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "invalid long-range exchange plan");
          long_range = &term;
          break;
        default:
          throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown KS exchange operator");
      }
    }
    if (full_range && (short_range || long_range))
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "KS execution cannot mix full- and range-separated exchange");
    if ((short_range == nullptr) != (long_range == nullptr))
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "range-separated exchange requires short- and long-range terms");
    if (short_range &&
        (ks_input->exchange_term_count != 2 || short_range->omega != long_range->omega))
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "range-separated exchange requires one shared omega");
    if (full_range && ks_input->exchange_term_count != 1)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "full-range exchange requires one contribution");
    if (full_range) {
      fock.exchange.present = full_range->coefficient != 0.0;
      fock.exchange.coefficient = full_range->fock_coefficient;
    } else if (short_range) {
      execution_plan.range_exchange = true;
      execution_plan.short_range_exchange = short_range->coefficient;
      execution_plan.long_range_exchange = long_range->coefficient;
      execution_plan.range_omega = short_range->omega;
      fock.exchange.present = short_range->coefficient != 0.0;
      fock.exchange.coefficient = short_range->fock_coefficient;
    }

    if (ks_input->has_nonlocal_correlation > 1)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "invalid nonlocal-correlation presence flag");
    if (ks_input->has_nonlocal_correlation) {
      dft::nlc::Vv10Variant variant;
      if (ks_input->nonlocal_variant == VIBEQC_NONLOCAL_VV10)
        variant = dft::nlc::Vv10Variant::vv10;
      else if (ks_input->nonlocal_variant == VIBEQC_NONLOCAL_RVV10)
        variant = dft::nlc::Vv10Variant::rvv10;
      else
        throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                          "invalid KS nonlocal-correlation variant");
      if (!std::isfinite(ks_input->nonlocal_b) || ks_input->nonlocal_b <= 0.0 ||
          !std::isfinite(ks_input->nonlocal_c) || ks_input->nonlocal_c <= 0.0 ||
          !std::isfinite(ks_input->nonlocal_coefficient) || ks_input->nonlocal_coefficient <= 0.0 ||
          !ks_input->nonlocal_maximum_bytes)
        throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                          "invalid KS nonlocal-correlation parameters or budget");
      execution_plan.nonlocal_correlation = true;
      execution_plan.nonlocal_parameters = {variant, ks_input->nonlocal_b, ks_input->nonlocal_c,
                                            ks_input->nonlocal_coefficient};
      execution_plan.nonlocal_maximum_bytes = ks_input->nonlocal_maximum_bytes;
    }
  }

  const bool complete_wb97mv = execution_plan.semilocal_family == dft::SemilocalFamily::Wb97mv &&
                               execution_plan.range_exchange && execution_plan.nonlocal_correlation;
  const bool cuda_wb97mv = backend == VIBEQC_BACKEND_CUDA && complete_wb97mv;
  const bool scaled_or_hybrid = options.semilocal_exchange_scale != 1.0 ||
                                options.semilocal_correlation_scale != 1.0 || fock.exchange.present;
  const double pbe0_fock_coefficient = fock.spin == scf::FockSpin::Restricted ? -0.125 : -0.25;
  const double b3lyp_fock_coefficient = fock.spin == scf::FockSpin::Restricted ? -0.1 : -0.2;
  const bool strict_cuda_global_hybrid =
      backend == VIBEQC_BACKEND_CUDA && !execution_plan.range_exchange &&
      !execution_plan.nonlocal_correlation &&
      options.density_fitting_mode == VIBEQC_DENSITY_FITTING_NONE &&
      options.precision_mode != VIBEQC_PRECISION_AUTO && fock.exchange.present;
  const bool cuda_pbe0 =
      strict_cuda_global_hybrid && execution_plan.semilocal_family == dft::SemilocalFamily::Pbe &&
      options.semilocal_exchange_scale == 0.75 && options.semilocal_correlation_scale == 1.0 &&
      fock.exchange.coefficient == pbe0_fock_coefficient;
  const bool cuda_b3lyp =
      strict_cuda_global_hybrid && execution_plan.semilocal_family == dft::SemilocalFamily::B3lyp &&
      options.semilocal_exchange_scale == 1.0 && options.semilocal_correlation_scale == 1.0 &&
      fock.exchange.coefficient == b3lyp_fock_coefficient;
  if (scaled_or_hybrid && backend == VIBEQC_BACKEND_CUDA && !cuda_pbe0 && !cuda_b3lyp &&
      !cuda_wb97mv)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "CUDA scaled/global-hybrid KS composition is not qualified");
  if (execution_plan.nonlocal_correlation &&
      execution_plan.semilocal_family != dft::SemilocalFamily::Pbe &&
      execution_plan.semilocal_family != dft::SemilocalFamily::Wb97mv)
    throw MethodError(
        VIBEQC_STATUS_NOT_IMPLEMENTED,
        "self-consistent nonlocal correlation has no lowerer for this semilocal graph");
  if (execution_plan.nonlocal_correlation && backend != VIBEQC_BACKEND_CPU_REFERENCE &&
      !cuda_wb97mv)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "CUDA self-consistent nonlocal correlation is qualified only for WB97M-V");
  if (options.precision_mode == VIBEQC_PRECISION_AUTO && execution_plan.nonlocal_correlation)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "self-consistent nonlocal correlation currently requires strict FP64");
  if (execution_plan.range_exchange && backend != VIBEQC_BACKEND_CPU_REFERENCE && !cuda_wb97mv)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "CUDA range-separated KS is qualified only for complete WB97M-V");
  if (execution_plan.range_exchange &&
      execution_plan.semilocal_family != dft::SemilocalFamily::Pbe &&
      execution_plan.semilocal_family != dft::SemilocalFamily::Wb97mv)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "native KS range exchange has no lowerer for this semilocal graph");
  if (options.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE) {
    if (options.precision_mode == VIBEQC_PRECISION_AUTO || execution_plan.range_exchange ||
        execution_plan.nonlocal_correlation)
      throw MethodError(
          VIBEQC_STATUS_NOT_IMPLEMENTED,
          "DFT density fitting requires FP64 full-range local/semilocal or global-hybrid KS");
    fock.coulomb.approximation = scf::FockApproximation::DensityFitted;
    if (fock.exchange.present) fock.exchange.approximation = scf::FockApproximation::DensityFitted;
  }
  options.resolved_fock_build = scf::resolve_fock_build(
      fock, backend == VIBEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu,
      options.screening_tolerance, options.density_fitting_relative_threshold);
  if (execution_plan.semilocal_family == dft::SemilocalFamily::Wb97mv) {
    if (!execution_plan.range_exchange || !execution_plan.nonlocal_correlation)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "WB97M-V requires complete B97M + SR/LR + VV10 primitives");
    const auto correction_backend =
        backend == VIBEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
    const auto correction =
        scf::resolve_fock_build(scf::make_rsh_correction_fock_spec(
                                    fock.spin, execution_plan.short_range_exchange,
                                    execution_plan.long_range_exchange, execution_plan.range_omega),
                                correction_backend, options.screening_tolerance);
    scf::require_wb97mv_composition(*options.resolved_fock_build, correction,
                                    execution_plan.nonlocal_parameters);
  }
  options.compute_forces = false;
  return options;
}

/** Copy every pointee before constructing scientific owners. The public entry
 * has already admitted a complete current descriptor. A null KS-options pointer
 * retains the existing default GridSpec and tile values; production callers pass
 * the compiler-resolved grid, not a second native production profile. */
dft::GridSpec ks_grid_options(const vibeqc_method_descriptor& descriptor, scf::ScfOptions& options,
                              const NativeKsExecutionPlan& execution_plan) {
  dft::GridSpec grid;
  if (!descriptor.ks_options) return grid;
  const auto& input = *descriptor.ks_options;
  if (input.struct_size < sizeof(vibeqc_ks_options) || input.abi_version != VIBEQC_ABI_VERSION)
    throw MethodError(VIBEQC_STATUS_ABI_MISMATCH, "KS execution-plan ABI mismatch");
  if (!input.scf_domain ||
      std::string_view(input.scf_domain) != expected_scf_domain(execution_plan))
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "unsupported KS tail/spin domain policy");
  if (!input.tile_points || input.tile_points > static_cast<std::uint64_t>(INT_MAX))
    throw std::invalid_argument("invalid KS XC tile points");
  options.xc_tile_points = input.tile_points;
  switch (input.xc_execution_schedule) {
    case VIBEQC_XC_EXECUTION_DEVICE_FUSED:
      options.xc_execution_schedule = scf::ScfOptions::XcExecutionSchedule::DeviceFused;
      break;
    case VIBEQC_XC_EXECUTION_HOST_UNFUSED:
      options.xc_execution_schedule = scf::ScfOptions::XcExecutionSchedule::HostUnfused;
      break;
    default:
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "unknown KS XC execution schedule");
  }
  grid.version = input.grid_version;
  grid.radial_points = input.radial_points;
  grid.angular_polar = input.angular_polar;
  grid.angular_azimuth = input.angular_azimuth;
  grid.partition_iterations = input.partition_iterations;
  grid.coincident_tolerance = input.coincident_tolerance;
  if ((input.element_radii == nullptr) != (input.element_radius_count == 0) ||
      (input.element_radii && input.element_radius_count != grid.element_radii.size()))
    throw std::invalid_argument("KS element radii require 119 entries or NULL/zero");
  if (input.element_radii) {
    for (std::size_t z = 1; z < grid.element_radii.size(); ++z) {
      const double radius = input.element_radii[z];
      if (!std::isfinite(radius) || (grid.version == 1 ? radius <= 0.0 : radius < 0.0))
        throw std::invalid_argument("invalid KS element radius");
      grid.element_radii[z] = grid.version == 1 && radius == 1.0 ? 0.0 : radius;
    }
  }
  dft::validate_grid_spec(grid);
  return grid;
}

Result adapt_result(scf::ScfResult native, vibeqc_backend backend) {
  Result result;
  result.energy = native.energy;
  result.convergence.iterations = native.iterations;
  result.convergence.energy_change = native.energy_change;
  result.convergence.residual_rms = native.density_rms;
  result.physical_residual_rms = native.physical_residual_rms;
  result.convergence.converged = native.converged;
  result.executed_backend = backend;
  result.fock_builds = native.fock_builds;
  result.precision = native.precision;
  native.dft_diagnostic.fock_builds = native.fock_builds;
  native.dft_diagnostic.initial_density_used = native.initial_density_used;
  // Move the snapshot instead of retaining another max-iteration history.
  result.ks_diagnostic = std::move(native.dft_diagnostic);
  return result;
}

/** The method's global ledger supplies the budget. Size the common direct
 * source explicitly so its standalone default cap is not a second KS limit. */
std::size_t ks_provider_bytes(const core::System& system, vibeqc_backend backend) {
#if VIBEQC_HAS_CUDA
  if (backend == VIBEQC_BACKEND_CUDA) {
    std::size_t primitives = 0;
    for (const auto& shell : system.shells)
      primitives = runtime::add_capacity(primitives, shell.primitives.size());
    return scf::cuda_direct_coulomb_device_bytes(1, molecule::ao_count(system), system.atoms.size(),
                                                 system.shells.size(), primitives);
  }
#endif
  return 0;
}

#if VIBEQC_HAS_CUDA
KsTransportDiagnostic adapt_transfers(const dft::CudaKsTransfers& value) {
  return {value.setup_h2d_bytes,
          value.density_h2d_bytes,
          value.scalar_d2h_bytes,
          value.matrix_d2h_bytes,
          value.final_state_d2h_bytes,
          value.final_state_reads,
          value.synchronizations,
          value.iterations,
          value.occupation_stabilized_proposals};
}

void add_transfers(dft::CudaKsTransfers& target, const dft::CudaKsTransfers& value) {
  const auto add = [](std::uint64_t& destination, std::uint64_t increment) {
    if (increment > std::numeric_limits<std::uint64_t>::max() - destination)
      throw std::overflow_error("CUDA KS transport counter overflow");
    destination += increment;
  };
  add(target.setup_h2d_bytes, value.setup_h2d_bytes);
  add(target.density_h2d_bytes, value.density_h2d_bytes);
  add(target.scalar_d2h_bytes, value.scalar_d2h_bytes);
  add(target.matrix_d2h_bytes, value.matrix_d2h_bytes);
  add(target.final_state_d2h_bytes, value.final_state_d2h_bytes);
  add(target.final_state_reads, value.final_state_reads);
  add(target.synchronizations, value.synchronizations);
  add(target.iterations, value.iterations);
  add(target.occupation_stabilized_proposals, value.occupation_stabilized_proposals);
}
#endif

/** Own auxiliary shells and rebind centers when a batch item moves. The source
 * owner copies the result, so descriptor/temporary system lifetimes never leak
 * into a prepared calculation. An omitted auxiliary basis means the orbital basis. */
std::optional<core::System> ks_auxiliary_template(const vibeqc_method_descriptor& descriptor) {
  if (!descriptor.density_fitting_auxiliary_basis) return std::nullopt;
  return descriptor.density_fitting_auxiliary_basis->data;
}

void validate_ks_auxiliary_geometry(const core::System& system,
                                    const std::optional<core::System>& auxiliary) {
  if (!auxiliary) return;
  if (auxiliary->atoms.size() != system.atoms.size())
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "DFT auxiliary basis atom count differs");
  for (std::size_t i = 0; i < system.atoms.size(); ++i)
    if (auxiliary->atoms[i].atomic_number != system.atoms[i].atomic_number ||
        auxiliary->atoms[i].position != system.atoms[i].position)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "DFT auxiliary basis must share the initial system geometry");
}

std::optional<core::System> ks_auxiliary_for_system(const core::System& system,
                                                    std::optional<core::System> auxiliary) {
  if (!auxiliary) return auxiliary;
  if (auxiliary->atoms.size() != system.atoms.size())
    throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT, "DFT auxiliary basis atom count differs");
  for (std::size_t i = 0; i < system.atoms.size(); ++i) {
    if (auxiliary->atoms[i].atomic_number != system.atoms[i].atomic_number)
      throw MethodError(VIBEQC_STATUS_INVALID_ARGUMENT,
                        "DFT auxiliary basis atom ordering differs");
    auxiliary->atoms[i].position = system.atoms[i].position;
  }
  return auxiliary;
}

/** Backend selection must precede materialization: constructing the reference
 * grid and then uploading it hides cubic host work in CUDA preparation. */
dft::MolecularGrid ks_molecular_grid(const core::System& system, dft::GridSpec spec,
                                     vibeqc_backend backend, int device) {
  if (backend == VIBEQC_BACKEND_CUDA) {
#if VIBEQC_HAS_CUDA
    return dft::MolecularGrid::from_cuda(system, spec, device);
#else
    throw std::runtime_error("CUDA quadrature is unavailable in this build");
#endif
  }
  return dft::MolecularGrid(system, spec);
}

class KsPreparedCalculation final : public PreparedCalculation {
 public:
  KsPreparedCalculation(Capabilities capabilities, core::System system,
                        NativeKsExecutionPlan execution_plan, scf::ScfOptions options,
                        dft::GridSpec grid, vibeqc_backend backend, int device,
                        const std::optional<core::System>& auxiliary)
      : capabilities_(capabilities),
        system_(std::move(system)),
        execution_plan_(execution_plan),
        options_(std::move(options)),
        backend_(backend),
        fock_(system_, auxiliary ? &*auxiliary : nullptr, *options_.resolved_fock_build, device,
              options_.density_fitting_mode == VIBEQC_DENSITY_FITTING_NONE
                  ? ks_provider_bytes(system_, backend)
                  : options_.density_fitting_memory_budget_bytes),
        basis_(system_),
        grid_(ks_molecular_grid(system_, grid, backend_, device)) {
    options_.retain_ks_state = backend_ != VIBEQC_BACKEND_CUDA;
    if (execution_plan_.range_exchange) prepare_range_exchange(device);
    if (execution_plan_.nonlocal_correlation) prepare_nonlocal(device);
#if VIBEQC_HAS_CUDA
    if (backend_ == VIBEQC_BACKEND_CUDA) {
      if (execution_plan_.semilocal_family == dft::SemilocalFamily::Wb97mv &&
          options_.xc_execution_schedule != scf::ScfOptions::XcExecutionSchedule::DeviceFused)
        throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                          "public CUDA WB97M-V requires device-fused XC/nonlocal execution");
      const auto* range = range_strategy_ ? &*range_strategy_ : nullptr;
      const auto domain = execution_plan_.semilocal_family == dft::SemilocalFamily::Wb97mv
                              ? dft::nlc::Vv10DensityDomain::MolecularV1
                              : dft::nlc::Vv10DensityDomain::StrictPositive;
      cuda_ = std::make_unique<dft::CudaKsPlan>(
          fock_, basis_, grid_, options_, execution_plan_.semilocal_family, options_.xc_tile_points,
          range, nonlocal_.get(), domain);
    }
#endif
    if (execution_plan_.d4_correction) prepare_d4(device);
    runtime::sample_cpu_capacity(host_numeric_capacity());
  }

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }
  const core::System& system() const noexcept { return system_; }
  const std::vector<scf::CudaDensityFittingMetricDiagnostic>& fitted_diagnostics() const noexcept {
    return fock_.diagnostic().fitted;
  }

  /** Explicit retained vectors; object metadata and transient setup are not
   * inferred from this lower-bound observation. Grid/basis buffers are owned. */
  std::size_t host_numeric_capacity() const noexcept {
    auto bytes =
        runtime::add_capacity(fock_.cpu_observation_capacity(),
                              runtime::vector_capacities(basis_.packed, grid_.points(),
                                                         grid_.weights(), grid_.owners(), warm_));
    if (range_correction_)
      bytes = runtime::add_capacity(bytes, range_correction_->cpu_observation_capacity());
    if (cpu_physical_)
      for (const auto* matrices : {&cpu_physical_->density, &cpu_physical_->fock})
        for (const auto& matrix : *matrices)
          bytes = runtime::add_capacity(bytes, runtime::vector_bytes(matrix));
#if VIBEQC_HAS_CUDA
    if (cuda_) bytes = runtime::add_capacity(bytes, cuda_->resources().retained_host_numeric_bytes);
#endif
    if (d4_) {
      const auto& resources = d4_->resources();
      bytes = runtime::add_capacity(bytes, static_cast<std::size_t>(resources.plan_host_bytes));
      bytes =
          runtime::add_capacity(bytes, static_cast<std::size_t>(resources.execution_host_bytes));
    }
    if (nonlocal_) {
      const auto& resources = nonlocal_->resources();
      bytes =
          runtime::add_capacity(bytes, static_cast<std::size_t>(resources.host_workspace_bytes));
    }
    return bytes;
  }

  /** Explicit output/rebuild export. Ordinary CUDA replays keep this on device. */
  std::vector<double> warm_density() {
#if VIBEQC_HAS_CUDA
    if (cuda_) return cuda_->warm_density();
#endif
    return warm_;
  }

  void clear_warm_start() noexcept {
    warm_.clear();
#if VIBEQC_HAS_CUDA
    if (cuda_) cuda_->clear_warm_start();
#endif
  }

  void invalidate_result() override { invalidate_final_state(); }

  void invalidate_final_state() noexcept {
    cpu_physical_.reset();
#if VIBEQC_HAS_CUDA
    if (cuda_) cuda_->invalidate_final_state();
#endif
  }

#if VIBEQC_HAS_CUDA
  dft::CudaKsPlan* cuda_plan() noexcept { return cuda_.get(); }
#endif

  std::optional<KsTransportDiagnostic> ks_transport_diagnostic() const override {
#if VIBEQC_HAS_CUDA
    if (cuda_) return adapt_transfers(cuda_->transfers());
#endif
    return std::nullopt;
  }

  vibeqc_status final_state_token(dft::CudaKsFinalStateToken& token, std::string& detail) const {
#if VIBEQC_HAS_CUDA
    if (cuda_) return cuda_->final_state_token(token, detail);
#endif
    token = {};
    if (!cpu_physical_) {
      detail = "CPU KS owner has no successful current final state";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    token = {1, cpu_physical_->identity};
    detail.clear();
    return VIBEQC_STATUS_SUCCESS;
  }

  vibeqc_status read_final_state(const dft::CudaKsFinalStateToken& expected,
                                 bool compute_weighted_density, dft::VerifiedKsFinalState& state,
                                 std::string& detail) {
#if VIBEQC_HAS_CUDA
    if (cuda_) return cuda_->read_final_state(expected, compute_weighted_density, state, detail);
#endif
    state = {};
    dft::CudaKsFinalStateToken current;
    const auto status = final_state_token(current, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    if (expected != current) {
      detail = "CPU KS final-state token is stale";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    // The SCF frame predates the last F[D] rebuild. Diagonalize that actual
    // retained physical F here; do not relabel the lagged orbital energies.
    // Explicit export costs one eigen solve and validation, zero Fock builds.
    const auto& ints = fock_.one_electron();
    const auto x = scf::reference::symmetric_orthogonalizer(ints.overlap, ints.nbf);
    std::vector<scf::reference::EigenResult> spins;
    spins.reserve(current.identity.model.spins);
    for (const auto& fock : cpu_physical_->fock)
      spins.push_back(scf::reference::generalized_eigen(fock, x, ints.nbf));
    dft::KsFinalStateCandidate candidate{current.identity,
                                         current.identity.determinant.factor.density_generation,
                                         true, std::move(spins)};
    scf::solver::FinalStateLimits limits{options_.density_tolerance, options_.energy_tolerance, 0,
                                         true};
    if (!dft::validate_ks_final_state(current.identity, ints.overlap, ints.hcore, *cpu_physical_,
                                      candidate, limits, compute_weighted_density, state, detail)) {
      invalidate_final_state();
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    return VIBEQC_STATUS_SUCCESS;
  }

  vibeqc_status read_derivative_state(const dft::CudaKsFinalStateToken& expected,
                                      KsDerivativeSnapshot& output, std::string& detail) {
    output = {};
    if (options_.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE) {
      detail = "DFT density-fitted derivative snapshots require auxiliary and metric response";
      return VIBEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::VerifiedKsFinalState state;
#if VIBEQC_HAS_CUDA
    const auto before = cuda_ ? cuda_->transfers() : dft::CudaKsTransfers{};
#endif
    const auto status = read_final_state(expected, true, state, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    // These are the provider's actual metric and the collocation/grid sources
    // used by this immutable KS owner, not caller-supplied identity labels.
    output = {std::move(state), system_,        fock_.one_electron().overlap,
              basis_.packed,    grid_.points(), grid_.weights(),
              grid_.owners()};
    // Both backends collocate this owner's exact host-built quadrature. Export
    // its raw measures directly; dividing partitioned weights loses tail data.
    output.atomic_weights = grid_.atomic_weights();
#if VIBEQC_HAS_CUDA
    if (cuda_) {
      const auto after = cuda_->transfers();
      output.export_d2h_bytes = after.final_state_d2h_bytes - before.final_state_d2h_bytes;
      output.export_reads = after.final_state_reads - before.final_state_reads;
      output.export_synchronizations = after.synchronizations - before.synchronizations;
    }
#endif
    return VIBEQC_STATUS_SUCCESS;
  }

  Result execute(bool compute_forces) override {
    invalidate_final_state();
    const char* method_name = semilocal_family_name(execution_plan_);
    if (compute_forces) {
      const char* issue =
          execution_plan_.semilocal_family == dft::SemilocalFamily::R2scan ? "#164" : "#163";
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        std::string(method_name) +
                            " KS nuclear gradients are tracked separately in issue " + issue);
    }
    auto result = adapt_result(run(nullptr, true, true), backend_);
    apply_d4(result);
    return result;
  }

  void apply_d4(Result& result) {
    if (!d4_) return;
    std::vector<double> coordinates;
    coordinates.reserve(3 * system_.atoms.size());
    for (const auto& atom : system_.atoms)
      coordinates.insert(coordinates.end(), atom.position.begin(), atom.position.end());
    const std::uint8_t active = 1;
    const std::uint8_t want_gradient = 0;
    std::vector<dft::dispersion::D4Status> statuses;
    std::vector<double> components, gradients, charges;
    std::string detail;
    const auto status =
        d4_->execute(coordinates, std::span(&active, 1), std::span(&want_gradient, 1), statuses,
                     components, gradients, charges, detail);
    if (status != VIBEQC_STATUS_SUCCESS || statuses.size() != 1 ||
        statuses[0] != dft::dispersion::D4Status::success || components.size() != 2)
      throw MethodError(status == VIBEQC_STATUS_SUCCESS ? VIBEQC_STATUS_NUMERICAL_FAILURE : status,
                        detail.empty() ? "PBE-D4 correction failed" : detail);
    result.energy += components[0] + components[1];
  }

  /** Single-system and native batch paths share the same scientific owner. */
  scf::ScfResult run(const std::vector<double>* initial_density, bool reuse_warm,
                     bool update_warm) {
    invalidate_final_state();
#if VIBEQC_HAS_CUDA
    if (cuda_) {
      // Native iterations read only scalar diagnostics. The public energy
      // result does not require a final AO matrix download; warm D stays resident.
      cuda_->set_warm_start_updates(update_warm);
      auto native = cuda_->run(initial_density, reuse_warm, false);
      if (cuda_->failed())
        throw MethodError(VIBEQC_STATUS_NUMERICAL_FAILURE, "CUDA KS physical evaluation failed");
      return native;
    }
#endif
    if (cpu_epoch_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CPU KS solve epoch exhausted");
    ++cpu_epoch_;
    const auto* seed =
        initial_density ? initial_density : (reuse_warm && !warm_.empty() ? &warm_ : nullptr);
    // The CPU driver already samples its provider/grid. Add only the retained
    // last-good density, which coexists with its current/proposed densities.
    runtime::CpuRetainedCapacity retained_warm(runtime::vector_bytes(warm_));
    scf::ScfResult native;
    if (execution_plan_.semilocal_family == dft::SemilocalFamily::Wb97mv) {
      if (!range_correction_ || !nonlocal_)
        throw std::runtime_error("WB97M-V requires both range-exchange and nonlocal owners");
      native = unrestricted(execution_plan_)
                   ? scf::run_wb97mv_uks(fock_, *range_correction_, basis_, grid_, options_,
                                         *nonlocal_, seed)
                   : scf::run_wb97mv_rks(fock_, *range_correction_, basis_, grid_, options_,
                                         *nonlocal_, seed);
    } else if (execution_plan_.range_exchange) {
      if (!range_correction_)
        throw std::runtime_error("KS range-exchange correction owner is missing");
      native = unrestricted(execution_plan_)
                   ? scf::run_pbe_rsh_uks(fock_, *range_correction_, basis_, grid_, options_, seed,
                                          nonlocal_.get())
                   : scf::run_pbe_rsh_rks(fock_, *range_correction_, basis_, grid_, options_, seed,
                                          nonlocal_.get());
    } else if (execution_plan_.semilocal_family == dft::SemilocalFamily::B3lyp)
      native = unrestricted(execution_plan_)
                   ? scf::run_b3lyp_uks(fock_, basis_, grid_, options_, seed)
                   : scf::run_b3lyp_rks(fock_, basis_, grid_, options_, seed);
    else if (execution_plan_.semilocal_family == dft::SemilocalFamily::Pbe && nonlocal_)
      native = unrestricted(execution_plan_)
                   ? scf::run_pbe_uks_nonlocal(fock_, basis_, grid_, options_, seed, *nonlocal_)
                   : scf::run_pbe_rks_nonlocal(fock_, basis_, grid_, options_, seed, *nonlocal_);
    else
      native = scf::run_curated_semilocal_ks(fock_, basis_, grid_, options_,
                                             execution_plan_.semilocal_family,
                                             execution_plan_.spin_channels, seed);
    // This owner has immutable model/geometry/spin identity. Only successful
    // executions may replace its compatible last-good density; DIIS is fresh.
    if (native.converged && options_.retain_ks_state) {
      const auto spins = execution_plan_.spin_channels;
      const auto matrix = fock_.one_electron().nbf * fock_.one_electron().nbf;
      if (native.density.size() != spins * matrix ||
          native.ks_physical_fock.size() != spins * matrix ||
          (spins == 2 && native.dft_diagnostic.occupations.size() != spins))
        throw std::runtime_error("CPU KS retained spin-state shape mismatch");
      std::vector<scf::reference::Matrix> densities, focks;
      densities.reserve(spins);
      focks.reserve(spins);
      for (unsigned spin = 0; spin < spins; ++spin) {
        const auto begin = spin * matrix;
        densities.emplace_back(native.density.begin() + begin,
                               native.density.begin() + begin + matrix);
        focks.emplace_back(native.ks_physical_fock.begin() + begin,
                           native.ks_physical_fock.begin() + begin + matrix);
      }
      std::vector<std::size_t> occupied;
      if (spins == 1)
        occupied = {static_cast<std::size_t>(system_.electron_count / 2)};
      else
        occupied.assign(native.dft_diagnostic.occupations.begin(),
                        native.dft_diagnostic.occupations.end());
      dft::KsFinalStateIdentity identity;
      identity.determinant = {
          {cpu_owner_, 1, 1, 1}, cpu_epoch_, fock_.strategy(), std::move(occupied)};
      identity.model = {1,
                        scf_domain_version(execution_plan_),
                        grid_.spec(),
                        options_.xc_tile_points,
                        dft::semilocal_family_code(execution_plan_.semilocal_family),
                        spins,
                        -1,
                        cpu_owner_,
                        options_.semilocal_exchange_scale,
                        options_.semilocal_correlation_scale};
      if (range_correction_) identity.model.range_correction = range_correction_->strategy();
      if (nonlocal_) identity.model.nonlocal_correlation = nonlocal_->parameters();
      if (execution_plan_.semilocal_family == dft::SemilocalFamily::Wb97mv)
        identity.model.nonlocal_density_domain = dft::nlc::Vv10DensityDomain::MolecularV1;
      dft::KsPhysicalState physical{identity,
                                    true,
                                    std::move(densities),
                                    std::move(focks),
                                    native.dft_diagnostic.components,
                                    native.energy,
                                    native.dft_diagnostic.physical_residual};
      cpu_physical_ = std::move(physical);
      if (seed && spins == 2) {
        // A warm UKS proposal may pass the SCF step gate while its latest
        // physical F[D] frame still fails the stricter derivative export.
        // Certify that frame before publishing success or replacing the last
        // good seed. Rejection uses the existing single cold retry in the
        // batch owner, never a hidden solve during snapshot/force export.
        dft::VerifiedKsFinalState verified;
        std::string detail;
        const dft::CudaKsFinalStateToken token{1, cpu_physical_->identity};
        if (read_final_state(token, false, verified, detail) != VIBEQC_STATUS_SUCCESS) {
          native.converged = false;
          native.ks_physical_fock.clear();
        }
      }
    }
    if (native.converged && update_warm) warm_ = std::move(native.density);
    runtime::sample_cpu_capacity(host_numeric_capacity());
    return native;
  }

 private:
  void prepare_range_exchange(int device) {
    const auto spin =
        unrestricted(execution_plan_) ? scf::FockSpin::Unrestricted : scf::FockSpin::Restricted;
    const auto fock_backend =
        backend_ == VIBEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
    range_strategy_ = scf::resolve_fock_build(
        scf::make_rsh_correction_fock_spec(spin, execution_plan_.short_range_exchange,
                                           execution_plan_.long_range_exchange,
                                           execution_plan_.range_omega),
        fock_backend, options_.screening_tolerance);
    if (fock_backend == scf::FockBackend::Cpu)
      range_correction_ =
          std::make_unique<scf::PreparedFockPlan>(system_, nullptr, *range_strategy_, device);
  }

  void prepare_nonlocal(int device) {
    if (grid_.point_count() > std::numeric_limits<std::uint32_t>::max())
      throw MethodError(VIBEQC_STATUS_OUT_OF_MEMORY,
                        "KS grid exceeds the VV10 public point-count domain");
    vibeqc_status status = VIBEQC_STATUS_INTERNAL_ERROR;
    std::string detail;
    nonlocal_ = dft::nlc::Vv10Plan::prepare(
        backend_, device, static_cast<std::uint32_t>(grid_.point_count()),
        static_cast<std::uint32_t>(options_.xc_tile_points), execution_plan_.nonlocal_parameters,
        execution_plan_.nonlocal_maximum_bytes, detail, status);
    if (!nonlocal_)
      throw MethodError(
          status, detail.empty() ? "self-consistent nonlocal plan preparation failed" : detail);
  }

  void prepare_d4(int device) {
    const auto source = ::vibeqc::generated::method_parameters::pbeD4();
    dft::dispersion::D4Parameters parameters{dft::dispersion::D4ReferenceModel::eeq,
                                             source.s6,
                                             source.s8,
                                             source.s9,
                                             source.a1,
                                             source.a2,
                                             source.cn_cutoff,
                                             source.pair_cutoff,
                                             source.atm_cutoff,
                                             source.ga,
                                             source.gc};
    std::vector<std::uint32_t> offsets{0, static_cast<std::uint32_t>(system_.atoms.size())};
    std::vector<std::int32_t> atomic_numbers;
    std::vector<double> coordinates;
    atomic_numbers.reserve(system_.atoms.size());
    coordinates.reserve(3 * system_.atoms.size());
    for (const auto& atom : system_.atoms) {
      atomic_numbers.push_back(atom.atomic_number);
      coordinates.insert(coordinates.end(), atom.position.begin(), atom.position.end());
    }
    vibeqc_status status = VIBEQC_STATUS_INTERNAL_ERROR;
    std::string detail;
    d4_ = dft::dispersion::D4Plan::prepare(
        backend_, device, std::move(offsets), std::move(atomic_numbers),
        std::vector<double>{static_cast<double>(system_.charge)}, std::move(coordinates),
        parameters, dft::dispersion::D4EEQProfile::standard, 256ull * 1024ull * 1024ull, detail,
        status);
    if (!d4_)
      throw MethodError(status,
                        detail.empty() ? "PBE-D4 production plan preparation failed" : detail);
  }

  Capabilities capabilities_;
  core::System system_;
  NativeKsExecutionPlan execution_plan_;
  scf::ScfOptions options_;
  vibeqc_backend backend_;
  scf::PreparedFockPlan fock_;
  std::optional<scf::ResolvedFockBuild> range_strategy_;
  std::unique_ptr<scf::PreparedFockPlan> range_correction_;
  dft::AoBasis basis_;
  dft::MolecularGrid grid_;
  std::vector<double> warm_;
  const std::uint64_t cpu_owner_{next_cpu_ks_owner()};
  std::uint64_t cpu_epoch_{};
  std::optional<dft::KsPhysicalState> cpu_physical_;
  std::unique_ptr<dft::dispersion::D4Plan> d4_;
  std::unique_ptr<dft::nlc::Vv10Plan> nonlocal_;
#if VIBEQC_HAS_CUDA
  std::unique_ptr<dft::CudaKsPlan> cuda_;
#endif
};

std::vector<double> positions(const core::System& system) {
  std::vector<double> out;
  out.reserve(3 * system.atoms.size());
  for (const auto& atom : system.atoms)
    out.insert(out.end(), atom.position.begin(), atom.position.end());
  return out;
}

bool valid_positions(const std::vector<double>& coordinates, const core::System& system) {
  return coordinates.size() == 3 * system.atoms.size() &&
         std::all_of(coordinates.begin(), coordinates.end(),
                     [](double value) { return std::isfinite(value); });
}

void set_positions(core::System& system, const std::vector<double>& coordinates) {
  for (std::size_t i = 0; i < system.atoms.size(); ++i)
    std::copy_n(coordinates.begin() + 3 * i, 3, system.atoms[i].position.begin());
}

vibeqc_status item_exception_status() {
  try {
    throw;
  } catch (const MethodError& error) {
    return error.status();
  } catch (const vibeqc::Error& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
}

/** Independent native KS owners, with ordinary-stream round-robin CUDA work.
 * Geometry is rebuilt per item, while model/basis/charge/spin remain immutable.
 * No HF graph or Python calculation loop participates in this schedule. */
class KsPreparedBatch final : public PreparedBatch {
 public:
  KsPreparedBatch(Capabilities capabilities, std::vector<core::System> systems,
                  NativeKsExecutionPlan execution_plan, scf::ScfOptions options, dft::GridSpec grid,
                  vibeqc_backend backend, int device, bool warm_enabled,
                  std::optional<core::System> auxiliary)
      : capabilities_(capabilities),
        systems_(std::move(systems)),
        execution_plan_(execution_plan),
        options_(std::move(options)),
        grid_spec_(std::move(grid)),
        backend_(backend),
        device_(device),
        warm_enabled_(warm_enabled),
        auxiliary_(std::move(auxiliary)),
        items_(systems_.size()) {
    for (std::size_t i = 0; i < size(); ++i) {
      runtime::CpuRetainedCapacity neighbors(host_numeric_capacity());
      items_[i].plan = make_plan(systems_[i]);
    }
  }

  std::size_t size() const noexcept override { return systems_.size(); }

  void invalidate_result() override {
    for (auto& item : items_)
      if (item.plan) item.plan->invalidate_final_state();
  }

  std::vector<BatchItemResult> execute(const Coordinates& coordinates,
                                       bool compute_forces) override {
    invalidate_result();
    if (compute_forces)
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                        "KS nuclear gradients are tracked separately in issue #163");
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("KS batch coordinates do not match system count");
    std::vector<BatchItemResult> results(size());
    std::vector<bool> ready(size(), false);
    // Allocate source-geometry metadata before launching any item. The success
    // path can then publish its last-good identity without a coordinate copy.
    std::vector<scf::HfWarmState> candidates(size());
    for (std::size_t i = 0; i < size(); ++i) {
      auto& result = results[i];
      result.bucket_id = i;  // One ordinary stream/owner per stable input slot.
      result.calculation.executed_backend = backend_;
      result.calculation.energy = std::numeric_limits<double>::quiet_NaN();
      try {
        auto target = systems_[i];
        if (!coordinates.empty() && coordinates[i]) {
          if (!valid_positions(*coordinates[i], target))
            throw std::invalid_argument("invalid KS batch item coordinates");
          set_positions(target, *coordinates[i]);
        }
        auto& item = items_[i];
        candidates[i].coordinates = positions(target);
        if (!item.plan || positions(item.plan->system()) != candidates[i].coordinates) {
          // Preserve the last GOOD seed before freeing its device owner. This
          // explicit rebuild download is never part of routine SCF iterations.
          materialize_warm(i);
#if VIBEQC_HAS_CUDA
          if (auto* cuda = item.plan ? item.plan->cuda_plan() : nullptr)
            add_transfers(item.retired_transfers, cuda->transfers());
#endif
          item.plan.reset();
          item.resident_warm = false;
          item.plan = make_plan(target);
        }
        result.warm_start_used = warm_enabled_ && item.warm.has_value();
        ready[i] = true;
      } catch (...) {
        result.status = item_exception_status();
      }
    }

    const auto finish = [&](std::size_t i, scf::ScfResult native) {
      auto& result = results[i];
      result.calculation = adapt_result(std::move(native), backend_);
      items_[i].plan->apply_d4(result.calculation);
      const auto& calculation = result.calculation;
      result.status =
          calculation.convergence.converged ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_NOT_CONVERGED;
      if (calculation.convergence.converged && warm_enabled_ && warm_updates_) {
        auto& state = candidates[i];
        state.energy = calculation.energy;
        state.energy_change = calculation.convergence.energy_change;
        state.density_rms = calculation.convergence.residual_rms;
        state.iterations = calculation.convergence.iterations;
        items_[i].warm = std::move(state);
        items_[i].resident_warm = true;
      }
    };

    // A rejected/nonconverged warm solve gets one cold retry. CUDA retries
    // retain the same per-item scheduler; a failed neighbor never halts it.
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
      std::vector<bool> running(size(), false);
      for (std::size_t i = 0; i < size(); ++i) {
        auto& result = results[i];
        // Retry only seed-related failures. Resource/driver failures preserve
        // their first status and leave the last-good seed for explicit replay.
        const bool seed_failure = result.status == VIBEQC_STATUS_NOT_CONVERGED ||
                                  result.status == VIBEQC_STATUS_NUMERICAL_FAILURE ||
                                  result.status == VIBEQC_STATUS_INVALID_ARGUMENT;
        if (!ready[i] || (attempt && (!result.warm_start_used || !seed_failure))) continue;
        if (attempt) {
          result.warm_start_fallback = true;
          // Release the failed attempt's exported history before starting another
          // solve, preserving the two-history resource bound.
          result.calculation.ks_diagnostic.reset();
        }
        auto& item = items_[i];
        const bool reuse = !attempt && result.warm_start_used;
        const auto* seed = reuse && !item.resident_warm ? &item.warm->density : nullptr;
        try {
#if VIBEQC_HAS_CUDA
          if (auto* cuda = item.plan->cuda_plan()) {
            cuda->set_warm_start_updates(warm_enabled_ && warm_updates_);
            cuda->begin(seed, reuse && item.resident_warm);
            running[i] = true;
            continue;
          }
#endif
          runtime::CpuRetainedCapacity neighbors(host_numeric_capacity(i));
          finish(i,
                 item.plan->run(seed, reuse && item.resident_warm, warm_enabled_ && warm_updates_));
        } catch (...) {
          result.status = item_exception_status();
        }
      }
#if VIBEQC_HAS_CUDA
      while (std::any_of(running.begin(), running.end(), [](bool value) { return value; })) {
        // Submit ALL active streams before synchronizing any scalar record.
        for (std::size_t i = 0; i < size(); ++i) {
          if (!running[i]) continue;
          try {
            items_[i].plan->cuda_plan()->enqueue_iteration();
          } catch (...) {
            results[i].status = item_exception_status();
            running[i] = false;
          }
        }
        for (std::size_t i = 0; i < size(); ++i) {
          if (!running[i]) continue;
          try {
            auto* cuda = items_[i].plan->cuda_plan();
            if (cuda->finish_iteration()) continue;
            running[i] = false;
            if (cuda->failed())
              throw MethodError(VIBEQC_STATUS_NUMERICAL_FAILURE,
                                "CUDA KS physical evaluation failed");
            finish(i, cuda->result(false));
          } catch (...) {
            results[i].status = item_exception_status();
            running[i] = false;
          }
        }
      }
#endif
    }
    runtime::sample_cpu_capacity(host_numeric_capacity());
    return results;
  }

  void clear_warm_starts() override {
    for (auto& item : items_) {
      item.warm.reset();
      item.resident_warm = false;
      if (item.plan) item.plan->clear_warm_start();
    }
  }

  std::size_t warm_density_size(std::size_t index) const override {
    const auto n = molecule::ao_count(systems_.at(index));
    const std::size_t spins = execution_plan_.spin_channels;
    if (!n || n > std::numeric_limits<std::size_t>::max() / n / spins / sizeof(double))
      throw std::invalid_argument("KS warm density dimensions overflow");
    return spins * n * n;
  }

  const std::optional<scf::HfWarmState>& warm_state(std::size_t index) const override {
    materialize_warm(index);
    return items_.at(index).warm;
  }

  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) override {
    if (!warm_enabled_ || states.size() != size())
      throw std::invalid_argument("KS seed restore requires a matching warm-enabled batch");
    for (std::size_t i = 0; i < size(); ++i) {
      if (!states[i]) continue;
      const auto& state = *states[i];
      if (state.density.size() != warm_density_size(i) ||
          !valid_positions(state.coordinates, systems_[i]) || state.iterations < 0 ||
          !std::isfinite(state.energy) || !std::isfinite(state.energy_change) ||
          !std::isfinite(state.density_rms) || state.density_rms < 0)
        throw std::invalid_argument("invalid KS seed dimensions or diagnostics");
      auto source = systems_[i];
      set_positions(source, state.coordinates);
      // This common validation reads only source S and checks the shared
      // spin-density convention. It performs no HF Fock/energy evaluation.
      scf::validate_hf_warm_density(
          source, unrestricted(execution_plan_) ? VIBEQC_METHOD_UHF : VIBEQC_METHOD_RHF,
          state.density);
    }
    // All source-metric validation precedes the no-throw commit. Missing
    // entries preserve neighbors, including their resident density ownership.
    for (std::size_t i = 0; i < size(); ++i) {
      if (!states[i]) continue;
      auto& item = items_[i];
      item.warm.swap(states[i]);
      item.resident_warm = false;
      if (item.plan) item.plan->clear_warm_start();
    }
  }

  void set_warm_start_updates(bool enabled) override { warm_updates_ = enabled; }

  std::optional<KsTransportDiagnostic> ks_transport_diagnostic(std::size_t index) const override {
    const auto& item = items_.at(index);
#if VIBEQC_HAS_CUDA
    if (auto* cuda = item.plan ? item.plan->cuda_plan() : nullptr) {
      auto cumulative = item.retired_transfers;
      add_transfers(cumulative, cuda->transfers());
      return adapt_transfers(cumulative);
    }
#endif
    return std::nullopt;
  }

  vibeqc_status final_state_token(std::size_t index, dft::CudaKsFinalStateToken& token,
                                  std::string& detail) const {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->final_state_token(token, detail);
    token = {};
    detail = "KS batch item has no prepared final-state owner";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  vibeqc_status read_final_state(std::size_t index, const dft::CudaKsFinalStateToken& expected,
                                 bool compute_weighted_density, dft::VerifiedKsFinalState& state,
                                 std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->read_final_state(expected, compute_weighted_density, state,
                                                  detail);
    state = {};
    detail = "KS batch item has no prepared final-state owner";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  vibeqc_status read_derivative_state(std::size_t index, const dft::CudaKsFinalStateToken& expected,
                                      KsDerivativeSnapshot& output, std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->read_derivative_state(expected, output, detail);
    output = {};
    detail = "KS batch item has no prepared final-state owner";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  // These profiles describe HF graph/provider layouts, not this method's
  // ordinary-stream schedule. Absence is explicit at the common interface.
  std::optional<std::vector<DirectShellClassProfileEntry>> last_direct_shell_class_profile()
      const override {
    return std::nullopt;
  }
  std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile() const override {
    return std::nullopt;
  }
  std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const override { return {}; }
  std::vector<scf::CudaDensityFittingMetricDiagnostic> last_density_fitting_metric_diagnostics()
      const override {
    std::vector<scf::CudaDensityFittingMetricDiagnostic> result;
    for (std::size_t index = 0; index < items_.size(); ++index)
      if (items_[index].plan)
        for (auto diagnostic : items_[index].plan->fitted_diagnostics()) {
          // KS assigns one provider/bucket per input slot, including equal shapes.
          diagnostic.bucket_id = index;
          diagnostic.system_index = index;
          result.push_back(diagnostic);
        }
    return result;
  }
  std::vector<InactiveEigensolverProfileEntry> last_inactive_eigensolver_profile() const override {
    return {};
  }

 private:
  /** All other owners remain alive while one CPU item executes. The selected
   * item's externally materialized seed is also distinct from its plan. */
  std::size_t host_numeric_capacity(std::size_t exclude_plan = SIZE_MAX) const noexcept {
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < items_.size(); ++i) {
      const auto& item = items_[i];
      if (item.plan && i != exclude_plan)
        bytes = runtime::add_capacity(bytes, item.plan->host_numeric_capacity());
      if (item.warm)
        bytes = runtime::add_capacity(
            bytes, runtime::vector_capacities(item.warm->density, item.warm->coordinates));
    }
    return bytes;
  }

  struct Item {
    std::unique_ptr<KsPreparedCalculation> plan;
    // Density is materialized only for explicit output, import, or rebuilding
    // an owner. Empty density with resident_warm=true is a valid lazy snapshot.
    mutable std::optional<scf::HfWarmState> warm;
    bool resident_warm{};
#if VIBEQC_HAS_CUDA
    dft::CudaKsTransfers retired_transfers;
#endif
  };
  std::unique_ptr<KsPreparedCalculation> make_plan(const core::System& system) const {
    return std::make_unique<KsPreparedCalculation>(capabilities_, system, execution_plan_, options_,
                                                   grid_spec_, backend_, device_,
                                                   ks_auxiliary_for_system(system, auxiliary_));
  }
  void materialize_warm(std::size_t i) const {
    const auto& item = items_.at(i);
    if (item.warm && item.warm->density.empty() && item.resident_warm)
      item.warm->density = item.plan->warm_density();
  }

  Capabilities capabilities_;
  std::vector<core::System> systems_;
  NativeKsExecutionPlan execution_plan_;
  scf::ScfOptions options_;
  dft::GridSpec grid_spec_;
  vibeqc_backend backend_;
  int device_;
  bool warm_enabled_, warm_updates_{true};
  std::optional<core::System> auxiliary_;
  std::vector<Item> items_;
};

}  // namespace

vibeqc_status dft_final_state_token(const PreparedCalculation& calculation,
                                    dft::CudaKsFinalStateToken& token, std::string& detail) {
  const auto* ks = dynamic_cast<const KsPreparedCalculation*>(&calculation);
  if (ks) return ks->final_state_token(token, detail);
  token = {};
  detail = "prepared calculation is not a KS final-state owner";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

vibeqc_status read_dft_final_state(PreparedCalculation& calculation,
                                   const dft::CudaKsFinalStateToken& expected,
                                   bool compute_weighted_density, dft::VerifiedKsFinalState& state,
                                   std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedCalculation*>(&calculation);
  if (ks) return ks->read_final_state(expected, compute_weighted_density, state, detail);
  state = {};
  detail = "prepared calculation is not a KS final-state owner";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

vibeqc_status dft_final_state_token(const PreparedBatch& batch, std::size_t index,
                                    dft::CudaKsFinalStateToken& token, std::string& detail) {
  const auto* ks = dynamic_cast<const KsPreparedBatch*>(&batch);
  if (ks) return ks->final_state_token(index, token, detail);
  token = {};
  detail = "prepared batch is not a KS final-state owner";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

vibeqc_status read_dft_final_state(PreparedBatch& batch, std::size_t index,
                                   const dft::CudaKsFinalStateToken& expected,
                                   bool compute_weighted_density, dft::VerifiedKsFinalState& state,
                                   std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks) return ks->read_final_state(index, expected, compute_weighted_density, state, detail);
  state = {};
  detail = "prepared batch is not a KS final-state owner";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

vibeqc_status read_dft_derivative_state(PreparedBatch& batch, std::size_t index,
                                        const dft::CudaKsFinalStateToken& expected,
                                        KsDerivativeSnapshot& output, std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks) return ks->read_derivative_state(index, expected, output, detail);
  output = {};
  detail = "prepared batch is not a KS final-state owner";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

void validate_ks_spin_state(const NativeKsExecutionPlan& execution_plan,
                            const core::System& system) {
  if (execution_plan.semilocal_family == dft::SemilocalFamily::Wb97mv && !system.ecp_terms.empty())
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "WB97M-V ECP execution is not qualified");
  if (!unrestricted(execution_plan)) {
    if (system.electron_count <= 0 || system.electron_count % 2 || system.multiplicity != 1)
      throw std::invalid_argument(
          "RKS requires a positive even electron count and spin multiplicity 1");
    return;
  }
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  if (system.electron_count <= 0 || spin_excess < 0 || spin_excess > system.electron_count ||
      (system.electron_count - spin_excess) % 2 != 0)
    throw std::invalid_argument(
        "UKS requires electron count and multiplicity to define integer "
        "nonnegative spin occupations");
}

vibeqc_status validate_dft_system(vibeqc_method, const core::System& system, std::string& detail) {
  if (system.shells.empty()) {
    detail = "DFT requires an explicit Gaussian orbital basis";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  if (system.electron_count > 0 && spin_excess >= 0 && spin_excess <= system.electron_count &&
      (system.electron_count - spin_excess) % 2 == 0)
    return VIBEQC_STATUS_SUCCESS;
  detail =
      "DFT requires electron count and multiplicity to define integer nonnegative spin occupations";
  return VIBEQC_STATUS_INVALID_ARGUMENT;
}

std::unique_ptr<PreparedCalculation> prepare_dft_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor) {
#if !VIBEQC_HAS_CUDA
  if (context.requested_backend == VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT CUDA backend is not built");
#endif
  NativeKsExecutionPlan execution_plan;
  auto options = dft_options(descriptor, context.requested_backend, execution_plan);
  validate_ks_spin_state(execution_plan, system);
  if (options.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE &&
      (!system.ecp_terms.empty() ||
       std::any_of(system.atoms.begin(), system.atoms.end(),
                   [](const auto& atom) { return atom.ecp_core != 0; })))
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT ECP density fitting is not qualified");
  validate_ks_auxiliary_geometry(system, ks_auxiliary_template(descriptor));
  auto grid = ks_grid_options(descriptor, options, execution_plan);
  return std::make_unique<KsPreparedCalculation>(
      capabilities, system, execution_plan, std::move(options), std::move(grid),
      context.requested_backend, context.device_id,
      ks_auxiliary_for_system(system, ks_auxiliary_template(descriptor)));
}

std::unique_ptr<PreparedBatch> prepare_dft_batch(const Capabilities& capabilities,
                                                 core::ContextState& context,
                                                 std::vector<core::System> systems,
                                                 const vibeqc_method_descriptor& descriptor,
                                                 vibeqc_batch_flags flags) {
  if ((flags & ~VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED,
                      "KS batches support warm starts but not HF-specific profiling flags");
#if !VIBEQC_HAS_CUDA
  if (context.requested_backend == VIBEQC_BACKEND_CUDA)
    throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT CUDA backend is not built");
#endif
  NativeKsExecutionPlan execution_plan;
  auto options = dft_options(descriptor, context.requested_backend, execution_plan);
  for (const auto& system : systems) {
    validate_ks_spin_state(execution_plan, system);
    if (options.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE &&
        (!system.ecp_terms.empty() ||
         std::any_of(system.atoms.begin(), system.atoms.end(),
                     [](const auto& atom) { return atom.ecp_core != 0; })))
      throw MethodError(VIBEQC_STATUS_NOT_IMPLEMENTED, "DFT ECP density fitting is not qualified");
  }
  if (!systems.empty())
    validate_ks_auxiliary_geometry(systems.front(), ks_auxiliary_template(descriptor));
  auto grid = ks_grid_options(descriptor, options, execution_plan);
  return std::make_unique<KsPreparedBatch>(
      capabilities, std::move(systems), execution_plan, std::move(options), std::move(grid),
      context.requested_backend, context.device_id, (flags & VIBEQC_BATCH_ENABLE_WARM_STARTS) != 0,
      ks_auxiliary_template(descriptor));
}

}  // namespace vibeqc::methods::detail

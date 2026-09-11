#ifndef VIBEQC_SCF_CUDA_RHF_POLICY_HPP
#define VIBEQC_SCF_CUDA_RHF_POLICY_HPP

#include <optional>

namespace vibeqc::scf::cuda_policy {

/** Runtime policy switches kept in a host-only translation unit. */
bool reuse_converged_fock_requested() noexcept;
std::optional<double> configured_mixed_precision_fock_threshold(
    double screening_tolerance) noexcept;
bool graph_native_eigensolver_override_requested() noexcept;
bool xsyev_probe_skip_diagnostic_requested() noexcept;
bool bounded_direct_streaming_override_requested() noexcept;
bool bounded_direct_count_diagnostic_requested() noexcept;
bool bounded_direct_aot_only_diagnostic_requested() noexcept;
bool bounded_direct_fock_only_diagnostic_requested() noexcept;
bool bounded_fock_class_timing_requested() noexcept;
bool direct_tile_validation_requested() noexcept;
double converged_fock_reuse_density_rms(double density_tolerance) noexcept;
bool force_density_product_screening_requested() noexcept;
bool resident_ppps_bra_requested() noexcept;
bool ppps_signature_bucketing_requested() noexcept;
bool psps_signature_bucketing_requested() noexcept;
bool ppss_signature_bucketing_requested() noexcept;
unsigned ppps_resident_block_threads_requested() noexcept;
bool one_electron_force_scalar_requested() noexcept;
/** 0: one AO pair per thread; 1: one shell pair per warp. */
unsigned one_electron_value_mapping_requested() noexcept;
/** Generated derivative candidates are opt-in and read at each force execution. */
bool generated_one_electron_derivatives_requested() noexcept;
/** 0: AO threads; 1: shell-pair warp lanes; 2: deterministic serial diagnostics. */
unsigned one_electron_derivative_mapping_requested() noexcept;
bool resident_psss_bra_requested() noexcept;
/** Generated weighted primitive candidate; frozen into a prepared bucket. */
bool generated_psss_weighted_requested() noexcept;

/** 0: cooperative dense elements; 1: deterministic serial traversal. */
unsigned df_derivative_mapping_requested() noexcept;
/** 0: contiguous auxiliary outputs; 1: AO components; 2: primitive lanes. */
unsigned df_value_mapping_requested() noexcept;

}  // namespace vibeqc::scf::cuda_policy

#endif

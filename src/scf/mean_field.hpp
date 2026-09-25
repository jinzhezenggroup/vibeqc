#ifndef VIBEQC_SCF_MEAN_FIELD_HPP
#define VIBEQC_SCF_MEAN_FIELD_HPP

#include <memory>
#include <optional>
#include <vector>

#include "core/types.hpp"
#include "dft/semilocal_family.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/density_fitting.hpp"
#include "scf/types.hpp"

namespace vibeqc::scf {
namespace initial_guess {
class OverlapOrthogonalizer;
}
/** Shared physical reference validation, including canonical and SCF residuals.
 * Arrays use
 * detached row-major spatial AO/MO conventions on every backend. */
void validate_physical_reference(PhysicalReference& reference);

struct CudaDensityFittingMetricDiagnostic;
struct CudaDensityFittingJkPlan;
class PreparedFockPlan;

}  // namespace vibeqc::scf

namespace vibeqc::dft {
class AoBasis;
class MolecularGrid;
struct SemilocalPointProgram;
namespace nlc {
class Vv10Plan;
struct Vv10Parameters;
}  // namespace nlc
}  // namespace vibeqc::dft

namespace vibeqc::scf {

/** Validate controls and derive the requested value/force capability for this
 * execution without changing the immutable prepared method request. */
ResolvedFockBuild fock_strategy_for_execution(const ScfOptions& options);
ScfResult run_prepared_fock_strategy(const PreparedFockPlan& plan, const ScfOptions& options,
                                     const std::vector<double>* initial_density = nullptr,
                                     initial_guess::OverlapOrthogonalizer* overlap_cache = nullptr);

/** CPU energy-only LDA RKS using a Coulomb-only prepared Fock source and the
 * matching prepared AO/grid/XC state. */
ScfResult run_lda_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density = nullptr);

/** CPU energy-only PBE RKS using the versioned scaled-v1 domain policy. */
ScfResult run_pbe_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density = nullptr);

/** PBE-family RKS with one MethodIR-owned VV10/rVV10 contribution. */
ScfResult run_pbe_rks_nonlocal(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                               const dft::MolecularGrid& grid, const ScfOptions& options,
                               const std::vector<double>* initial_density,
                               dft::nlc::Vv10Plan& nonlocal_correlation);

/** MethodIR-owned PBE-family range-separated exact exchange. The primary plan
 * carries Coulomb plus the short-range fraction as full-range K; the correction
 * plan contributes only (long-short) long-range K. An optional VV10/rVV10
 * provider is composed in the same self-consistent physical Fock. */
ScfResult run_pbe_rsh_rks(const PreparedFockPlan& primary,
                          const PreparedFockPlan& long_range_correction, const dft::AoBasis& basis,
                          const dft::MolecularGrid& grid, const ScfOptions& options,
                          const std::vector<double>* initial_density = nullptr,
                          dft::nlc::Vv10Plan* nonlocal_correlation = nullptr);

ScfResult run_r2scan_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                         const dft::MolecularGrid& grid, const ScfOptions& options,
                         const std::vector<double>* initial_density = nullptr);

/** CPU qualification entry for one evidence-bound compiled semilocal point
 * program. It reuses the ordinary RKS loop and does not register a public
 * method or infer production admission from the descriptor. */
ScfResult run_semilocal_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                            const dft::MolecularGrid& grid, const ScfOptions& options,
                            const dft::SemilocalPointProgram& program,
                            const std::vector<double>* initial_density = nullptr);

/** Generic RSH lowering. Fractions are physical exact-exchange weights. */
FockBuildSpec make_global_hybrid_fock_spec(FockSpin spin, double exact_exchange);
FockBuildSpec make_rsh_primary_fock_spec(FockSpin spin, double short_range_exchange);
FockBuildSpec make_rsh_correction_fock_spec(FockSpin spin, double short_range_exchange,
                                            double long_range_exchange, double omega);

ScfResult run_b3lyp_rks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                        const dft::MolecularGrid& grid, const ScfOptions& options,
                        const std::vector<double>* initial_density = nullptr);

/** Canonical generated B97M/RSH/VV10 contract. No independent SCF loop. */
void require_wb97mv_composition(const ResolvedFockBuild& primary,
                                const ResolvedFockBuild& correction,
                                const dft::nlc::Vv10Parameters& nonlocal);
ScfResult run_wb97mv_rks(const PreparedFockPlan& primary, const PreparedFockPlan& correction,
                         const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                         const ScfOptions& options, dft::nlc::Vv10Plan& nonlocal,
                         const std::vector<double>* initial_density = nullptr);
ScfResult run_wb97mv_uks(const PreparedFockPlan& primary, const PreparedFockPlan& correction,
                         const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                         const ScfOptions& options, dft::nlc::Vv10Plan& nonlocal,
                         const std::vector<double>* initial_density = nullptr);

ScfResult run_cam_b3lyp_rks(const PreparedFockPlan& primary,
                            const PreparedFockPlan& long_range_correction,
                            const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                            const ScfOptions& options,
                            const std::vector<double>* initial_density = nullptr);

/** CPU energy-only spin-polarized LDA UKS with independent alpha/beta
 * densities and a Coulomb
 * source built from their total density. */
ScfResult run_lda_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density = nullptr);

/** CPU energy-only PBE UKS using the versioned polarized production tail. */
ScfResult run_pbe_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                      const dft::MolecularGrid& grid, const ScfOptions& options,
                      const std::vector<double>* initial_density = nullptr);

/** PBE-family UKS with one MethodIR-owned VV10/rVV10 contribution. */
ScfResult run_pbe_uks_nonlocal(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                               const dft::MolecularGrid& grid, const ScfOptions& options,
                               const std::vector<double>* initial_density,
                               dft::nlc::Vv10Plan& nonlocal_correlation);

ScfResult run_pbe_rsh_uks(const PreparedFockPlan& primary,
                          const PreparedFockPlan& long_range_correction, const dft::AoBasis& basis,
                          const dft::MolecularGrid& grid, const ScfOptions& options,
                          const std::vector<double>* initial_density = nullptr,
                          dft::nlc::Vv10Plan* nonlocal_correlation = nullptr);

ScfResult run_r2scan_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                         const dft::MolecularGrid& grid, const ScfOptions& options,
                         const std::vector<double>* initial_density = nullptr);

/** Common CPU entry for curated LDA/PBE/r2SCAN semilocal execution. The typed
 * family is shared with the prepared CUDA owner; composed B3/RSH/VV10 paths
 * retain their dedicated composition checks. */
ScfResult run_curated_semilocal_ks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                                   const dft::MolecularGrid& grid, const ScfOptions& options,
                                   dft::SemilocalFamily family, unsigned spin_channels,
                                   const std::vector<double>* initial_density = nullptr);

/** UKS counterpart of run_semilocal_rks for qualification execution. */
ScfResult run_semilocal_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                            const dft::MolecularGrid& grid, const ScfOptions& options,
                            const dft::SemilocalPointProgram& program,
                            const std::vector<double>* initial_density = nullptr);

ScfResult run_b3lyp_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                        const dft::MolecularGrid& grid, const ScfOptions& options,
                        const std::vector<double>* initial_density = nullptr);

ScfResult run_cam_b3lyp_uks(const PreparedFockPlan& primary,
                            const PreparedFockPlan& long_range_correction,
                            const dft::AoBasis& basis, const dft::MolecularGrid& grid,
                            const ScfOptions& options,
                            const std::vector<double>* initial_density = nullptr);

/** Independent unit-occupation spins, Coulomb of total D, and semilocal XC.
 * initial_density is alpha followed by beta; warm normalization preserves
 * each requested population. DIIS and all physical states belong to this run. */
ScfResult run_uks(const PreparedFockPlan& plan, const dft::AoBasis& basis,
                  const dft::MolecularGrid& grid, const ScfOptions& options, bool pbe,
                  const std::vector<double>* initial_density = nullptr);

/** Reuse independent CUDA sources when immutable inputs match. Build a new
 * candidate completely before replacing cached sources; fused standard HF
 * and the existing CPU storage lifetime retain their established dispatch. */
ScfResult run_fock_strategy_cached(std::unique_ptr<PreparedFockPlan>& cache,
                                   const core::System& system, const core::System* auxiliary,
                                   const ScfOptions& options, int device_id,
                                   const std::vector<double>* initial_density = nullptr,
                                   initial_guess::OverlapOrthogonalizer* overlap_cache = nullptr);

/** Rebuild only the source overlap on the CPU and apply the shared SCF
 * ensemble-density guard (Hermiticity, metric occupations, electron/spin trace).
 * This does not assert target compatibility or target convergence. */
void validate_hf_warm_density(const core::System& source, vibeqc_method method,
                              const std::vector<double>& density);

/** Execute the existing CPU SCF solver with an explicitly resolved independent
 * J/K model. Iterations, final energy and analytic forces share one provider
 * binding. A null auxiliary pointer uses the orbital basis for requested DF
 * terms. This does not add XC or advertise a complete DFT method.
 */
ScfResult run_cpu_fock_strategy(const core::System& system, const core::System* auxiliary,
                                const ScfOptions& options,
                                const std::vector<double>* initial_density = nullptr);

/** General independent CUDA route. Reuses host DIIS/eigensolve/finalization
 * control with CUDA direct/DF J/K and matched two-electron derivatives. The
 * standard HF entry retains its existing fused device solver separately. */
ScfResult run_cuda_independent_fock_strategy(const core::System& system,
                                             const core::System* auxiliary,
                                             const ScfOptions& options, int device_id,
                                             const std::vector<double>* initial_density = nullptr);

/** Method-neutral single-item dispatch. CPU independent providers and existing
 * CUDA fused HF schedules share this boundary; callers supply semantics through
 * options.resolved_fock_build instead of branching on a combined DF enum.
 */
ScfResult run_fock_strategy(const core::System& system, const core::System* auxiliary,
                            const ScfOptions& options, int device_id,
                            const std::vector<double>* initial_density = nullptr,
                            initial_guess::OverlapOrthogonalizer* overlap_cache = nullptr);

/** Run closed-shell RHF and assemble its variational analytic gradient. */
ScfResult run_rhf(const core::System& system, const ScfOptions& options,
                  const std::vector<double>* initial_density = nullptr);

/** Run spin-unrestricted HF and assemble its variational analytic gradient. */
ScfResult run_uhf(const core::System& system, const ScfOptions& options,
                  const std::vector<double>* initial_density = nullptr);

/**
 * Run RHF with a prepared auxiliary topology and the CPU DF reference
 * contractions. The auxiliary atoms are required to have the same geometry
 * as `system`; callers preparing a moving batch should update those positions
 * before invoking the function.
 */
ScfResult run_rhf_density_fitting(const core::System& system, const core::System& auxiliary_system,
                                  const ScfOptions& options,
                                  const std::vector<double>* initial_density = nullptr);

/** UHF counterpart of `run_rhf_density_fitting`. */
ScfResult run_uhf_density_fitting(const core::System& system, const core::System& auxiliary_system,
                                  const ScfOptions& options,
                                  const std::vector<double>* initial_density = nullptr);

/**
 * Execute DF SCF through the CUDA contraction plan. Two-/three-center DF
 * values and their first nuclear derivatives are generated by the CUDA
 * quartet recurrence; one-electron/overlap-Pulay assembly remains host-side
 * in this bridge. Iterative densities, Fock assembly, eigensolves, RI-J/RI-K
 * contractions, and the raw two-electron force response stay on the selected
 * CUDA device when the provider is available.
 */
ScfResult run_rhf_density_fitting_cuda(
    const core::System& system, const core::System& auxiliary_system, const ScfOptions& options,
    int device_id, const std::vector<double>* initial_density = nullptr,
    initial_guess::OverlapOrthogonalizer* overlap_cache = nullptr);

/** CUDA DF counterpart for UHF. */
ScfResult run_uhf_density_fitting_cuda(
    const core::System& system, const core::System& auxiliary_system, const ScfOptions& options,
    int device_id, const std::vector<double>* initial_density = nullptr,
    initial_guess::OverlapOrthogonalizer* overlap_cache = nullptr);

/**
 * Execute a homogeneous fleet bucket through one batched CUDA DF J/K plan.
 *
 * The returned vector preserves the input order.  A missing auxiliary
 * template means that each orbital basis is also used as its auxiliary basis,
 * matching the single-system DF entry points.  Individual preparation,
 * eigensolver, and convergence failures are isolated to their item whenever
 * the shared batched contractions can still be evaluated.
 */
std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket(
    const std::vector<core::System>& systems, const std::optional<core::System>& auxiliary_template,
    const ScfOptions& options, const std::vector<const std::vector<double>*>& initial_densities,
    int device_id, std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics = nullptr);

/**
 * Cached counterpart used by a persistent FleetPlan. The pointed-to plan is
 * retained by the caller across same-topology replays and may be replaced
 * when the caller invalidates its geometry cache.
 */
std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket_cached(
    CudaDensityFittingJkPlan** plan, const std::vector<core::System>& systems,
    const std::optional<core::System>& auxiliary_template, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics = nullptr,
    std::vector<std::optional<DensityFittingScfData>>* prepared_cache = nullptr,
    const std::vector<initial_guess::OverlapOrthogonalizer*>* overlap_caches = nullptr);

/** UHF counterpart of the batched CUDA DF bucket executor. */
std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket(
    const std::vector<core::System>& systems, const std::optional<core::System>& auxiliary_template,
    const ScfOptions& options, const std::vector<const std::vector<double>*>& initial_densities,
    int device_id, std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics = nullptr);

/** Cached UHF counterpart for persistent FleetPlan replay. */
std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket_cached(
    CudaDensityFittingJkPlan** plan, const std::vector<core::System>& systems,
    const std::optional<core::System>& auxiliary_template, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics = nullptr,
    std::vector<std::optional<DensityFittingScfData>>* prepared_cache = nullptr,
    const std::vector<initial_guess::OverlapOrthogonalizer*>* overlap_caches = nullptr);

/** Execute RHF through the native CUDA scientific path. */
ScfResult run_rhf_cuda(const core::System& system, const ScfOptions& options, int device_id,
                       const std::vector<double>* initial_density = nullptr);

/** Execute UHF through the native CUDA scientific path. */
ScfResult run_uhf_cuda(const core::System& system, const ScfOptions& options, int device_id,
                       const std::vector<double>* initial_density = nullptr);

}  // namespace vibeqc::scf

#endif

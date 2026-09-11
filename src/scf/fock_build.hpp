#ifndef VIBEQC_SCF_FOCK_BUILD_HPP
#define VIBEQC_SCF_FOCK_BUILD_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vibeqc::scf {

enum class FockSpin { Restricted, Unrestricted };
enum class FockOperator { FullRange, ShortRange, LongRange };
enum class FockApproximation { Exact, DensityFitted };
enum class FockBackend { Cpu, Cuda };
enum class FockSchedule {
  CpuReference,
  CudaFused,
  LegacyDensityFitting,
  CpuIndependent,
  /** Host SCF control with independently bound CUDA integral consumers. */
  CudaIndependent
};
enum class FockPrecision { Float64 };
enum class FockMatrixLayout { RowMajor, ColumnMajor };

struct FockTermSpec {
  bool present{true};
  double coefficient{1.0};
  FockOperator op{FockOperator::FullRange};
  double omega{};
  FockApproximation approximation{FockApproximation::Exact};
  bool operator==(const FockTermSpec&) const = default;
};

/** Raw-output selection, independent of the coefficients used by assembly.
 * A requested term with coefficient zero still produces its raw matrix.
 */
struct JkTermSelection {
  bool coulomb{true};
  bool exchange{true};
};

/** Signed Fock coefficients. Quadratic energies carry another factor 1/2. */
struct JkCoefficients {
  double coulomb{1.0};
  double exchange{-0.5};
};

/** Mathematical request. RHF density includes double occupation; UHF has two
 * unit-occupation spin densities. J consumes the total density; K each spin.
 * Coefficients multiply raw J/K once, during Fock/energy assembly.
 */
struct FockBuildSpec {
  std::uint32_t version{1};
  FockSpin spin{FockSpin::Restricted};
  std::uint32_t derivative_order{1};
  FockTermSpec coulomb{};
  FockTermSpec exchange{true, -0.5, FockOperator::FullRange, 0.0, FockApproximation::Exact};
  bool operator==(const FockBuildSpec&) const = default;
};

struct FockProviderCapabilities {
  bool restricted{true};
  bool unrestricted{true};
  bool full_range{true};
  bool short_range{};
  bool long_range{};
  std::uint32_t maximum_derivative_order{1};
  unsigned maximum_angular_momentum{3};
  bool cartesian{true};
  bool spherical{true};
  bool batching{true};
  bool independent_terms{};
  bool arbitrary_coefficients{};
  bool legacy_adapter_only{};
};

/** Resolved once at preparation. Mathematical and execution identities remain
 * separate fields: an execution variant never authorizes another approximation.
 * Geometry/basis/auxiliary ownership belongs to the enclosing prepared plan.
 */
struct ResolvedFockBuild {
  FockBuildSpec spec{};
  FockBackend backend{FockBackend::Cpu};
  FockSchedule schedule{FockSchedule::CpuReference};
  double screening_tolerance{1.0e-12};
  double metric_relative_threshold{};
  bool legacy_density_fitting{};
  FockPrecision precision{FockPrecision::Float64};
  bool operator==(const ResolvedFockBuild&) const = default;
};

FockProviderCapabilities fock_provider_capabilities(FockApproximation approximation,
                                                    FockBackend backend);
FockBuildSpec make_hf_fock_spec(FockSpin spin,
                                FockApproximation approximation = FockApproximation::Exact);
ResolvedFockBuild resolve_fock_build(FockBuildSpec spec, FockBackend backend,
                                     double screening_tolerance = 1.0e-12,
                                     double metric_relative_threshold = 1.0e-10);
/** Reject forged/noncanonical execution state before calling any provider. */
void validate_resolved_fock_build(const ResolvedFockBuild& strategy);

/** Guard for legacy HF solver entry points, including the force route.
 * Requires the standard complete HF coefficients and first derivatives.
 * This is intentionally stricter than the independent CPU raw consumers.
 */
void require_exact_direct_strategy(const ResolvedFockBuild& strategy, FockSpin spin,
                                   FockBackend backend);

/** Raw unscaled matrices in row-major public AO representation. An absent term
 * has an empty vector. Restricted exchange is in exchange_alpha; beta is empty.
 */
struct DirectJkMatrices {
  std::size_t nbf{};
  std::vector<double> coulomb;
  std::vector<double> exchange_alpha;
  std::vector<double> exchange_beta;
};
struct FockMatrices {
  std::vector<double> alpha;
  std::vector<double> beta;
};

/** Exact CPU reference consumer. ERI is full chemists' [i,j,k,l] row-major;
 * nonsymmetric test densities are accepted without implicit symmetrization.
 */
DirectJkMatrices build_exact_direct_jk(const ResolvedFockBuild& strategy, std::size_t nbf,
                                       std::span<const double> eri, std::span<const double> density,
                                       std::span<const double> beta = {});
FockMatrices assemble_fock(const ResolvedFockBuild& strategy, std::span<const double> hcore,
                           const DirectJkMatrices& jk);
/** Two-electron energy at fixed density, with the same weights as Fock assembly. */
double contract_fock_energy(const ResolvedFockBuild& strategy, const DirectJkMatrices& jk,
                            std::span<const double> density, std::span<const double> beta = {});
/** Fixed-density two-electron energy derivative, excluding one-electron, Pulay,
 * and nuclear-repulsion terms. API forces are the negative of the total gradient.
 */
double contract_exact_direct_energy_derivative(const ResolvedFockBuild& strategy, std::size_t nbf,
                                               std::span<const double> eri_derivative,
                                               std::span<const double> density,
                                               std::span<const double> beta = {});

}  // namespace vibeqc::scf
#endif

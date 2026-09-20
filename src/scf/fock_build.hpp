#ifndef VIBEQC_SCF_FOCK_BUILD_HPP
#define VIBEQC_SCF_FOCK_BUILD_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "runtime/provider_registry.hpp"

namespace vibeqc::scf {

enum class FockSpin { Restricted, Unrestricted };
enum class FockOperator { FullRange, ShortRange, LongRange };
enum class FockApproximation { Exact, DensityFitted, SeminumericalCosx };
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

/** Versioned discrete COSX approximation identity.
 *
 * Version zero means no COSX model is attached. Version one is the current
 * explicit, symmetrized, unfitted and unscreened reference model. The grid
 * prescription lives here rather than inheriting XC quadrature implicitly.
 */
struct FockCosxSpec {
  std::uint32_t version{};
  std::uint32_t grid_version{};
  std::size_t radial_points{};
  std::size_t angular_polar{};
  std::size_t angular_azimuth{};
  unsigned partition_iterations{};
  double coincident_tolerance{};
  std::array<double, 119> element_radii{};
  bool symmetrize{};
  bool overlap_fitting{};
  bool screening{};
  bool operator==(const FockCosxSpec&) const = default;
};

struct FockTermSpec {
  bool present{true};
  double coefficient{1.0};
  FockOperator op{FockOperator::FullRange};
  double omega{};
  FockApproximation approximation{FockApproximation::Exact};
  FockCosxSpec cosx{};
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
  bool available{};
  bool restricted{};
  bool unrestricted{};
  bool full_range{};
  bool short_range{};
  bool long_range{};
  std::uint32_t maximum_derivative_order{};
  unsigned maximum_angular_momentum{};
  bool cartesian{};
  bool spherical{};
  bool batching{};
  bool coulomb{};
  bool exchange{};
  bool independent_terms{};
  bool arbitrary_coefficients{};
  bool legacy_adapter_only{};
  std::uint32_t provider_version{};
};

/** Typed scientific domain retained by each executable-provider registration.
 * Runtime registration owns availability/identity only; Fock semantics stay here. */
struct FockProviderDomain {
  FockApproximation approximation{FockApproximation::Exact};
  FockProviderCapabilities capabilities{};
};
using FockProviderRegistration = runtime::ProviderDescriptor<FockProviderDomain>;

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

/** Registration is the single source of execution availability. Capability
 * queries suppress scientific claims for providers that are not executable in
 * this build, while semantic resolution may still represent such a backend. */
const FockProviderRegistration& fock_provider_registration(FockApproximation approximation,
                                                           FockBackend backend);
FockProviderCapabilities fock_provider_capabilities(FockApproximation approximation,
                                                    FockBackend backend);
void require_fock_provider_executable(FockApproximation approximation, FockBackend backend);
/** Build the explicit COSX v1 model identity independently of XC GridSpec. */
FockCosxSpec make_cosx_v1_spec(std::size_t radial_points = 48, std::size_t angular_polar = 16,
                               std::size_t angular_azimuth = 32, unsigned partition_iterations = 3,
                               double coincident_tolerance = 1.0e-12,
                               std::array<double, 119> element_radii = {});
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
struct FockEnergyComponents {
  double coulomb{};
  double exchange{};
  double total() const noexcept { return coulomb + exchange; }
};

/** Two-electron energy at fixed density, split by physical source. */
FockEnergyComponents contract_fock_energy_components(const ResolvedFockBuild& strategy,
                                                     const DirectJkMatrices& jk,
                                                     std::span<const double> density,
                                                     std::span<const double> beta = {});
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

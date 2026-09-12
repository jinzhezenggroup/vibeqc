#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "scf/density_fitting.hpp"

namespace vibeqc::scf {

enum class DensityFactorSpin { Restricted, Alpha, Beta };

/** Native SCF provenance, independent of dimensions and orbital values.
 * The prepared source assigns basis/reference IDs; its SCF owner advances
 * orbital and density generations together only for an unmixed orbital state.
 * Zero IDs never authorize the occupied-factor path. External/response density
 * calls supply no factor and retain the dense-density implementation.
 */
struct DensityFactorIdentity {
  std::uint64_t basis{}, reference{}, orbital_generation{}, density_generation{};
  bool operator==(const DensityFactorIdentity&) const = default;
};

/** Immutable host snapshot of D = B B^T for one spin channel.
 * B uses [AO, occupied] row-major layout and includes sqrt(occupation).
 * The density witness uses the SCF occupation*C*C evaluation order, allowing
 * exact validation without a density diagonalization or an approximate fit.
 */
class OccupiedDensityFactor {
 public:
  /** Coefficients contain exactly nbf*rank values; occupations contain rank.
   * This first contract accepts occupation 2 for RHF or 1 for each UHF spin.
   * Rank zero is a valid empty spin block. Fractional occupations are rejected
   * until their solver and response conventions are separately validated.
   */
  OccupiedDensityFactor(DensityFactorIdentity identity, DensityFactorSpin spin, std::size_t nbf,
                        std::span<const double> coefficients, std::span<const double> occupations);

  std::size_t nbf() const noexcept { return nbf_; }
  std::size_t rank() const noexcept { return occupations_.size(); }
  DensityFactorSpin spin() const noexcept { return spin_; }
  const DensityFactorIdentity& identity() const noexcept { return identity_; }
  std::span<const double> values() const noexcept { return values_; }
  std::span<const double> occupations() const noexcept { return occupations_; }
  std::span<const double> density() const noexcept { return density_; }
  /** Exact witness comparison rejects stale, damped, mixed or external D.
   * Different floating-point construction orders conservatively fall back;
   * dimensions or generation labels alone never establish density equality.
   */
  bool matches(DensityFactorIdentity identity, DensityFactorSpin spin,
               std::span<const double> density) const noexcept;

 private:
  DensityFactorIdentity identity_;
  DensityFactorSpin spin_;
  std::size_t nbf_;
  std::vector<double> occupations_, values_, density_;
};

/** Tiny independent occupied RI-K reference in the established L[i,j,Q] layout.
 * For each Q form T[i,o] = sum_j L[i,j,Q] B[j,o], then K += T T^T.
 * Scratch is nbf*rank; no additional full three-center tensor is allocated.
 * Missing/incompatible factors return nullopt, asking the caller to use dense
 * K. J and RHF/UHF Fock prefactors remain in their existing consumers.
 */
std::optional<std::vector<double>> occupied_density_fitting_exchange(
    const DensityFittingThreeCenter& three_center, const OccupiedDensityFactor* factor,
    DensityFactorIdentity identity, DensityFactorSpin spin, std::span<const double> density);

}  // namespace vibeqc::scf

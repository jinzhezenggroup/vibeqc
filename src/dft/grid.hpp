#ifndef VIBEQC_DFT_GRID_HPP
#define VIBEQC_DFT_GRID_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "core/types.hpp"

namespace vibeqc::dft {

/** Table-free molecular quadrature. Version 1 preserves the historical
 * one-Bohr fallback exactly. Version 2 is a resolved production contract and
 * requires an explicit radius for every element that is materialized. */
struct GridSpec {
  std::uint32_t version{1};
  std::size_t radial_points{48};
  std::size_t angular_polar{16};
  std::size_t angular_azimuth{32};
  unsigned partition_iterations{3};
  double coincident_tolerance{1.0e-12};
  /** In v1 zero selects the one-Bohr reference radius. In v2 zero is
   * unsupported/fail-closed. Slot zero is unused. Fixed storage preserves
   * immutable identity without borrowed pointers. */
  std::array<double, 119> element_radii{};
  bool operator==(const GridSpec&) const = default;
};

/** Pure prescription validation, shared by preparation and materialization. */
void validate_grid_spec(const GridSpec& spec);

/** Owned molecular grid in atom-radial-polar-azimuth order. The ordinary
 * constructor is the independent CPU reference; CUDA preparation uses the
 * generated, bounded factory and retains the same host export contract. */
class MolecularGrid {
 public:
  explicit MolecularGrid(const core::System& system, GridSpec spec = {});
  /** Materialize with compiler-generated CUDA kernels. No CPU partition
   * fallback; throws on unsupported input, allocation or normalization error. */
  static MolecularGrid from_cuda(const core::System& system, GridSpec spec, int device);

  const GridSpec& spec() const noexcept { return spec_; }
  const core::System& system() const noexcept { return system_; }
  std::size_t point_count() const noexcept { return weights_.size(); }
  const std::vector<double>& points() const noexcept { return points_; }
  const std::vector<double>& weights() const noexcept { return weights_; }
  const std::vector<std::uint32_t>& owners() const noexcept { return owners_; }
  /** Explicit derivative export of the atomic measure before partitioning.
   * Reconstruct only quadrature rules, not Becke weights, on request; energy
   * execution retains no additional point-sized array. */
  std::vector<double> atomic_weights() const;
  /** Contract dE/dw(point) directly with the analytic nuclear response of
   * the exact materialized Becke weights. Grid points translate with their
   * owner atoms; this routine differentiates only the partition weights, not
   * point coordinates or the element/radial/angular atomic measure. */
  std::vector<double> contract_weight_derivative(std::span<const double> weight_sensitivity) const;

 private:
  struct Deferred {};
  MolecularGrid(const core::System& system, GridSpec spec, Deferred);
  static std::pair<std::vector<double>, std::vector<double>> legendre_rule(std::size_t count);
  core::System system_;
  GridSpec spec_;
  std::vector<double> points_;
  std::vector<double> weights_;
  std::vector<std::uint32_t> owners_;
};

/** Pure shape query for all temporary CUDA quadrature allocations. */
std::size_t cuda_quadrature_bytes(std::size_t atoms, std::size_t points);

}  // namespace vibeqc::dft

#endif

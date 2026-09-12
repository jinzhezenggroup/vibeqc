#ifndef VIBEQC_DFT_GRID_HPP
#define VIBEQC_DFT_GRID_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/types.hpp"

namespace vibeqc::dft {

/** Table-free molecular quadrature matching GridSpec version 1. */
struct GridSpec {
  std::uint32_t version{1};
  std::size_t radial_points{48};
  std::size_t angular_polar{16};
  std::size_t angular_azimuth{32};
  unsigned partition_iterations{3};
  double coincident_tolerance{1.0e-12};
  bool operator==(const GridSpec&) const = default;
};

/** Materialized CPU reference grid in atom-radial-polar-azimuth order. */
class MolecularGrid {
 public:
  explicit MolecularGrid(const core::System& system, GridSpec spec = {});

  const GridSpec& spec() const noexcept { return spec_; }
  const core::System& system() const noexcept { return system_; }
  std::size_t point_count() const noexcept { return weights_.size(); }
  const std::vector<double>& points() const noexcept { return points_; }
  const std::vector<double>& weights() const noexcept { return weights_; }
  const std::vector<std::uint32_t>& owners() const noexcept { return owners_; }

 private:
  core::System system_;
  GridSpec spec_;
  std::vector<double> points_;
  std::vector<double> weights_;
  std::vector<std::uint32_t> owners_;
};

}  // namespace vibeqc::dft

#endif

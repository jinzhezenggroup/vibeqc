#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "scf/cuda/arena.hpp"
#include "scf/cuda_batch.hpp"
#include "vibeqc/vibeqc.h"

namespace {

void check_h2_cartesian_rhf() {
  using vibeqc::scf::small_hf_cuda_resource_layout_v2;
  using vibeqc::scf::cuda_execution::ArenaLayout;
  using vibeqc::scf::cuda_execution::make_layout;

  const std::array<std::uint8_t, 2> angular{0, 0};
  const std::array<std::size_t, 2> primitives{3, 3};
  std::size_t queried = 0, plan_bytes = 0;
  assert(small_hf_cuda_resource_layout_v2(
      2, 2, 2, angular.data(), primitives.data(), angular.size(), 8, 1,
      VIBEQC_PRECISION_FP64, 1e-10, 1e-12, queried, plan_bytes));

  ArenaLayout energy{}, force{};
  assert(make_layout(1, 2, 2, 2, 2, 3, 1, 0, 27, 0, 0, 0, 0, 0, 0, 0, 6, 8, 0, 1, true,
                     false, false, false, false, false, false, energy));
  // Three s-s pairs are retained as the possible PSSS ket list even though
  // this all-s topology has no PSSS bra tasks. Six exact ssss quartets each
  // occupy one compiler-valid tile.
  assert(make_layout(1, 2, 2, 2, 2, 3, 1, 0, 27, 0, 3, 6, 0, 6, 0, 0, 6, 8, 0, 1, false,
                     false, false, false, false, false, false, force));
  assert(queried == std::max(energy.bytes, force.bytes));
  assert(plan_bytes != 0);
}

void check_spherical_d_uhf() {
  using vibeqc::scf::small_hf_cuda_resource_layout_v2;
  using vibeqc::scf::cuda_execution::ArenaLayout;
  using vibeqc::scf::cuda_execution::make_layout;

  const std::array<std::uint8_t, 1> angular{2};
  const std::array<std::size_t, 1> primitives{1};
  std::size_t queried = 0, plan_bytes = 0;
  assert(small_hf_cuda_resource_layout_v2(
      5, 6, 1, angular.data(), primitives.data(), angular.size(), 8, 2,
      VIBEQC_PRECISION_FP64, 1e-10, 1e-12, queried, plan_bytes));

  ArenaLayout energy{}, force{};
  assert(make_layout(1, 5, 6, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 8, 0, 2, true,
                     false, false, false, false, false, false, energy));
  // One spherical d shell expands to six Cartesian AOs. Its single dddd
  // quartet is one exact tile and requires transformed-direct storage.
  assert(make_layout(1, 5, 6, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 8, 0, 2, false,
                     true, false, false, false, false, false, force));
  assert(queried == std::max(energy.bytes, force.bytes));
  assert(plan_bytes != 0);
}

}  // namespace

int main() {
  check_h2_cartesian_rhf();
  check_spherical_d_uhf();
  return 0;
}

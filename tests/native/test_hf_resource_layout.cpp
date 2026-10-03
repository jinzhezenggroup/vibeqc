// Resource admission checks must execute in Release qualification builds too.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "generativeqc/generativeqc.h"
#include "molecule/basis.hpp"
#include "scf/cuda/arena.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_batch.hpp"

namespace {

void check_h2_cartesian_rhf() {
  using generativeqc::scf::small_hf_cuda_resource_layout_v2;
  using generativeqc::scf::cuda_execution::ArenaLayout;
  using generativeqc::scf::cuda_execution::make_layout;

  const std::array<std::uint8_t, 2> angular{0, 0};
  const std::array<std::size_t, 2> primitives{3, 3};
  std::size_t queried = 0, plan_bytes = 0;
  assert(small_hf_cuda_resource_layout_v2(2, 2, 2, angular.data(), primitives.data(),
                                          angular.size(), 8, 1, GENERATIVEQC_PRECISION_FP64, 1e-10,
                                          1e-12, queried, plan_bytes));

  ArenaLayout energy{}, force{};
  assert(make_layout(1, 2, 2, 2, 2, 3, 1, 0, 27, 0, 0, 0, 0, 0, 0, 0, 6, 8, 0, 1, true, false,
                     false, false, false, false, false, true, energy));
  // Three s-s pairs are retained as the possible PSSS ket list even though
  // this all-s topology has no PSSS bra tasks. Six exact ssss quartets each
  // occupy one compiler-valid tile.
  assert(make_layout(1, 2, 2, 2, 2, 3, 1, 0, 27, 0, 3, 6, 0, 6, 0, 0, 6, 8, 0, 1, false, false,
                     false, false, false, false, false, true, force));
  assert(queried == std::max(energy.bytes, force.bytes));
  assert(plan_bytes != 0);
}

void check_spherical_d_uhf() {
  using generativeqc::scf::small_hf_cuda_resource_layout_v2;
  using generativeqc::scf::cuda_execution::ArenaLayout;
  using generativeqc::scf::cuda_execution::make_layout;

  const std::array<std::uint8_t, 1> angular{2};
  const std::array<std::size_t, 1> primitives{1};
  std::size_t queried = 0, plan_bytes = 0;
  assert(small_hf_cuda_resource_layout_v2(5, 6, 1, angular.data(), primitives.data(),
                                          angular.size(), 8, 2, GENERATIVEQC_PRECISION_FP64, 1e-10,
                                          1e-12, queried, plan_bytes));

  ArenaLayout energy{}, force{};
  assert(make_layout(1, 5, 6, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 8, 0, 2, true, false, false,
                     false, false, false, false, true, energy));
  // One spherical d shell expands to six Cartesian AOs. Its single dddd
  // quartet is one exact tile and requires transformed-direct storage.
  assert(make_layout(1, 5, 6, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 8, 0, 2, false, true, false,
                     false, false, false, false, true, force));
  assert(queried == std::max(energy.bytes, force.bytes));
  assert(plan_bytes != 0);
}

void check_small_spherical_force_packs_direct_transform() {
  generativeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.5, 1.0}}},
      {0, 2, {{0.8, 1.0}}},
      {1, 0, {{1.2, 1.0}}},
  };
  system.charge = 1;
  system.multiplicity = 1;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  assert(generativeqc::molecule::validate_and_normalize(system, detail) ==
         GENERATIVEQC_STATUS_SUCCESS);

  std::vector<const std::vector<double>*> no_warm(1, nullptr);
  generativeqc::scf::cuda_execution::HostBatch energy_host;
  assert(generativeqc::scf::cuda_execution::pack_host_batch({system}, no_warm, energy_host, false,
                                                            false, false));
  assert(energy_host.nbf == 7 && energy_host.direct_nbf == 8);
  assert(energy_host.ao_to_direct_transform.empty());

  generativeqc::scf::cuda_execution::HostBatch force_host;
  assert(generativeqc::scf::cuda_execution::pack_host_batch({system}, no_warm, force_host, false,
                                                            false, true));
  assert(force_host.nbf == 7 && force_host.direct_nbf == 8);
  assert(force_host.ao_to_direct_transform.size() == 7 * 8);
}

void check_df_values_pack_g_metadata_without_scf_work() {
  using namespace generativeqc::scf::cuda_execution;
  generativeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, 0.0}}};
  system.shells = {{0, 0, {{0.5, 1.0}}}, {0, 4, {{0.8, 1.0}}}};
  system.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  std::string detail;
  assert(generativeqc::molecule::validate_and_normalize(system, detail) ==
         GENERATIVEQC_STATUS_SUCCESS);
  const std::vector<const std::vector<double>*> no_warm(1, nullptr);
  HostBatch scf, values;
  // g metadata is admitted only for the explicit DF value owner. Ordinary
  // SCF remains bounded by its existing f recurrences and three-term AO ABI.
  assert(!pack_host_batch({system}, no_warm, scf, false, true));
  assert(
      pack_host_batch({system}, no_warm, values, false, true, false, HostBasisPacking::DfValues));
  assert(values.nbf == 16 && values.direct_nbf == 16);
  assert(values.ao_term_counts.size() == 16);
  assert(std::all_of(values.ao_term_counts.begin(), values.ao_term_counts.end(),
                     [](auto count) { return count == 1; }));
  assert(values.shell_pair_first.empty() && values.psss_resident_tasks.empty());
  assert(values.psss_resident_ket_pairs.empty() && values.warm_density.empty());
  assert(values.occupied.empty() && values.warm_mask.empty());
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  HostBatch invalid;
  assert(
      !pack_host_batch({system}, no_warm, invalid, false, true, false, HostBasisPacking::DfValues));
}

}  // namespace

int main() {
  check_h2_cartesian_rhf();
  check_spherical_d_uhf();
  check_small_spherical_force_packs_direct_transform();
  check_df_values_pack_g_metadata_without_scf_work();
  return 0;
}

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/direct_task_layout.hpp"
#include "scf/rhf.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(double actual,
                   double expected,
                   double tolerance,
                   const char* message) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(message) + ": actual=" +
                             std::to_string(actual) +
                             " expected=" + std::to_string(expected));
  }
}

std::size_t matrix_index(std::size_t row, std::size_t column, std::size_t n) {
  return row * n + column;
}

std::size_t eri_index(std::size_t i,
                      std::size_t j,
                      std::size_t k,
                      std::size_t l,
                      std::size_t n) {
  return ((i * n + j) * n + k) * n + l;
}

vibeqc::core::System hydrogen_sp_dimer() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.2, 1.0}}}, {0, 1, {{0.7, 1.0}}},
      {1, 0, {{1.2, 1.0}}}, {1, 1, {{0.7, 1.0}}},
  };
  system.multiplicity = 1;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "s/p system normalization failed");
  return system;
}

vibeqc::core::System helium_hydrogen_sdf() {
  vibeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{1.5, 1.0}}},
      {0, 2, {{0.8, 1.0}}},
      {0, 3, {{0.6, 1.0}}},
      {1, 0, {{1.2, 1.0}}},
  };
  system.charge = 1;
  system.multiplicity = 1;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "s/d/f system normalization failed");
  return system;
}

/** Reproduce the shell/pair cardinality of the issue-52 32-water case. */
vibeqc::core::System bounded_streaming_topology() {
  vibeqc::core::System system;
  system.atoms = {{2, {0.0, 0.0, 0.0}}};
  system.shells.reserve(384);
  for (std::size_t shell = 0; shell < 384; ++shell) {
    system.shells.push_back(
        {0, 0, {{1.0 + 1.0e-4 * static_cast<double>(shell), 1.0}}});
  }
  system.multiplicity = 1;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) ==
              VIBEQC_STATUS_SUCCESS,
          "large streaming topology normalization failed");
  return system;
}

vibeqc::scf::detail::DirectQuartetTaskLayout direct_task_layout(
    const vibeqc::core::System& system) {
  std::vector<std::int64_t> shell_ao_offsets{0};
  std::vector<std::uint8_t> shell_angular;
  for (const vibeqc::core::Shell& shell : system.shells) {
    shell_angular.push_back(
        static_cast<std::uint8_t>(shell.angular_momentum));
    shell_ao_offsets.push_back(
        shell_ao_offsets.back() + static_cast<std::int64_t>(
            vibeqc::molecule::cartesian_count(shell.angular_momentum)));
  }
  std::vector<std::int32_t> shell_pair_first;
  std::vector<std::int32_t> shell_pair_second;
  for (std::size_t first = 0; first < system.shells.size(); ++first) {
    for (std::size_t second = 0; second <= first; ++second) {
      shell_pair_first.push_back(static_cast<std::int32_t>(first));
      shell_pair_second.push_back(static_cast<std::int32_t>(second));
    }
  }
  const std::vector<std::int64_t> system_shell_pair_offsets{
      0, static_cast<std::int64_t>(shell_pair_first.size())};
  vibeqc::scf::detail::DirectQuartetTaskLayout layout;
  require(vibeqc::scf::detail::make_direct_quartet_task_layout(
              shell_ao_offsets, shell_angular, system_shell_pair_offsets,
              shell_pair_first, shell_pair_second, layout),
          "direct-J/K task layout rejected a valid shell topology");
  return layout;
}

}  // namespace

int main() {
  try {
    require(vibeqc::scf::detail::kDirectFixedTopologyTileLimit ==
                536870911ULL &&
                !vibeqc::scf::detail::
                    direct_topology_requires_bounded_streaming(536870911ULL) &&
                vibeqc::scf::detail::
                    direct_topology_requires_bounded_streaming(536870912ULL),
            "bounded direct-topology crossover is inconsistent");
    require(vibeqc::scf::detail::bounded_direct_queue_refill_count(0) == 0 &&
                vibeqc::scf::detail::bounded_direct_queue_refill_count(256) ==
                    1 &&
                vibeqc::scf::detail::bounded_direct_queue_refill_count(257) ==
                    2 &&
                vibeqc::scf::detail::bounded_direct_queue_refill_count(
                    2732120160ULL) == 10672345ULL,
            "bounded direct queue does not resume after capacity refills");
    {
      const std::vector<std::int64_t> shell_ao_offsets{0, 1};
      const std::vector<std::int64_t> system_pair_offsets{0, 1};
      const std::vector<std::int32_t> shell_pair{0};
      vibeqc::scf::detail::DirectQuartetTaskLayout invalid_layout;
      require(!vibeqc::scf::detail::make_direct_quartet_task_layout(
                  shell_ao_offsets, {4}, system_pair_offsets, shell_pair,
                  shell_pair, invalid_layout),
              "direct-J/K task layout accepted angular momentum above f");
      require(!vibeqc::scf::detail::make_direct_quartet_task_layout(
                  shell_ao_offsets, {}, system_pair_offsets, shell_pair,
                  shell_pair, invalid_layout),
              "direct-J/K task layout accepted missing shell angular data");
    }
    const vibeqc::core::System system = hydrogen_sp_dimer();
    require(vibeqc::molecule::ao_count(system) == 8,
            "s/p shell expansion produced the wrong AO count");
    const vibeqc::scf::CudaRhfBasisLayoutStats layout =
        vibeqc::scf::inspect_rhf_cuda_basis_layout({system});
    require(layout.shell_count == 4 && layout.shell_pair_count == 10 &&
                layout.shell_quartet_count == 55 &&
                layout.ao_count == 8,
            "CUDA basis layout lost the shell-to-AO topology");
    require(layout.unique_primitive_count == 4,
            "CUDA basis layout duplicated shell primitives");
    require(layout.expanded_primitive_references == 8,
            "expanded primitive diagnostic has the wrong Cartesian count");
    require(layout.device_basis_bytes == 796,
            "CUDA basis topology payload changed unexpectedly");
    require(!layout.bounded_direct_streaming &&
                layout.direct_descriptor_capacity == 0,
            "small CUDA basis unexpectedly selected bounded streaming");
    const vibeqc::scf::CudaRhfBasisLayoutStats streaming_layout =
        vibeqc::scf::inspect_rhf_cuda_basis_layout(
            {bounded_streaming_topology()});
    require(streaming_layout.shell_count == 384 &&
                streaming_layout.shell_pair_count == 73920 &&
                streaming_layout.shell_quartet_count == 2732120160ULL &&
                streaming_layout.bounded_direct_streaming &&
                streaming_layout.direct_descriptor_capacity ==
                    vibeqc::scf::detail::kBoundedDirectQueueCapacity,
            "issue-52 topology did not select the bounded descriptor queue");
    const vibeqc::scf::CudaRhfBasisLayoutStats sdf_layout =
        vibeqc::scf::inspect_rhf_cuda_basis_layout({helium_hydrogen_sdf()});
    require(sdf_layout.shell_count == 4 &&
                sdf_layout.shell_pair_count == 10 &&
                sdf_layout.shell_quartet_count == 55 &&
                sdf_layout.ao_count == 18,
            "s/d/f CUDA shell-to-AO topology is inconsistent");
    require(sdf_layout.unique_primitive_count == 4 &&
                sdf_layout.expanded_primitive_references == 18,
            "s/d/f primitive storage was expanded per Cartesian component");
    require(sdf_layout.device_basis_bytes == 1326,
            "s/d/f CUDA basis topology payload changed unexpectedly");
    const vibeqc::scf::detail::DirectQuartetTaskLayout sdf_tasks =
        direct_task_layout(helium_hydrogen_sdf());
    require(sdf_tasks.shell_quartet_count == 55 &&
                sdf_tasks.exact_tile_count == 100 &&
                sdf_tasks.maximum_tiles_per_shell_quartet == 15 &&
                sdf_tasks.uniform_tile_count == 825,
            "Cartesian s/d/f direct-J/K tile compaction is inconsistent");
    require(sdf_tasks.angular_order_tile_counts ==
                std::array<std::size_t, 13>{6, 0, 6, 6, 6, 7, 8,
                                            6, 11, 11, 13, 13, 7} &&
                sdf_tasks.angular_order_tile_offsets ==
                std::array<std::size_t, 14>{0, 6, 6, 12, 18, 24, 31,
                                            39, 45, 56, 67, 80, 93, 100},
            "Cartesian direct-J/K angular buckets are inconsistent");
    std::array<std::size_t, 13> sdf_shell_class_orders{};
    for (std::size_t shell_class = 0;
         shell_class < vibeqc::scf::detail::kDirectQuartetShellClassCount;
         ++shell_class) {
      const std::size_t order =
          vibeqc::scf::detail::direct_quartet_shell_class_angular_order(
              shell_class);
      require(order < sdf_shell_class_orders.size(),
              "direct-J/K shell class decoded an invalid angular order");
      sdf_shell_class_orders[order] +=
          sdf_tasks.shell_class_tile_counts[shell_class];
    }
    require(sdf_tasks.shell_class_tile_offsets.back() ==
                sdf_tasks.exact_tile_count &&
                sdf_shell_class_orders == sdf_tasks.angular_order_tile_counts &&
                sdf_tasks.shell_class_tile_counts[
                    vibeqc::scf::detail::direct_quartet_shell_class(0, 0, 0, 0)] ==
                    6 &&
                sdf_tasks.shell_class_tile_counts[
                    vibeqc::scf::detail::direct_quartet_shell_class(3, 3, 3, 3)] ==
                    7,
            "Cartesian exact shell-class partitions are inconsistent");
    vibeqc::core::System spherical_sdf = helium_hydrogen_sdf();
    spherical_sdf.basis_representation = VIBEQC_BASIS_SPHERICAL;
    const vibeqc::scf::CudaRhfBasisLayoutStats spherical_layout =
        vibeqc::scf::inspect_rhf_cuda_basis_layout({spherical_sdf});
    require(spherical_layout.shell_count == 4 &&
                spherical_layout.shell_pair_count == 10 &&
                spherical_layout.shell_quartet_count == 55 &&
                spherical_layout.ao_count == 14,
            "spherical s/d/f CUDA shell-to-AO topology is inconsistent");
    require(spherical_layout.unique_primitive_count == 4 &&
                spherical_layout.expanded_primitive_references == 18 &&
                spherical_layout.device_basis_bytes == 1174,
            "spherical CUDA basis metadata is not compact and shell-owned");
    const vibeqc::scf::detail::DirectQuartetTaskLayout spherical_tasks =
        direct_task_layout(spherical_sdf);
    require(spherical_tasks.shell_quartet_count == 55 &&
                spherical_tasks.exact_tile_count == 100 &&
                spherical_tasks.maximum_tiles_per_shell_quartet == 15 &&
                spherical_tasks.uniform_tile_count == 825,
            "spherical Cartesian-source tile compaction is inconsistent");
    require(spherical_tasks.angular_order_tile_counts ==
                std::array<std::size_t, 13>{6, 0, 6, 6, 6, 7, 8,
                                            6, 11, 11, 13, 13, 7} &&
                spherical_tasks.angular_order_tile_offsets ==
                std::array<std::size_t, 14>{0, 6, 6, 12, 18, 24, 31,
                                            39, 45, 56, 67, 80, 93, 100},
            "spherical Cartesian-source angular buckets are inconsistent");
    std::array<std::size_t, 13> spherical_shell_class_orders{};
    for (std::size_t shell_class = 0;
         shell_class < vibeqc::scf::detail::kDirectQuartetShellClassCount;
         ++shell_class) {
      spherical_shell_class_orders[
          vibeqc::scf::detail::direct_quartet_shell_class_angular_order(
              shell_class)] +=
          spherical_tasks.shell_class_tile_counts[shell_class];
    }
    require(spherical_tasks.shell_class_tile_offsets.back() ==
                spherical_tasks.exact_tile_count &&
                spherical_shell_class_orders ==
                    spherical_tasks.angular_order_tile_counts,
            "spherical exact shell-class partitions are inconsistent");
    const vibeqc::integrals::IntegralData integrals =
        vibeqc::integrals::build_cartesian_integrals(system);
    require(integrals.nbf == 8, "integral engine reported the wrong AO count");

    // Values were generated independently with PySCF 2.11/libcint using
    // cart=True and the raw basis {s: (1.2,1), p: (0.7,1)} on each hydrogen.
    require_close(std::accumulate(integrals.overlap.begin(),
                                  integrals.overlap.end(), 0.0),
                  10.256697920226276, 2.0e-12,
                  "Cartesian overlap checksum differs from libcint");
    require_close(std::accumulate(integrals.hcore.begin(),
                                  integrals.hcore.end(), 0.0),
                  -2.580918402607146, 3.0e-12,
                  "Cartesian core-Hamiltonian checksum differs from libcint");
    require_close(std::accumulate(integrals.eri.begin(), integrals.eri.end(),
                                  0.0),
                  83.63765625818158, 2.0e-11,
                  "Cartesian ERI checksum differs from libcint");

    const std::size_t n = integrals.nbf;
    require_close(integrals.overlap[matrix_index(0, 7, n)],
                  -0.5894285335123269, 2.0e-13,
                  "s-p overlap ordering/sign differs from libcint");
    require_close(integrals.hcore[matrix_index(3, 7, n)],
                  -0.24217477539847332, 3.0e-13,
                  "p-p core Hamiltonian differs from libcint");
    require_close(integrals.eri[eri_index(0, 1, 0, 1, n)],
                  0.12127971271176609, 3.0e-13,
                  "mixed s/p ERI differs from libcint");
    require_close(integrals.eri[eri_index(0, 7, 4, 3, n)],
                  -0.3003230202786651, 3.0e-13,
                  "four-center s/p ERI ordering/sign differs from libcint");

    std::cout << "validated 8-AO Cartesian s/p integrals against PySCF/libcint\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

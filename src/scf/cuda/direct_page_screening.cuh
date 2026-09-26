#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_queue_index.cuh"

namespace vibeqc::scf::cuda_execution {

/**
 * Bound the density factors for one exact pair-class page.
 *
 * The class-major stream preserves topology insertion order; it is not
 * sorted by the geometry-dependent Schwarz bounds. These class maxima
 * therefore provide a conservative coarse rejection for the current ket
 * only. They must never terminate the remaining row: a later ket may have
 * a larger Schwarz bound and survive the exact shell-quartet predicate.
 *
 * For exchange, enumerate both orientations of the low pair class so
 * every physical ket orientation remains covered by the class-level bound.
 */
struct BoundedPageDensityTails {
  double fock{};
  double force{};
};

__device__ __forceinline__ double bounded_page_class_density_bound(
    const GeneratedShellPairStream& topology, std::int32_t system, unsigned pair_class) {
  return pair_class < detail::kDirectShellPairClassCount
             ? topology.system_pair_density_bounds[static_cast<std::size_t>(system) *
                                                       detail::kDirectShellPairClassCount +
                                                   pair_class]
             : 0.0;
}

__device__ __forceinline__ BoundedPageDensityTails bounded_page_density_tails(
    const DeviceBatch& batch, const GeneratedShellPairStream& topology, std::int32_t system,
    std::uint32_t bra_pair, unsigned high_pair_class, unsigned low_pair_class) {
  const bool has_pair_class_bounds = topology.system_pair_density_bounds != nullptr;
  const bool has_system_bound = topology.system_density_bounds != nullptr;
  if (!has_pair_class_bounds) {
    if (!has_system_bound) return {};
    const double maximum = topology.system_density_bounds[system];
    return {maximum, maximum * maximum};
  }

  const unsigned bra_first_shell = static_cast<unsigned>(batch.shell_pair_first[bra_pair]);
  const unsigned bra_second_shell = static_cast<unsigned>(batch.shell_pair_second[bra_pair]);
  const unsigned bra_first_angular = batch.shell_angular[bra_first_shell];
  const unsigned bra_second_angular = batch.shell_angular[bra_second_shell];
  const unsigned low_high = direct_triangular_class_high(low_pair_class);
  const unsigned low_low = low_pair_class - low_high * (low_high + 1U) / 2U;

  double fock_maximum = fmax(bounded_page_class_density_bound(topology, system, high_pair_class),
                             bounded_page_class_density_bound(topology, system, low_pair_class));
  double force_maximum = bounded_page_class_density_bound(topology, system, high_pair_class) *
                         bounded_page_class_density_bound(topology, system, low_pair_class);
  // The pair-class maxima include both Coulomb and exchange density terms.
  // RHF exchange carries a one-half coefficient, while UHF does not; using
  // the larger UHF factor remains a valid conservative tail for both.
  for (unsigned orientation = 0U; orientation < 2U; ++orientation) {
    const unsigned ket_first_angular = orientation == 0U ? low_high : low_low;
    const unsigned ket_second_angular = orientation == 0U ? low_low : low_high;
    const unsigned ac = direct_shell_pair_class_cuda(bra_first_angular, ket_first_angular);
    const unsigned ad = direct_shell_pair_class_cuda(bra_first_angular, ket_second_angular);
    const unsigned bc = direct_shell_pair_class_cuda(bra_second_angular, ket_first_angular);
    const unsigned bd = direct_shell_pair_class_cuda(bra_second_angular, ket_second_angular);
    fock_maximum =
        fmax(fock_maximum, fmax(fmax(bounded_page_class_density_bound(topology, system, ac),
                                     bounded_page_class_density_bound(topology, system, ad)),
                                fmax(bounded_page_class_density_bound(topology, system, bc),
                                     bounded_page_class_density_bound(topology, system, bd))));
    force_maximum =
        fmax(force_maximum, fmax(bounded_page_class_density_bound(topology, system, ac) *
                                     bounded_page_class_density_bound(topology, system, bd),
                                 bounded_page_class_density_bound(topology, system, ad) *
                                     bounded_page_class_density_bound(topology, system, bc)));
  }
  return {fock_maximum, force_maximum};
}

}  // namespace vibeqc::scf::cuda_execution

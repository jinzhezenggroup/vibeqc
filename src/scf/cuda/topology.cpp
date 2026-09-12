#include "scf/cuda/topology.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "molecule/basis.hpp"
#include "scf/cuda/checked_layout.hpp"
#include "scf/cuda/direct_constants.hpp"

namespace vibeqc::scf::cuda_execution {

/** Prepare immutable batch metadata without CUDA compilation; numerical kernels consume the
 * resulting views. */
bool pack_host_batch(const std::vector<core::System>& systems,
                     const std::vector<const std::vector<double>*>& initial_densities,
                     HostBatch& host, bool unrestricted, bool matrix_direct) {
  if (systems.empty() || systems.size() != initial_densities.size()) return false;
  if (std::any_of(systems.begin(), systems.end(),
                  [](const auto& s) { return !s.ecp_terms.empty(); }))
    host.ecp_systems = systems;
  host.nbf = molecule::ao_count(systems.front());
  host.direct_nbf = molecule::cartesian_ao_count(systems.front());
  host.spin_count = unrestricted ? 2 : 1;
  if (host.nbf == 0 || host.direct_nbf == 0 ||
      host.nbf > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      host.direct_nbf > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      systems.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    return false;
  const std::size_t matrix_size = host.nbf * host.nbf;
  host.atom_offsets.push_back(0);
  host.system_shell_offsets.push_back(0);
  host.shell_ao_offsets.push_back(0);
  host.shell_direct_ao_offsets.push_back(0);
  host.shell_primitive_offsets.push_back(0);
  host.system_shell_pair_offsets.push_back(0);
  host.system_shell_quartet_offsets.push_back(0);
  host.system_shell_pair_block_offsets.push_back(0);
  host.system_shell_pair_block_quartet_offsets.push_back(0);
  host.shell_pair_primitive_offsets.push_back(0);
  host.warm_density.resize(systems.size() * host.spin_count * matrix_size, 0.0);
  if (!matrix_direct && host.direct_nbf != host.nbf && host.nbf > kPersistentEriAoLimit) {
    host.ao_to_direct_transform.resize(systems.size() * host.nbf * host.direct_nbf, 0.0);
  }
  for (std::size_t system_index = 0; system_index < systems.size(); ++system_index) {
    const core::System& system = systems[system_index];
    if (molecule::ao_count(system) != host.nbf ||
        molecule::cartesian_ao_count(system) != host.direct_nbf || system.electron_count <= 0) {
      return false;
    }
    const int spin_excess = static_cast<int>(system.multiplicity) - 1;
    if ((!unrestricted && (system.electron_count % 2 != 0 || system.multiplicity != 1)) ||
        (unrestricted && (spin_excess < 0 || spin_excess > system.electron_count ||
                          ((system.electron_count + spin_excess) & 1) != 0))) {
      return false;
    }
    const std::int64_t atom_base = static_cast<std::int64_t>(host.atomic_numbers.size());
    for (const core::Atom& atom : system.atoms) {
      host.atom_systems.push_back(static_cast<std::int32_t>(system_index));
      host.atomic_numbers.push_back(atom.ionic_charge());
      host.positions.insert(host.positions.end(), atom.position.begin(), atom.position.end());
    }
    host.atom_offsets.push_back(static_cast<std::int64_t>(host.atomic_numbers.size()));
    const std::size_t system_ao_begin = host.ao_shells.size();
    const std::size_t system_direct_ao_begin = host.direct_ao_shells.size();
    const std::size_t system_shell_begin = host.shell_atoms.size();
    for (const core::Shell& shell : system.shells) {
      if (shell.angular_momentum > kMaximumAngularMomentum ||
          shell.atom_index >= system.atoms.size())
        return false;
      if (host.shell_atoms.size() >=
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return false;
      }
      const std::int32_t shell_index = static_cast<std::int32_t>(host.shell_atoms.size());
      host.shell_atoms.push_back(atom_base + static_cast<std::int64_t>(shell.atom_index));
      host.shell_angular.push_back(static_cast<std::uint8_t>(shell.angular_momentum));
      for (const core::Primitive& primitive : shell.primitives) {
        host.primitive_exponents.push_back(primitive.exponent);
        host.primitive_coefficients.push_back(primitive.coefficient);
      }
      host.shell_primitive_offsets.push_back(
          static_cast<std::int64_t>(host.primitive_exponents.size()));
      const std::vector<molecule::CartesianComponent> cartesian_components =
          molecule::cartesian_components(shell.angular_momentum);
      const std::size_t direct_shell_begin = host.direct_ao_shells.size() - system_direct_ao_begin;
      for (const molecule::CartesianComponent& component : cartesian_components) {
        host.direct_ao_shells.push_back(shell_index);
        host.direct_ao_angular.push_back(static_cast<std::uint8_t>(component[0]));
        host.direct_ao_angular.push_back(static_cast<std::uint8_t>(component[1]));
        host.direct_ao_angular.push_back(static_cast<std::uint8_t>(component[2]));
        host.direct_ao_coefficients.push_back(
            molecule::cartesian_component_normalization(component));
      }
      host.shell_direct_ao_offsets.push_back(
          static_cast<std::int64_t>(host.direct_ao_shells.size()));
      for (const molecule::AoExpansion& expansion :
           molecule::ao_expansions(shell.angular_momentum, system.basis_representation)) {
        if (expansion.empty() || expansion.size() > kMaximumAoExpansionTerms) {
          return false;
        }
        const std::size_t target_ao = host.ao_shells.size() - system_ao_begin;
        host.ao_shells.push_back(shell_index);
        host.ao_term_counts.push_back(static_cast<std::uint8_t>(expansion.size()));
        for (std::size_t term_index = 0; term_index < kMaximumAoExpansionTerms; ++term_index) {
          if (term_index < expansion.size()) {
            const molecule::CartesianExpansionTerm& term = expansion[term_index];
            host.ao_term_angular.push_back(static_cast<std::uint8_t>(term.component[0]));
            host.ao_term_angular.push_back(static_cast<std::uint8_t>(term.component[1]));
            host.ao_term_angular.push_back(static_cast<std::uint8_t>(term.component[2]));
            host.ao_term_coefficients.push_back(
                term.coefficient * molecule::cartesian_component_normalization(term.component));
            if (!host.ao_to_direct_transform.empty()) {
              const auto component = std::find(cartesian_components.begin(),
                                               cartesian_components.end(), term.component);
              if (component == cartesian_components.end()) return false;
              const std::size_t direct_ao =
                  direct_shell_begin +
                  static_cast<std::size_t>(component - cartesian_components.begin());
              const std::size_t transform_offset = system_index * host.nbf * host.direct_nbf;
              host.ao_to_direct_transform[transform_offset + target_ao + direct_ao * host.nbf] =
                  term.coefficient;
            }
          } else {
            host.ao_term_angular.insert(host.ao_term_angular.end(), {0, 0, 0});
            host.ao_term_coefficients.push_back(0.0);
          }
        }
      }
      host.shell_ao_offsets.push_back(static_cast<std::int64_t>(host.ao_shells.size()));
    }
    if (host.ao_shells.size() - system_ao_begin != host.nbf ||
        host.direct_ao_shells.size() - system_direct_ao_begin != host.direct_nbf) {
      return false;
    }
    host.system_shell_offsets.push_back(static_cast<std::int64_t>(host.shell_atoms.size()));
    for (std::size_t first = system_shell_begin; first < host.shell_atoms.size(); ++first) {
      for (std::size_t second = system_shell_begin; second <= first; ++second) {
        host.shell_pair_systems.push_back(static_cast<std::int32_t>(system_index));
        host.shell_pair_first.push_back(static_cast<std::int32_t>(first));
        host.shell_pair_second.push_back(static_cast<std::int32_t>(second));
        const std::int64_t first_primitive_count =
            host.shell_primitive_offsets[first + 1] - host.shell_primitive_offsets[first];
        const std::int64_t second_primitive_count =
            host.shell_primitive_offsets[second + 1] - host.shell_primitive_offsets[second];
        if (first_primitive_count <= 0 || second_primitive_count <= 0 ||
            first_primitive_count >
                std::numeric_limits<std::int64_t>::max() / second_primitive_count) {
          return false;
        }
        const std::int64_t pair_primitive_count = first_primitive_count * second_primitive_count;
        if (host.shell_pair_primitive_offsets.back() >
            std::numeric_limits<std::int64_t>::max() - pair_primitive_count) {
          return false;
        }
        host.shell_pair_primitive_offsets.push_back(host.shell_pair_primitive_offsets.back() +
                                                    pair_primitive_count);
      }
    }
    const std::size_t system_shell_pair_end = host.shell_pair_first.size();
    const std::size_t system_shell_pair_begin =
        static_cast<std::size_t>(host.system_shell_pair_offsets.back());
    const std::size_t system_shell_pair_count = system_shell_pair_end - system_shell_pair_begin;
    std::vector<std::uint32_t> psss_bra_pairs;
    const std::size_t resident_ket_begin = host.psss_resident_ket_pairs.size();
    for (std::size_t pair = system_shell_pair_begin; pair < system_shell_pair_end; ++pair) {
      if (pair > std::numeric_limits<std::uint32_t>::max()) return false;
      const std::int32_t first_shell = host.shell_pair_first[pair];
      const std::int32_t second_shell = host.shell_pair_second[pair];
      const unsigned first_angular = host.shell_angular[first_shell];
      const unsigned second_angular = host.shell_angular[second_shell];
      if (first_angular + second_angular == 1U) {
        psss_bra_pairs.push_back(static_cast<std::uint32_t>(pair));
      } else if (first_angular == 0U && second_angular == 0U) {
        host.psss_resident_ket_pairs.push_back(static_cast<std::uint32_t>(pair));
      }
    }
    const std::size_t resident_ket_count = host.psss_resident_ket_pairs.size() - resident_ket_begin;
    if (resident_ket_begin > std::numeric_limits<std::uint32_t>::max() ||
        resident_ket_count > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    for (const std::uint32_t bra_pair : psss_bra_pairs) {
      if (matrix_direct) break;
      for (std::size_t ket = 0; ket < resident_ket_count; ket += kResidentPsssThreads) {
        const std::size_t chunk_count =
            std::min<std::size_t>(kResidentPsssThreads, resident_ket_count - ket);
        const std::size_t chunk_begin = resident_ket_begin + ket;
        if (chunk_begin > std::numeric_limits<std::uint32_t>::max()) {
          return false;
        }
        host.psss_resident_tasks.push_back({bra_pair, static_cast<std::uint32_t>(chunk_begin),
                                            static_cast<std::uint32_t>(chunk_count)});
      }
    }
    host.system_shell_pair_offsets.push_back(static_cast<std::int64_t>(system_shell_pair_end));
    const std::size_t system_shell_pair_block_count = detail::bounded_direct_queue_refill_count(
        system_shell_pair_count, detail::kBoundedDirectShellPairBlockSize);
    const std::int64_t previous_block_offset = host.system_shell_pair_block_offsets.back();
    if (system_shell_pair_block_count >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() -
                                 previous_block_offset)) {
      return false;
    }
    host.system_shell_pair_block_offsets.push_back(
        previous_block_offset + static_cast<std::int64_t>(system_shell_pair_block_count));
    std::size_t system_shell_pair_block_plus_one = 0;
    std::size_t system_shell_pair_block_quartet_count = 0;
    if (!checked_add(system_shell_pair_block_count, 1, system_shell_pair_block_plus_one) ||
        !checked_multiply(system_shell_pair_block_count, system_shell_pair_block_plus_one,
                          system_shell_pair_block_quartet_count)) {
      return false;
    }
    system_shell_pair_block_quartet_count /= 2;
    const std::int64_t previous_block_quartet_offset =
        host.system_shell_pair_block_quartet_offsets.back();
    if (system_shell_pair_block_quartet_count >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() -
                                 previous_block_quartet_offset)) {
      return false;
    }
    host.system_shell_pair_block_quartet_offsets.push_back(
        previous_block_quartet_offset +
        static_cast<std::int64_t>(system_shell_pair_block_quartet_count));
    std::size_t system_shell_pair_plus_one = 0;
    std::size_t system_shell_quartet_count = 0;
    if (!checked_add(system_shell_pair_count, 1, system_shell_pair_plus_one) ||
        !checked_multiply(system_shell_pair_count, system_shell_pair_plus_one,
                          system_shell_quartet_count)) {
      return false;
    }
    system_shell_quartet_count /= 2;
    const std::int64_t previous_quartet_offset = host.system_shell_quartet_offsets.back();
    if (system_shell_quartet_count >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() -
                                 previous_quartet_offset)) {
      return false;
    }
    host.system_shell_quartet_offsets.push_back(
        previous_quartet_offset + static_cast<std::int64_t>(system_shell_quartet_count));
    if (unrestricted) {
      const int alpha = (system.electron_count + spin_excess) / 2;
      host.occupied.push_back(alpha);
      host.occupied.push_back(system.electron_count - alpha);
    } else {
      host.occupied.push_back(system.electron_count / 2);
    }
    const std::vector<double>* warm = initial_densities[system_index];
    const std::size_t warm_size = host.spin_count * matrix_size;
    // A supplied warm state is an explicit input, not an optional hint.  The
    // CPU path rejects malformed matrices; silently converting one to a cold
    // guess would make CUDA and CPU disagree and could hide a caller bug.
    if (warm != nullptr && (warm->size() != warm_size ||
                            !std::all_of(warm->begin(), warm->end(),
                                         [](double value) { return std::isfinite(value); }))) {
      return false;
    }
    const bool valid_warm = warm != nullptr;
    host.warm_mask.push_back(valid_warm ? 1 : 0);
    if (valid_warm) {
      std::copy(warm->begin(), warm->end(), host.warm_density.begin() + system_index * warm_size);
    }
  }
  return true;
}

bool same_topology(const HostBatch& first, const HostBatch& second) {
  if (first.ecp_systems.size() != second.ecp_systems.size()) return false;
  for (std::size_t i = 0; i < first.ecp_systems.size(); ++i)
    if (first.ecp_systems[i].ecp_terms != second.ecp_systems[i].ecp_terms) return false;
  return first.nbf == second.nbf && first.direct_nbf == second.direct_nbf &&
         first.spin_count == second.spin_count && first.atom_offsets == second.atom_offsets &&
         first.atom_systems == second.atom_systems &&
         first.atomic_numbers == second.atomic_numbers &&
         first.system_shell_offsets == second.system_shell_offsets &&
         first.shell_atoms == second.shell_atoms && first.shell_angular == second.shell_angular &&
         first.shell_ao_offsets == second.shell_ao_offsets &&
         first.shell_direct_ao_offsets == second.shell_direct_ao_offsets &&
         first.shell_primitive_offsets == second.shell_primitive_offsets &&
         first.system_shell_pair_offsets == second.system_shell_pair_offsets &&
         first.system_shell_quartet_offsets == second.system_shell_quartet_offsets &&
         first.system_shell_pair_block_offsets == second.system_shell_pair_block_offsets &&
         first.system_shell_pair_block_quartet_offsets ==
             second.system_shell_pair_block_quartet_offsets &&
         first.shell_pair_systems == second.shell_pair_systems &&
         first.shell_pair_first == second.shell_pair_first &&
         first.shell_pair_second == second.shell_pair_second &&
         first.shell_pair_primitive_offsets == second.shell_pair_primitive_offsets &&
         first.ao_shells == second.ao_shells && first.ao_term_counts == second.ao_term_counts &&
         first.ao_term_angular == second.ao_term_angular &&
         first.ao_term_coefficients == second.ao_term_coefficients &&
         first.direct_ao_shells == second.direct_ao_shells &&
         first.direct_ao_angular == second.direct_ao_angular &&
         first.direct_ao_coefficients == second.direct_ao_coefficients &&
         first.ao_to_direct_transform == second.ao_to_direct_transform &&
         first.primitive_exponents == second.primitive_exponents &&
         first.primitive_coefficients == second.primitive_coefficients &&
         first.occupied == second.occupied;
}

}  // namespace vibeqc::scf::cuda_execution

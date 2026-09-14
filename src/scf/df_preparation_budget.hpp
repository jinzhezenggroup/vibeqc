#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

namespace vibeqc::scf {

/** Host live-set bound for one source-backed CUDA DF preparation item.
 * The caller selects the actual derivative exporter. Generated one-electron
 * response retains geometry instead of coordinate-major AO derivative arrays.
 * Zero-budget compatibility preparation is outside this bounded source path.
 */
struct DfPreparationShape {
  std::size_t cartesian_nbf{}, public_nbf{}, atoms{};
  std::size_t orbital_shells{}, orbital_primitives{};
  std::size_t auxiliary_shells{}, auxiliary_primitives{};
  bool include_derivatives{}, generated_one_electron{};
};

struct DfPreparationStorage {
  std::size_t retained_bytes{};
  std::size_t peak_bytes{};
  std::size_t metadata_bytes{};
};

/** Saturate infeasible shapes instead of wrapping them into a small allowance. */
inline std::size_t df_preparation_bytes(long double bytes) noexcept {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  return bytes >= static_cast<long double>(maximum) ? maximum : static_cast<std::size_t>(bytes);
}

inline DfPreparationStorage df_preparation_storage(DfPreparationShape shape) noexcept {
  const long double c = shape.cartesian_nbf, n = shape.public_nbf, atoms = shape.atoms;
  const long double coordinates = 3 * atoms;
  const bool matrices = shape.include_derivatives && !shape.generated_one_electron;
  const long double copies = 1 + (matrices ? coordinates : 0);
  const long double nuclear = shape.include_derivatives ? coordinates : 0;
  const long double cartesian = 2 * c * c * copies + nuclear;
  const long double retained = 2 * n * n * copies + nuclear;
  const long double shells = shape.orbital_shells;
  // LP64 capacity allowance for basis copies, public expansions, packed pair
  // metadata, and the geometry owners bound later to generated force response.
  const long double metadata =
      512 * (1 + atoms + shells + shape.orbital_primitives + c + n + shells * (shells + 1) / 2 +
             shape.auxiliary_shells + shape.auxiliary_primitives);
  // The bridge owns packed S/H/nuclear values, a warm-density matrix and pair
  // indices. Transformation holds Cartesian and public outputs simultaneously
  // (even Cartesian return-by-value copies), plus two derivative row matrices.
  const long double staging = 4 * c * c + 1 + (matrices ? 2 * n * n : 0);
  return {df_preparation_bytes(sizeof(double) * retained + metadata),
          df_preparation_bytes(sizeof(double) * (cartesian + retained + staging) + 2 * metadata),
          df_preparation_bytes(metadata)};
}

/** Energy owns the whole positive value allowance. Forces reserve the other
 * half for sequential DF/one-electron response; zero keeps legacy defaults.
 * Clamp positive subdivisions so a tiny request cannot become unbounded zero.
 */
inline std::size_t df_value_budget(std::size_t requested, bool forces) noexcept {
  return requested && forces ? std::max<std::size_t>(1, requested / 2) : requested;
}

inline std::size_t df_force_budget(std::size_t requested) noexcept {
  return requested ? std::max<std::size_t>(1, requested / 2) : 128U * 1024U * 1024U;
}

}  // namespace vibeqc::scf

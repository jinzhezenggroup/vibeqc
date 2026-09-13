#pragma once

#include <cstddef>

#include "scf/density_factor.hpp"

namespace vibeqc::dft {

/** Explicit algorithm candidates; no automatic performance policy is implied. */
enum class XcDensityRoute { DensityMatrix, OccupiedOrbitals };
enum class XcDensityRole { State, Response };
enum class XcDensityFallback { None, Response, MissingFactor, Basis, Identity, Spin, Density };

/** Synchronous borrowed source for the CURRENT restricted total density.
 * The owner must keep D and the immutable factor alive for the entire call.
 * A matching shape is insufficient: all expected provenance fields and the
 * exact density witness must match. Unsupported/external states omit factor.
 * Native factors currently use canonical real occupations; the broader
 * fractional/spin fixed-input contract is supplied by the prepared D/C API.
 */
struct XcDensitySource {
  XcDensityRoute route{XcDensityRoute::DensityMatrix};
  const scf::OccupiedDensityFactor* factor{};
  scf::DensityFactorIdentity identity{};
  /** A response never opts into C, even if a particular perturbation is PSD. */
  XcDensityRole role{XcDensityRole::State};
};

/** Per-call execution and capacity record. Borrowed D/factor storage is kept
 * separate from the owned AO tile and full potential matrix, so a composing
 * SCF observer charges shared storage once. Scalar stack and library-private
 * scratch are outside these vector-capacity observations.
 */
struct XcDensityDiagnostic {
  XcDensityRoute requested{XcDensityRoute::DensityMatrix};
  XcDensityRoute executed{XcDensityRoute::DensityMatrix};
  XcDensityFallback fallback{XcDensityFallback::None};
  std::size_t npoint{}, active_ao{}, nocc{}, max_tile_points{};
  unsigned ingredient_mask{};
  std::size_t borrowed_density_bytes{}, borrowed_factor_bytes{}, owned_numeric_bytes{};
};

/** Aggregate native RKS diagnostics, including both final physical builds.
 * Packing counts describe actual copied occupied coefficient elements.
 * Factor capacity includes the separate exact D witness; SCF still retains D.
 */
struct RksDensityDiagnostic {
  std::size_t density_calls{}, orbital_calls{}, fallback_calls{};
  std::size_t packed_coefficient_elements{}, factor_peak_bytes{}, xc_peak_bytes{};
  scf::DensityFactorIdentity final_identity{};
  double physical_residual{};
};

}  // namespace vibeqc::dft

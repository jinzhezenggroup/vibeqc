#pragma once

#include <cstddef>
#include <vector>

#include "core/types.hpp"

namespace vibeqc::dft {

/** Owned, normalized through-f basis for spatial AO jets on explicit points.
 *
 * Packed FP64 sections are centers[atom,3], primitives[primitive,2], and
 * AOs[ao,16]. Each AO record is (atom, primitive_begin, primitive_count,
 * expansion_count, (lx,ly,lz,weight)[3]). Primitive contractions are stored
 * once per shell. Integer fields are exact and bounded at INT32_MAX. Weights
 * include the existing normalized Cartesian-to-real-spherical expansion.
 */
class AoBasis {
 public:
  explicit AoBasis(const core::System& normalized);
  std::size_t natom{}, nprimitive{}, nao{};
  std::vector<double> packed;

  /** Ordinary spatial derivatives, with no factorial scaling. Jet order is
   * total degree then CCA powers: 1,x,y,z,xx,xy,xz,yy,yz,zz,xxx,...,zzz.
   * Output is [jet, point, selected_ao]. Public basis l stays <=3 even when
   * differentiation temporarily raises polynomial powers to six.
   * Optional sorted unique ao_ids select records directly instead of the
   * contiguous slice. No omitted AO is evaluated or temporarily stored.
   */
  void evaluate(const double* points, std::size_t npoint, unsigned order, std::size_t ao_begin,
                std::size_t ao_count, double* output, std::size_t elements,
                const std::size_t* ao_ids = nullptr) const;
};
}  // namespace vibeqc::dft

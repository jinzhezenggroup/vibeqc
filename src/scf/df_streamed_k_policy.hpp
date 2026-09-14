#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

namespace vibeqc::scf {

/** One shared dense/occupied generated-K traversal of existing tile storage.
 * Capacity is in doubles in EACH of the four disjoint plan buffers. A panel
 * holds rows * nbf pairs and output_auxiliaries Q directions; the raw scratch
 * independently holds raw_auxiliaries P directions. No new storage is implied.
 */
struct DfStreamedKPanel {
  std::size_t rows{};
  std::size_t output_auxiliaries{};
  std::size_t raw_auxiliaries{};
  std::size_t row_tiles{};
  std::size_t output_tiles{};
};

/** Minimize raw source work, including the repeated column-panel traversal.
 * For R row blocks and T output blocks, each AO pair is generated R*T times:
 * the diagonal reuses its row panel, and all other columns are regenerated.
 * Each raw panel serves ALL active output Q through GEMM. Counting only
 * output panels would favor full AO/Q=1 even when it repeats more integrals.
 *
 * Visit the smallest row width for each distinct ceil(nbf/rows). Larger widths
 * with the same R cannot improve Q capacity or the primary work bound. This
 * quotient traversal takes O(sqrt(nbf)) candidates, including for shape-only
 * planning. Equal-work candidates prefer fewer transformed panels, then wider
 * Q (more raw reuse). Products used for comparison are widened before multiply.
 * A zero result means even one AO row and auxiliary value cannot fit.
 */
inline DfStreamedKPanel df_streamed_k_panel(std::size_t nbf, std::size_t naux,
                                            std::size_t capacity) {
  DfStreamedKPanel best;
  if (!nbf || !naux || capacity < nbf) return best;
  const auto row_auxiliary_capacity = capacity / nbf;
  const auto max_rows = std::min(nbf, row_auxiliary_capacity);
  long double best_work = std::numeric_limits<long double>::infinity();
  long double best_panels = best_work;
  for (std::size_t rows = 1; rows <= max_rows;) {
    const auto r = 1 + (nbf - 1) / rows;
    const auto q = std::min(naux, row_auxiliary_capacity / rows);
    const auto t = 1 + (naux - 1) / q;
    const long double work = static_cast<long double>(r) * t;
    const long double panels = work * r;
    if (work < best_work ||
        (work == best_work &&
         (panels < best_panels || (panels == best_panels && q > best.output_auxiliaries)))) {
      best = {rows, q, q, r, t};
      best_work = work;
      best_panels = panels;
    }
    if (r == 1) break;
    rows = (nbf - 1) / (r - 1) + 1;
  }
  return best;
}

}  // namespace vibeqc::scf

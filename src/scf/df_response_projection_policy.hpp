#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace vibeqc::scf {

/** Work census, not a latency estimate. All sizes count double elements. */
struct DfFittedPanelWork {
  std::size_t panels{}, reader_calls{}, projected_columns{}, staging_elements{};
};

inline std::size_t df_response_checked_product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("DF response work size overflows size_t");
  return a * b;
}

inline std::size_t df_response_checked_sum(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::overflow_error("DF response workspace size overflows size_t");
  return a + b;
}

/** The general-density fallback recomputes every fitted Q panel for each P
 * panel. Keep this cost visible: a smaller tile increases actual work, not
 * only launch overhead. The diagonal visit replaces, rather than adds to,
 * its occurrence in the inner traversal.
 */
inline DfFittedPanelWork df_fitted_panel_work(std::size_t naux, std::size_t tile,
                                            std::size_t stored_pairs) {
  if (!naux || !tile || !stored_pairs)
    throw std::invalid_argument("DF response census requires positive dimensions");
  const auto panels = naux / tile + (naux % tile != 0);
  const auto columns = df_response_checked_product(panels, naux);
  return {panels, df_response_checked_product(panels, panels), columns,
          df_response_checked_product(columns, stored_pairs)};
}

/** Reuse only the projection interval that is dead while S_Q=C^T B_Q C (or
 * C^T A_Q C) is accumulated into a disjoint retained interval. No value-owner
 * or future warm-state allocation is released. A pair-major producer needs
 * a second AO panel for the existing transpose; an auxiliary-major producer
 * does not. Zero means that the scalar path must use its existing AO scratch.
 */
struct DfOccupiedProjectionBatch {
  std::size_t columns{}, input_elements{}, staging_elements{}, product_elements{};
  std::size_t total_elements{};
};

inline DfOccupiedProjectionBatch plan_df_occupied_projection_batch(
    std::size_t nbf, std::size_t naux, std::size_t maximum_rank,
    std::size_t reusable_elements, std::size_t column_cap, bool pair_major_source) {
  if (!nbf || !naux || maximum_rank > nbf || !column_cap)
    throw std::invalid_argument("invalid occupied DF projection shape or cap");
  const auto matrix = df_response_checked_product(nbf, nbf);
  const auto product = df_response_checked_product(nbf, maximum_rank);
  const auto staging = pair_major_source ? matrix : 0;
  const auto stride = df_response_checked_sum(df_response_checked_sum(matrix, staging), product);
  const auto columns = std::min({naux, column_cap, reusable_elements / stride});
  return {columns, df_response_checked_product(columns, matrix),
          df_response_checked_product(columns, staging),
          df_response_checked_product(columns, product),
          df_response_checked_product(columns, stride)};
}

}  // namespace vibeqc::scf

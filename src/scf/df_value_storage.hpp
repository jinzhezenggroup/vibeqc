#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vibeqc::scf {

/** Persistent value representation, independent of packed derivative weights.
 * Dense permits arbitrary public tensors. SymmetricLower is admitted only
 * for physical integral sources; it stores unit-weight (mu>=nu) pairs with Q
 * contiguous. Geometry/metric/source ownership remains part of the plan token.
 */
enum class DfPairStorage { Dense, SymmetricLower };

/** Explicit comparison selector for physical SCF preparation. Automatic
 * selection stays dense until complete endpoint/capacity qualification; the
 * native explicit constructor never consults this ambient setting.
 */
inline DfPairStorage requested_df_pair_storage() {
  const char* value = std::getenv("VIBEQC_DF_VALUE_STORAGE");
  if (!value || std::strcmp(value, "auto") == 0 || std::strcmp(value, "dense") == 0)
    return DfPairStorage::Dense;
  if (std::strcmp(value, "packed") == 0) return DfPairStorage::SymmetricLower;
  throw std::invalid_argument("VIBEQC_DF_VALUE_STORAGE must be auto, dense or packed");
}

/** A lower-pair address cannot be mistaken for mu*nbf+nu. The owning shape
 * validates dimensions before either host or device constructs this offset.
 */
struct DfSymmetricAoPair {
  std::size_t index{};
};

/** Explicit experiment selection; ordinary/public tensor callers stay dense.
 * rank_capacity bounds the optional complete occupied U allocation, not the
 * mathematical rank of accepted input. Higher ranks use bounded projection.
 */
struct DfValueStorageOptions {
  DfPairStorage pairs{DfPairStorage::Dense};
  std::size_t rank_capacity{};
};

/** Checked capacities shared by the common planner and native allocator.
 * The first mutable buffer holds either complete U or an auxiliary panel;
 * the other two hold dense compatibility panels. Scratch is reused serially
 * across batch items. Packed raw A is an independent immutable allocation.
 */
struct DfPackedValueCapacity {
  std::size_t pairs{}, tensor_per_system{}, all_tensor_elements{};
  std::size_t projection_elements{}, panel_elements{};
  std::size_t factor_bytes{}, scratch_bytes{};
};

inline DfPackedValueCapacity df_packed_value_capacity(std::size_t batch, std::size_t n,
                                                      std::size_t a, std::size_t rank,
                                                      std::size_t auxiliary_tile) {
  if (!batch || !n || !a || rank > n || !auxiliary_tile || auxiliary_tile > a)
    throw std::invalid_argument("invalid packed DF storage dimensions");
  const auto multiply = [](std::size_t x, std::size_t y) {
    if (y && x > std::numeric_limits<std::size_t>::max() / y)
      throw std::overflow_error("packed DF storage overflows size_t");
    return x * y;
  };
  const auto add = [](std::size_t x, std::size_t y) {
    if (y > std::numeric_limits<std::size_t>::max() - x)
      throw std::overflow_error("packed DF storage overflows size_t");
    return x + y;
  };
  // Divide before multiplying so a representable triangular count is not
  // rejected merely because its unhalved intermediate would overflow.
  const auto successor = add(n, 1);
  DfPackedValueCapacity c;
  c.pairs = n % 2 ? multiply(n, successor / 2) : multiply(n / 2, successor);
  c.tensor_per_system = multiply(c.pairs, a);
  c.all_tensor_elements = multiply(batch, c.tensor_per_system);
  c.panel_elements = multiply(multiply(n, n), auxiliary_tile);
  c.projection_elements = std::max(c.panel_elements, multiply(multiply(n, rank), a));
  c.factor_bytes = multiply(c.all_tensor_elements, sizeof(double));
  c.scratch_bytes =
      multiply(add(c.projection_elements, multiply(2, c.panel_elements)), sizeof(double));
  // Validate both immutable owners together before either is allocated.
  (void)add(multiply(2, c.factor_bytes), c.scratch_bytes);
  return c;
}

}  // namespace vibeqc::scf

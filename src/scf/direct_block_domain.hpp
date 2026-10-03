#pragma once

#include <cstdint>

namespace generativeqc::scf::detail {

/** Independently claim bounded pages within each geometry-live block product.
 * Schwarz sorting can concentrate expensive shell tasks into very few blocks;
 * one worker must not serially own all 1024 candidates in such a block. Empty
 * diagonal/tail pages are harmless and preserve the linear prefix inventory. */
inline constexpr std::uint64_t kBoundedDirectIndexedCandidatePages = 16;

/** Borrow an exclusive prefix over globally indexed shell-pair-block rows.
 * A null prefix retains the complete triangular scheduler. Every nonempty row
 * contains a contiguous ket prefix within its own system. No quartet list is
 * retained; row_count+1 words describe all geometry-admitted block products. */
struct BoundedDirectBlockDomain {
  const std::uint64_t* prefix{};
  std::uint64_t row_count{}, quartet_count{};
};

}  // namespace generativeqc::scf::detail

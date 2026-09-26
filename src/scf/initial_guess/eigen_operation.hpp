#pragma once

#include <functional>

#include "scf/reference/linalg.hpp"

namespace vibeqc::scf::initial_guess {
/** Synchronous, borrowed eigen operation with the ordinary DF adapter's layout.
 * Null S/X means a symmetric solve; otherwise both matrices are supplied and
 * the result satisfies F C = S C epsilon, with ascending values and row-major
 * coefficient columns. The caller validates the backend's returned frame.
 * Initial-guess/cache helpers never retain this callback or its owner. */
using EigenOperation = std::function<reference::EigenResult(
    const reference::Matrix&, const reference::Matrix*, const reference::Matrix*, std::size_t)>;
}  // namespace vibeqc::scf::initial_guess

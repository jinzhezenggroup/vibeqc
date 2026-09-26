#pragma once

#include <cstddef>

#include "molecule/basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Existing angular/workspace limits shared by retained integral evaluators. */
constexpr int kMaximumAngularMomentum = 3;
constexpr std::size_t kMaximumAoExpansionTerms = molecule::kMaximumAoExpansionTerms;
constexpr int kHermiteIDimension = kMaximumAngularMomentum + 1;
constexpr int kHermiteJDimension = kMaximumAngularMomentum + 3;
constexpr int kHermiteTDimension = 2 * kMaximumAngularMomentum + 4;
constexpr int kMaximumCoulombOrder = 4 * kMaximumAngularMomentum;

}  // namespace vibeqc::scf::cuda_execution

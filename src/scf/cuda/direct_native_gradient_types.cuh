#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Retained direct integral arithmetic for gradient types.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** Density-weighted psss derivatives for the first three canonical centers. */
struct PsssWeightedGradient {
  double center[3][3];
};

/** Density-weighted psps derivatives for the first three canonical centers. */
struct PspsWeightedGradient {
  double center[3][3];
};

/** Cartesian derivatives of one contracted quartet, indexed by input slot. */
struct CartesianQuartetGradient {
  double center[4][3];
};

/** Density-weighted ssss derivatives for the first three input centers. */
struct SsssWeightedGradient {
  double center[3][3];
};

}  // namespace vibeqc::scf::cuda_execution

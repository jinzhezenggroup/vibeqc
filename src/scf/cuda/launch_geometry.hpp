#pragma once

namespace vibeqc::scf::cuda_execution {

// Capture-safe scalar kernels are small or register-heavy and use one warp per
// block. Direct quartets keep their separately documented virtual tiling.
constexpr unsigned kCaptureSafeKernelThreads = 32;

}  // namespace vibeqc::scf::cuda_execution

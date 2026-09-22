#pragma once

#include <cuda_runtime_api.h>

#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf {
/** Borrow the DF owner's ordinary stream for device density/XC consumers.
 * The plan owns its lifetime and must outlive all enqueued work. This preserves
 * ordering without staging densities or matrices through the host. */
cudaStream_t cuda_density_fitting_stream(const CudaDensityFittingJkPlan* plan);
/** Device ordinal in Slurm/process visibility; null returns -1. */
int cuda_density_fitting_device(const CudaDensityFittingJkPlan* plan) noexcept;
}  // namespace vibeqc::scf

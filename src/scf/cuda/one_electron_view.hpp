#pragma once

#include "scf/cuda/one_electron_values.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

/** Borrow normalized one-electron metadata; the caller retains all storage. */
OneElectronDeviceView one_electron_view(const DeviceBatch& batch);

}  // namespace vibeqc::scf::cuda_execution

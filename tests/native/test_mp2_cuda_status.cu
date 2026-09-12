#include <climits>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "posthf/block_capacity_generated.hpp"
#include "posthf/cuda_transform.hpp"
#include "tensor/cuda_runtime.cuh"

int main() {
  if (!std::getenv("VIBEQC_MP2_CUDA_TEST")) return 77;
  try {
    bool cuda_oom = false, blas_oom = false;
    try {
      vibeqc_tensor::cuda_check(cudaErrorMemoryAllocation);
    } catch (const vibeqc_tensor::DeviceAllocationError&) {
      cuda_oom = true;
    }
    try {
      vibeqc_tensor::blas_check(CUBLAS_STATUS_ALLOC_FAILED);
    } catch (const vibeqc_tensor::DeviceAllocationError&) {
      blas_oom = true;
    }
    if (!cuda_oom || !blas_oom) throw std::runtime_error("allocation status type lost");
    const std::array<std::size_t, 4> shape{2, 2, 2, 2}, tile{1, 1, 1, 1};
    const auto plan = vibeqc::posthf::numeric_block_plan(INT_MAX, 0, 0, shape, tile, true);
    std::size_t free_bytes = 0, total_bytes = 0;
    vibeqc_tensor::cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (plan.allocation_bytes <= total_bytes)
      throw std::runtime_error("OOM probe requires a capacity larger than this device");
    // Failure happens at allocation, before the coefficient pointer is read.
    // No competing application memory is touched and no stress loop is used.
    void* handle = nullptr;
    double coefficients[8]{};
    char error[2048]{};
    const auto status = posthf_cuda_create_v1(0, INT_MAX, shape.data(), tile.data(), coefficients,
                                              plan.allocation_bytes, &handle, error, sizeof(error));
    if (handle) posthf_cuda_destroy_v1(handle);
    if (status != 2 || handle)
      throw std::runtime_error("CG10 native allocation failure category/rollback lost");
    std::cout << "CUDA and cuBLAS OOM types; native CG10 OOM category and rollback passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

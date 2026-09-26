#pragma once

#include <vector>

#include "runtime/resource_cuda.cuh"

namespace vibeqc::scf::cuda_df {

/** Temporary metric-factorization storage; errors release every allocation.
 * Successful setup transfers each plan's retained factors before this expires.
 */
struct SetupBuffers {
  double* metrics{};
  double* eigenvalues{};
  double* scales{};
  double* scaled_eigenvectors{};
  double* inverse_square_roots{};
  double* raw_three_center{};
  void* solver_workspace{};
  std::vector<unsigned char> solver_host_workspace;
  int* solver_info{};

  ~SetupBuffers() {
    (void)runtime::resource_cuda_free(metrics);
    (void)runtime::resource_cuda_free(eigenvalues);
    (void)runtime::resource_cuda_free(scales);
    (void)runtime::resource_cuda_free(scaled_eigenvectors);
    (void)runtime::resource_cuda_free(inverse_square_roots);
    (void)runtime::resource_cuda_free(raw_three_center);
    (void)runtime::resource_cuda_free(solver_workspace);
    (void)runtime::resource_cuda_free(solver_info);
  }
};

}  // namespace vibeqc::scf::cuda_df

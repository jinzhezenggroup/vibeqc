#pragma once

#include "scf/cuda/df_scf_state.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf::cuda_df {

/** cuBLAS/cuSOLVER integration for retained DF SCF state.
 * Borrow plan streams and own solver workspace setup independently of J/K selection.
 */
vibeqc_status scf_gemm(CudaDensityFittingJkPlan& plan, bool transpose_left, std::size_t batch_size,
                       std::size_t nbf, const double* left, const double* right, double* output,
                       std::string& detail);

vibeqc_status setup_device_solver(CudaDensityFittingJkPlan& plan, std::size_t nbf,
                                  std::size_t batch_size, double* eigensystem, double* eigenvalues,
                                  DeviceSolver& solver, std::string& detail);

/** Submit one eigensystem per system/spin on the owning plan stream.
 * Diagnostic tracing distinguishes graph construction from ordinary execution. */
vibeqc_status solve_device_batch(CudaDensityFittingJkPlan& plan, DeviceSolver& solver,
                                 std::size_t nbf, std::size_t batch_size, double* eigensystem,
                                 double* eigenvalues, int* info, std::string& detail);

/** After ending a failed capture, clear only expected capture/mode errors.
 * Ordinary launches must start on a noncapturing stream with no stale CUDA
 * error. Unrelated runtime failures remain failures; this is not a CPU retry. */
vibeqc_status recover_scf_capture(cudaStream_t stream, cudaError_t capture_error,
                                  vibeqc_status iteration_status, bool& capture_rejected,
                                  std::string& detail);

}  // namespace vibeqc::scf::cuda_df

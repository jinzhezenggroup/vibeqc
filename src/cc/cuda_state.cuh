// CC iteration state operations. Physical equations and Jacobi updates are
// generated TensorIR. All pointers below belong to a checked #146 reservation;
// no function here allocates, copies a tensor to the host, or owns a stream.
#pragma once

#include "cuda_runtime.cuh"

namespace vibeqc::cc {

// Dense histories use the same Euclidean metric as #148's packed coordinates
// with orbit weights. Keeping both ijab and jiba does not change the metric.
inline void diis_gram(vibeqc_tensor::Context& context, const double* errors, int elements,
                      int history, double* gram) {
  if (elements < 1 || history < 2 || history > 20 || !errors || !gram || !context.handle)
    throw std::invalid_argument("invalid CC DIIS storage/dimensions/handle");
  context.check_device();
  const double one = 1.0, zero = 0.0;
  vibeqc_tensor::blas_check(cublasDgemm(context.handle, CUBLAS_OP_T, CUBLAS_OP_N, history, history,
                                        elements, &one, errors, elements, errors, elements, &zero,
                                        gram, history));
}

// Only the bounded (<=21)^2 augmented DIIS system uses a serial GPU solve.
// Status 1 asks the owner to discard the oldest history and retry; it never
// silently promotes an unstable extrapolate to the next physical state.
__global__ void diis_coefficients(const double* gram, int history, double* system,
                                  double* coefficients, int* status) {
  if (blockIdx.x || threadIdx.x) return;
  *status = 1;
  if (history < 2 || history > 20) return;
  const int width = history + 1;
  double scale = 0.0;
  for (int i = 0; i < history * history; ++i) {
    if (!isfinite(gram[i])) return;
    scale = fmax(scale, fabs(gram[i]));
  }
  if (!(scale > 0.0)) return;
  for (int row = 0; row < width; ++row) {
    coefficients[row] = row == history ? -1.0 : 0.0;
    for (int col = 0; col < width; ++col) {
      system[row * width + col] = row == history && col == history ? 0.0
                                  : row == history || col == history
                                      ? -1.0
                                      : gram[row + history * col] / scale;
    }
  }
  for (int col = 0; col < width; ++col) {
    int pivot = col;
    for (int row = col + 1; row < width; ++row)
      if (fabs(system[row * width + col]) > fabs(system[pivot * width + col])) pivot = row;
    const double divisor = system[pivot * width + col];
    if (!isfinite(divisor) || fabs(divisor) < 1e-14) return;
    if (pivot != col) {
      for (int j = 0; j < width; ++j) {
        const double value = system[col * width + j];
        system[col * width + j] = system[pivot * width + j];
        system[pivot * width + j] = value;
      }
      const double value = coefficients[col];
      coefficients[col] = coefficients[pivot];
      coefficients[pivot] = value;
    }
    for (int j = col; j < width; ++j) system[col * width + j] /= divisor;
    coefficients[col] /= divisor;
    for (int row = 0; row < width; ++row) {
      if (row == col) continue;
      const double factor = system[row * width + col];
      for (int j = col; j < width; ++j) system[row * width + j] -= factor * system[col * width + j];
      coefficients[row] -= factor * coefficients[col];
    }
  }
  double sum = 0.0;
  for (int i = 0; i < history; ++i) {
    if (!isfinite(coefficients[i]) || fabs(coefficients[i]) > 1e6) return;
    sum += coefficients[i];
  }
  if (fabs(sum - 1.0) > 1e-8) return;
  *status = 0;
}

__global__ void diis_combine(const double* vectors, const double* coefficients,
                             vibeqc_tensor::I elements, int history, const int* status,
                             double* result, int* arithmetic_error) {
  if (*status) return;
  for (vibeqc_tensor::I i = vibeqc_tensor::I(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
       i += vibeqc_tensor::I(blockDim.x) * gridDim.x) {
    double value = 0.0;
    for (int j = 0; j < history; ++j)
      value = __dadd_rn(value, __dmul_rn(coefficients[j], vectors[j * elements + i]));
    result[i] = vibeqc_tensor::finite(value, arithmetic_error, 0);
  }
}

// The physical max norm cannot be replaced by an update norm or DIIS error.
// Two deterministic reductions avoid float atomics and keep nonfinite checks.
__global__ void residual_partials(const double* residual, vibeqc_tensor::I count, double* partials,
                                  int* error) {
  __shared__ double shared[256];
  double value = 0.0;
  for (vibeqc_tensor::I i = vibeqc_tensor::I(blockIdx.x) * 256 + threadIdx.x; i < count;
       i += vibeqc_tensor::I(gridDim.x) * 256)
    value = fmax(value, fabs(vibeqc_tensor::finite(residual[i], error, 0)));
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] = fmax(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  if (!threadIdx.x) partials[blockIdx.x] = shared[0];
}

__global__ void residual_finish(const double* partials, int count, double* output) {
  if (threadIdx.x || blockIdx.x) return;
  double value = 0.0;
  for (int i = 0; i < count; ++i) value = fmax(value, partials[i]);
  *output = value;
}

}  // namespace vibeqc::cc

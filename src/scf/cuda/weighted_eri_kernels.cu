#include <cmath>
#include <weighted_eri.cuh>

#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_cartesian.cuh"
#include "scf/cuda/scalar_math.cuh"
#include "scf/cuda/weighted_eri_kernels.hpp"

namespace vibeqc::scf::cuda_execution {

/** Independent unscreened primitive fallback, with no HF density contraction. */
__device__ __noinline__ CudaWeightedEriResult
weighted_eri_reference_primitive(const CudaWeightedEriPrimitive& record, int psss_axis) {
  Angular angular[4];
  for (unsigned slot = 0; slot < 4; ++slot) {
    angular[slot] = {record.angular[slot][0], record.angular[slot][1], record.angular[slot][2]};
  }
  if (psss_axis >= 0) {
    angular[0] = {static_cast<unsigned>(psss_axis == 0), static_cast<unsigned>(psss_axis == 1),
                  static_cast<unsigned>(psss_axis == 2)};
  }
  CudaWeightedEriResult result{};
  for (unsigned center = 0; center < 3; ++center) {
    Vec3<Dual3> positions[4];
    for (unsigned slot = 0; slot < 4; ++slot) {
      const double seed = slot == center ? 1.0 : 0.0;
      positions[slot] = {{record.centers[slot][0], seed, 0.0, 0.0},
                         {record.centers[slot][1], 0.0, seed, 0.0},
                         {record.centers[slot][2], 0.0, 0.0, seed}};
    }
    // The public f/f/f/f value order is twelve; Dual3 carries the extra Boys
    // response order. This conservative fallback is intentionally independent
    // of the generated weighted recurrence and its scheduling choices.
    const auto value = primitive_eri_cartesian<12>(record.exponents[0], positions[0], angular[0],
                                                   record.exponents[1], positions[1], angular[1],
                                                   record.exponents[2], positions[2], angular[2],
                                                   record.exponents[3], positions[3], angular[3]);
    result.value = value.value;
    result.center[center][0] = value.derivative_x;
    result.center[center][1] = value.derivative_y;
    result.center[center][2] = value.derivative_z;
  }
  for (unsigned axis = 0; axis < 3; ++axis) {
    result.center[3][axis] =
        -result.center[0][axis] - result.center[1][axis] - result.center[2][axis];
  }
  return result;
}

/** Add only a tile's scalar/gradient, never a four-index response tensor. */
template <typename Result>
__device__ void add_weighted_eri_result(CudaWeightedEriResult* output, const Result& value,
                                        double weight) {
  atomicAdd(&output->value, weight * value.value);
  for (unsigned center = 0; center < 4; ++center) {
    for (unsigned axis = 0; axis < 3; ++axis) {
      atomicAdd(&output->center[center][axis], weight * value.center[center][axis]);
    }
  }
}

__global__ void weighted_eri_reference_kernel(const CudaWeightedEriPrimitive* records,
                                              std::size_t count, bool generated,
                                              CudaWeightedEriResult* output) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const auto& record = records[index];
  if (record.kind == 1U && generated) return;
  const unsigned components = record.kind == 1U ? 3U : 1U;
  for (unsigned component = 0; component < components; ++component) {
    const double weight = record.weights[component];
    if (weight == 0.0) continue;  // Exact zero only; no density-bound screening.
    const auto value = weighted_eri_reference_primitive(
        record, record.kind == 1U ? static_cast<int>(component) : -1);
    add_weighted_eri_result(output + record.output_tile, value, weight);
  }
}

/** Generated psss external weights use the same callable as native HF force. */
__global__ void weighted_eri_generated_psss_kernel(const CudaWeightedEriPrimitive* records,
                                                   std::size_t count,
                                                   CudaWeightedEriResult* output) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count || records[index].kind != 1U) return;
  const auto& record = records[index];
  const double alpha = record.exponents[0], beta = record.exponents[1];
  const double gamma = record.exponents[2], delta = record.exponents[3];
  const double p = alpha + beta, q = gamma + delta;
  const double mu = alpha * beta / p, nu = gamma * delta / q;
  Vec3<double> positions[4];
  for (unsigned slot = 0; slot < 4; ++slot) {
    positions[slot] = {record.centers[slot][0], record.centers[slot][1], record.centers[slot][2]};
  }
  const auto P = product_center(alpha, positions[0], beta, positions[1]);
  const auto Q = product_center(gamma, positions[2], delta, positions[3]);
  generated_weighted_eri::Geometry geometry{};
  geometry.inverse_two_p = 0.5 / p;
  geometry.rho = p * q / (p + q);
  geometry.prefactor = 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) *
                       exp(-mu * distance_squared(positions[0], positions[1]) -
                           nu * distance_squared(positions[2], positions[3]));
  geometry.product_scales[0] = alpha / p;
  geometry.product_scales[1] = beta / p;
  geometry.product_scales[2] = gamma / q;
  boys_values<2>(geometry.rho * distance_squared(P, Q), geometry.boys);
#pragma unroll
  for (unsigned axis = 0; axis < 3; ++axis) {
    geometry.difference[axis] = vec_axis(P, axis) - vec_axis(Q, axis);
    geometry.shifts[0][axis] = vec_axis(P, axis) - record.centers[0][axis];
    geometry.decay[0][axis] = -2.0 * mu * (record.centers[0][axis] - record.centers[1][axis]);
    geometry.decay[1][axis] = -geometry.decay[0][axis];
    geometry.decay[2][axis] = -2.0 * nu * (record.centers[2][axis] - record.centers[3][axis]);
  }
  const auto value = generated_weighted_eri::psss(geometry, record.weights);
  add_weighted_eri_result(output + record.output_tile, value, 1.0);
}

void launch_weighted_eri_reference_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream,
                                          const CudaWeightedEriPrimitive* records,
                                          std::size_t count, bool generated,
                                          CudaWeightedEriResult* output) {
  weighted_eri_reference_kernel<<<grid, block, shared_bytes, stream>>>(records, count, generated,
                                                                       output);
}

void launch_weighted_eri_generated_psss_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                               cudaStream_t stream,
                                               const CudaWeightedEriPrimitive* records,
                                               std::size_t count, CudaWeightedEriResult* output) {
  weighted_eri_generated_psss_kernel<<<grid, block, shared_bytes, stream>>>(records, count, output);
}

}  // namespace vibeqc::scf::cuda_execution

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/one_electron_native_attraction_gradient.cuh"
#include "scf/cuda/one_electron_native_overlap.cuh"

// Retained cooperative density-weighted one-electron force exception.
// Normal production S/T/V derivative arithmetic is compiler-owned. This
// implementation remains only for explicit reference/performance comparison.
namespace vibeqc::scf::cuda_execution {

/**
 * Contract one AO pair with nuclear centers distributed across one warp.
 *
 * A block owns one public AO pair. Lane zero evaluates the translation-
 * invariant overlap/kinetic terms once, while every lane owns one point-charge
 * auxiliary center at a time. The large Hermite table depends only on the AO
 * pair and primitive pair, so lane zero materializes it in shared memory and
 * all nuclear-center lanes reuse it for their Coulomb recurrences. Basis-center
 * attraction derivatives are reduced in registers; this preserves the retained
 * reference route's one force update per AO pair and center.
 */
template <unsigned MaximumAngular>
__device__ inline void contracted_one_electron_force_pair_cooperative(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, double density,
    double weighted_density, double* forces,
    OneElectronDerivativeHermiteCoefficients* shared_coefficients) {
  static_assert(MaximumAngular >= 1);
  static_assert(MaximumAngular <= 2 * kMaximumAngularMomentum + 1);
  const unsigned lane = threadIdx.x;
  const std::int64_t ao_i = static_cast<std::int64_t>(system) * batch.nbf + i;
  const std::int64_t ao_j = static_cast<std::int64_t>(system) * batch.nbf + j;
  const std::int32_t shell_i = batch.ao_shells[ao_i];
  const std::int32_t shell_j = batch.ao_shells[ao_j];
  const std::int64_t first_atom = batch.shell_atoms[shell_i];
  const std::int64_t second_atom = batch.shell_atoms[shell_j];
  const Vec3<double> first = atom_position<double>(batch, first_atom, -1);
  const Vec3<double> second = atom_position<double>(batch, second_atom, -1);
  const unsigned first_terms = batch.ao_term_counts[ao_i];
  const unsigned second_terms = batch.ao_term_counts[ao_j];
  const double pair_weight = i == j ? 1.0 : 2.0;
  const double density_scale = -pair_weight * density;
  const double overlap_scale = pair_weight * weighted_density;
  double first_force[3]{};
  double second_force[3]{};

  // These terms do not depend on a point-charge center. Keeping them on lane
  // zero avoids repeating the compact overlap recurrence across the warp.
  if (lane == 0U) {
    for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
         a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
      for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
           b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
        const double primitive_weight =
            batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
        for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
          const Angular angular_first = ao_angular(batch, ao_i, first_term);
          const double first_coefficient = ao_term_coefficient(batch, ao_i, first_term);
          for (unsigned second_term = 0; second_term < second_terms; ++second_term) {
            const Angular angular_second = ao_angular(batch, ao_j, second_term);
            const double weight = primitive_weight * first_coefficient *
                                  ao_term_coefficient(batch, ao_j, second_term);
            double overlap_gradient[3];
            double kinetic_gradient[3];
            primitive_overlap_second_center_gradient(batch.primitive_exponents[a], first,
                                                     angular_first, batch.primitive_exponents[b],
                                                     second, angular_second, overlap_gradient);
            primitive_kinetic_second_center_gradient(batch.primitive_exponents[a], first,
                                                     angular_first, batch.primitive_exponents[b],
                                                     second, angular_second, kinetic_gradient);
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              const double contribution = weight * (density_scale * kinetic_gradient[coordinate] +
                                                    overlap_scale * overlap_gradient[coordinate]);
              first_force[coordinate] -= contribution;
              second_force[coordinate] += contribution;
            }
          }
        }
      }
    }
  }

  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom_base = atom_begin; atom_base < atom_end; atom_base += warpSize) {
    const std::int64_t atom = atom_base + lane;
    double nuclear_force[3]{};
    for (std::int64_t a = batch.shell_primitive_offsets[shell_i];
         a < batch.shell_primitive_offsets[shell_i + 1]; ++a) {
      const double alpha = batch.primitive_exponents[a];
      for (std::int64_t b = batch.shell_primitive_offsets[shell_j];
           b < batch.shell_primitive_offsets[shell_j + 1]; ++b) {
        const double beta = batch.primitive_exponents[b];
        const double exponent = alpha + beta;
        const Vec3<double> product = product_center(alpha, first, beta, second);
        const double primitive_weight =
            batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
        for (unsigned first_term = 0; first_term < first_terms; ++first_term) {
          const Angular angular_first = ao_angular(batch, ao_i, first_term);
          const double first_coefficient = ao_term_coefficient(batch, ao_i, first_term);
          for (unsigned second_term = 0; second_term < second_terms; ++second_term) {
            const Angular angular_second = ao_angular(batch, ao_j, second_term);
            const double weight = density_scale * primitive_weight * first_coefficient *
                                  ao_term_coefficient(batch, ao_j, second_term);
            if (lane == 0U) {
              for (int axis = 0; axis < 3; ++axis) {
                fill_one_electron_derivative_hermite(
                    angular_axis(angular_first, axis) + 1, angular_axis(angular_second, axis) + 1,
                    vec_axis(product, axis), vec_axis(first, axis), vec_axis(second, axis), alpha,
                    beta, shared_coefficients[axis]);
              }
            }
            __syncthreads();
            if (atom < atom_end) {
              double first_gradient[3];
              double second_gradient[3];
              primitive_nuclear_attraction_cartesian_atom_gradient_from_hermite<MaximumAngular>(
                  batch, alpha, angular_first, beta, angular_second, atom, exponent, product,
                  shared_coefficients, first_gradient, second_gradient);
              for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
                first_force[coordinate] += weight * first_gradient[coordinate];
                second_force[coordinate] += weight * second_gradient[coordinate];
                nuclear_force[coordinate] -=
                    weight * (first_gradient[coordinate] + second_gradient[coordinate]);
              }
            }
            // No lane may overwrite the shared Hermite table while another
            // lane is still evaluating its point-charge recurrence.
            __syncthreads();
          }
        }
      }
    }
    if (atom < atom_end) {
      const std::int64_t coordinate = atom * 3;
      for (unsigned axis = 0; axis < 3; ++axis) {
        if (nuclear_force[axis] != 0.0) {
          atomicAdd(forces + coordinate + axis, nuclear_force[axis]);
        }
      }
    }
  }

  for (unsigned offset = warpSize / 2; offset != 0; offset /= 2) {
    for (unsigned axis = 0; axis < 3; ++axis) {
      first_force[axis] += __shfl_down_sync(0xffffffffU, first_force[axis], offset);
      second_force[axis] += __shfl_down_sync(0xffffffffU, second_force[axis], offset);
    }
  }
  if (lane == 0U) {
    const std::int64_t first_coordinate = first_atom * 3;
    const std::int64_t second_coordinate = second_atom * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (first_force[axis] != 0.0) {
        atomicAdd(forces + first_coordinate + axis, first_force[axis]);
      }
      if (second_force[axis] != 0.0) {
        atomicAdd(forces + second_coordinate + axis, second_force[axis]);
      }
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution

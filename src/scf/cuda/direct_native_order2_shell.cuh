#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/direct_native_eri_order2.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for order2 shell.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** Maximum full Cartesian output count among the three order-two classes. */
struct Order2IntegralVector {
  double component[9];
};

/**
 * Contract one canonical order-two shell quartet into all requested outputs.
 *
 * `active_component_mask` uses the canonical full Cartesian product order.
 * Symmetry-diagonal tasks therefore avoid computing product entries that are
 * absent from their lower-triangular AO-quartet domain.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular>
__device__ inline __noinline__ Order2IntegralVector contracted_eri_cartesian_source_order2_shell(
    const DeviceBatch& batch, std::int32_t first_shell, std::int32_t second_shell,
    std::int32_t third_shell, std::int32_t fourth_shell, unsigned active_component_mask) {
  static_assert(FirstShellAngular + SecondShellAngular + ThirdShellAngular + FourthShellAngular ==
                2);
  constexpr unsigned FirstCount = order2_shell_component_count<FirstShellAngular>();
  constexpr unsigned SecondCount = order2_shell_component_count<SecondShellAngular>();
  constexpr unsigned ThirdCount = order2_shell_component_count<ThirdShellAngular>();
  constexpr unsigned FourthCount = order2_shell_component_count<FourthShellAngular>();
  constexpr unsigned OutputCount = FirstCount * SecondCount * ThirdCount * FourthCount;
  static_assert(OutputCount <= 9);

  const std::int32_t shells[4] = {first_shell, second_shell, third_shell, fourth_shell};
  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[first_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[second_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[third_shell], -1),
      atom_position<double>(batch, batch.shell_atoms[fourth_shell], -1),
  };
  const std::int64_t ao_begin[4] = {
      batch.shell_direct_ao_offsets[first_shell],
      batch.shell_direct_ao_offsets[second_shell],
      batch.shell_direct_ao_offsets[third_shell],
      batch.shell_direct_ao_offsets[fourth_shell],
  };
  const double first_pair_distance = distance_squared(positions[0], positions[1]);
  const double second_pair_distance = distance_squared(positions[2], positions[3]);

  Order2IntegralVector result{};
  for (std::int64_t a = batch.shell_primitive_offsets[shells[0]];
       a < batch.shell_primitive_offsets[shells[0] + 1]; ++a) {
    const double alpha = batch.primitive_exponents[a];
    const double coefficient_a = batch.primitive_coefficients[a];
    for (std::int64_t b = batch.shell_primitive_offsets[shells[1]];
         b < batch.shell_primitive_offsets[shells[1] + 1]; ++b) {
      const double beta = batch.primitive_exponents[b];
      const double p = alpha + beta;
      const double mu = alpha * beta / p;
      const Vec3<double> product_p = product_center(alpha, positions[0], beta, positions[1]);
      const double first_pair_coefficient = coefficient_a * batch.primitive_coefficients[b];
      for (std::int64_t c = batch.shell_primitive_offsets[shells[2]];
           c < batch.shell_primitive_offsets[shells[2] + 1]; ++c) {
        const double gamma = batch.primitive_exponents[c];
        const double first_three_coefficient =
            first_pair_coefficient * batch.primitive_coefficients[c];
        for (std::int64_t d = batch.shell_primitive_offsets[shells[3]];
             d < batch.shell_primitive_offsets[shells[3] + 1]; ++d) {
          const double delta = batch.primitive_exponents[d];
          const double q = gamma + delta;
          const double nu = gamma * delta / q;
          const double rho = p * q / (p + q);
          const Vec3<double> product_q = product_center(gamma, positions[2], delta, positions[3]);
          const double x = product_p.x - product_q.x;
          const double y = product_p.y - product_q.y;
          const double z = product_p.z - product_q.z;
          double boys[3];
          boys_values<2>(rho * (x * x + y * y + z * z), boys);
          const Order2CoulombValues coulomb = order2_coulomb_values(rho, x, y, z, boys);
          const double pair_decay = exp(-mu * first_pair_distance - nu * second_pair_distance);
          const double primitive_coefficient =
              first_three_coefficient * batch.primitive_coefficients[d];
          const double common =
              primitive_coefficient * 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;

          if constexpr (FirstShellAngular == 1 && SecondShellAngular == 0 &&
                        ThirdShellAngular == 1 && FourthShellAngular == 0) {
            const double hp = 0.5 / p;
            const double hq = 0.5 / q;
            const double hpq = hp * hq;
            const Vec3<double> pa{product_p.x - positions[0].x, product_p.y - positions[0].y,
                                  product_p.z - positions[0].z};
            const Vec3<double> qc{product_q.x - positions[2].x, product_q.y - positions[2].y,
                                  product_q.z - positions[2].z};
            if ((active_component_mask & (1U << 0)) != 0U) {
              result.component[0] += common * (pa.x * qc.x * coulomb.c0 + hp * qc.x * coulomb.cx -
                                               hq * pa.x * coulomb.cx - hpq * coulomb.cxx);
            }
            if ((active_component_mask & (1U << 1)) != 0U) {
              result.component[1] += common * (pa.x * qc.y * coulomb.c0 + hp * qc.y * coulomb.cx -
                                               hq * pa.x * coulomb.cy - hpq * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 2)) != 0U) {
              result.component[2] += common * (pa.x * qc.z * coulomb.c0 + hp * qc.z * coulomb.cx -
                                               hq * pa.x * coulomb.cz - hpq * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 3)) != 0U) {
              result.component[3] += common * (pa.y * qc.x * coulomb.c0 + hp * qc.x * coulomb.cy -
                                               hq * pa.y * coulomb.cx - hpq * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 4)) != 0U) {
              result.component[4] += common * (pa.y * qc.y * coulomb.c0 + hp * qc.y * coulomb.cy -
                                               hq * pa.y * coulomb.cy - hpq * coulomb.cyy);
            }
            if ((active_component_mask & (1U << 5)) != 0U) {
              result.component[5] += common * (pa.y * qc.z * coulomb.c0 + hp * qc.z * coulomb.cy -
                                               hq * pa.y * coulomb.cz - hpq * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 6)) != 0U) {
              result.component[6] += common * (pa.z * qc.x * coulomb.c0 + hp * qc.x * coulomb.cz -
                                               hq * pa.z * coulomb.cx - hpq * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 7)) != 0U) {
              result.component[7] += common * (pa.z * qc.y * coulomb.c0 + hp * qc.y * coulomb.cz -
                                               hq * pa.z * coulomb.cy - hpq * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 8)) != 0U) {
              result.component[8] += common * (pa.z * qc.z * coulomb.c0 + hp * qc.z * coulomb.cz -
                                               hq * pa.z * coulomb.cz - hpq * coulomb.czz);
            }
          } else if constexpr (FirstShellAngular == 1 && SecondShellAngular == 1) {
            const double h = 0.5 / p;
            const double h2 = h * h;
            const Vec3<double> pa{product_p.x - positions[0].x, product_p.y - positions[0].y,
                                  product_p.z - positions[0].z};
            const Vec3<double> pb{product_p.x - positions[1].x, product_p.y - positions[1].y,
                                  product_p.z - positions[1].z};
            if ((active_component_mask & (1U << 0)) != 0U) {
              result.component[0] += common * ((pa.x * pb.x + h) * coulomb.c0 +
                                               h * (pb.x + pa.x) * coulomb.cx + h2 * coulomb.cxx);
            }
            if ((active_component_mask & (1U << 1)) != 0U) {
              result.component[1] += common * (pa.x * pb.y * coulomb.c0 + h * pb.y * coulomb.cx +
                                               h * pa.x * coulomb.cy + h2 * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 2)) != 0U) {
              result.component[2] += common * (pa.x * pb.z * coulomb.c0 + h * pb.z * coulomb.cx +
                                               h * pa.x * coulomb.cz + h2 * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 3)) != 0U) {
              result.component[3] += common * (pa.y * pb.x * coulomb.c0 + h * pb.x * coulomb.cy +
                                               h * pa.y * coulomb.cx + h2 * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 4)) != 0U) {
              result.component[4] += common * ((pa.y * pb.y + h) * coulomb.c0 +
                                               h * (pb.y + pa.y) * coulomb.cy + h2 * coulomb.cyy);
            }
            if ((active_component_mask & (1U << 5)) != 0U) {
              result.component[5] += common * (pa.y * pb.z * coulomb.c0 + h * pb.z * coulomb.cy +
                                               h * pa.y * coulomb.cz + h2 * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 6)) != 0U) {
              result.component[6] += common * (pa.z * pb.x * coulomb.c0 + h * pb.x * coulomb.cz +
                                               h * pa.z * coulomb.cx + h2 * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 7)) != 0U) {
              result.component[7] += common * (pa.z * pb.y * coulomb.c0 + h * pb.y * coulomb.cz +
                                               h * pa.z * coulomb.cy + h2 * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 8)) != 0U) {
              result.component[8] += common * ((pa.z * pb.z + h) * coulomb.c0 +
                                               h * (pb.z + pa.z) * coulomb.cz + h2 * coulomb.czz);
            }
          } else {
            static_assert(FirstShellAngular == 2 && SecondShellAngular == 0 &&
                          ThirdShellAngular == 0 && FourthShellAngular == 0);
            const double h = 0.5 / p;
            const double h2 = h * h;
            const Vec3<double> pa{product_p.x - positions[0].x, product_p.y - positions[0].y,
                                  product_p.z - positions[0].z};
            if ((active_component_mask & (1U << 0)) != 0U) {
              result.component[0] += common * ((pa.x * pa.x + h) * coulomb.c0 +
                                               2.0 * h * pa.x * coulomb.cx + h2 * coulomb.cxx);
            }
            if ((active_component_mask & (1U << 1)) != 0U) {
              result.component[1] += common * (pa.x * pa.y * coulomb.c0 + h * pa.y * coulomb.cx +
                                               h * pa.x * coulomb.cy + h2 * coulomb.cxy);
            }
            if ((active_component_mask & (1U << 2)) != 0U) {
              result.component[2] += common * (pa.x * pa.z * coulomb.c0 + h * pa.z * coulomb.cx +
                                               h * pa.x * coulomb.cz + h2 * coulomb.cxz);
            }
            if ((active_component_mask & (1U << 3)) != 0U) {
              result.component[3] += common * ((pa.y * pa.y + h) * coulomb.c0 +
                                               2.0 * h * pa.y * coulomb.cy + h2 * coulomb.cyy);
            }
            if ((active_component_mask & (1U << 4)) != 0U) {
              result.component[4] += common * (pa.y * pa.z * coulomb.c0 + h * pa.z * coulomb.cy +
                                               h * pa.y * coulomb.cz + h2 * coulomb.cyz);
            }
            if ((active_component_mask & (1U << 5)) != 0U) {
              result.component[5] += common * ((pa.z * pa.z + h) * coulomb.c0 +
                                               2.0 * h * pa.z * coulomb.cz + h2 * coulomb.czz);
            }
          }
        }
      }
    }
  }

  // Cartesian normalization is primitive-independent. Applying it once after
  // contraction avoids four coefficient loads for every component of every
  // primitive quartet.
  unsigned output = 0;
#pragma unroll
  for (unsigned first_component = 0; first_component < FirstCount; ++first_component) {
#pragma unroll
    for (unsigned second_component = 0; second_component < SecondCount; ++second_component) {
#pragma unroll
      for (unsigned third_component = 0; third_component < ThirdCount; ++third_component) {
#pragma unroll
        for (unsigned fourth_component = 0; fourth_component < FourthCount;
             ++fourth_component, ++output) {
          if ((active_component_mask & (1U << output)) == 0U) continue;
          result.component[output] *= batch.direct_ao_coefficients[ao_begin[0] + first_component] *
                                      batch.direct_ao_coefficients[ao_begin[1] + second_component] *
                                      batch.direct_ao_coefficients[ao_begin[2] + third_component] *
                                      batch.direct_ao_coefficients[ao_begin[3] + fourth_component];
        }
      }
    }
  }
  return result;
}

}  // namespace vibeqc::scf::cuda_execution

"""Emit low-order weighted analytic-force CUDA workers.

The cooperative component-per-lane schedule used by higher shell classes is
poorly matched to ``psps`` because only nine Cartesian components would keep
lanes busy.  This emitter instead assigns one complete compacted shell task to
each CUDA thread.  The thread forms all nine screened density coefficients,
traverses each primitive quartet once, and contracts the complete analytic
gradient before writing the four physical centers.
"""

from __future__ import annotations


PSPS_BLOCK_THREADS = 256


_PSPS_WEIGHTED_FORCE_CUDA = r"""
struct GeneratedPspsVec3 {
  double x;
  double y;
  double z;
};

struct GeneratedPspsPrimitivePairData {
  double exponent_sum;
  double reduced_exponent;
  GeneratedPspsVec3 product_center;
  double weighted_coefficient;
  double first_product_scale;
  double second_product_scale;
};

struct GeneratedPspsShellTask {
  std::uint64_t primitive_begin[4];
  std::uint64_t primitive_end[4];
  std::uint64_t ao_begin[4];
  std::uint64_t ao_coefficient_begin[4];
  std::uint64_t density_offset;
  std::uint64_t spin_offset;
  std::uint32_t matrix_order;
  std::uint32_t shell_pair[2];
  std::uint32_t reversed_shell_pair_mask;
  std::uint32_t shell[4];
  std::uint32_t atom[4];
};

struct GeneratedPspsWeightedCoulomb {
  double c0;
  double cx;
  double cy;
  double cz;
  double value;
  double chain[3];
};

constexpr unsigned kGeneratedPspsBlockThreads = 256U;

__device__ __forceinline__ std::size_t generated_psps_matrix_index(
    std::size_t row, std::size_t column, std::size_t order) {
  return row + column * order;
}

__device__ __forceinline__ double generated_psps_axis(
    const GeneratedPspsVec3& value, unsigned axis) {
  return axis == 0U ? value.x : (axis == 1U ? value.y : value.z);
}

__device__ __forceinline__ void generated_psps_eri_permutation(
    unsigned permutation,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    std::size_t& a, std::size_t& b, std::size_t& c, std::size_t& d) {
  switch (permutation) {
    case 0: a = i; b = j; c = k; d = l; break;
    case 1: a = j; b = i; c = k; d = l; break;
    case 2: a = i; b = j; c = l; d = k; break;
    case 3: a = j; b = i; c = l; d = k; break;
    case 4: a = k; b = l; c = i; d = j; break;
    case 5: a = l; b = k; c = i; d = j; break;
    case 6: a = k; b = l; c = j; d = i; break;
    default: a = l; b = k; c = j; d = i; break;
  }
}

__device__ __forceinline__ bool generated_psps_unique_permutation(
    unsigned permutation,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
#pragma unroll
  for (unsigned previous = 0; previous < 8U; ++previous) {
    if (previous >= permutation) break;
    std::size_t pa = 0, pb = 0, pc = 0, pd = 0;
    generated_psps_eri_permutation(
        previous, i, j, k, l, pa, pb, pc, pd);
    if (a == pa && b == pb && c == pc && d == pd) return false;
  }
  return true;
}

template <bool Unrestricted>
__device__ __forceinline__ double generated_psps_density_coefficient(
    const GeneratedPspsShellTask& task,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l,
    const double* density) {
  const std::size_t n = static_cast<std::size_t>(task.matrix_order);
  const std::size_t matrix_size = n * n;
  double coefficient = 0.0;
#pragma unroll
  for (unsigned permutation = 0; permutation < 8U; ++permutation) {
    std::size_t a = 0, b = 0, c = 0, d = 0;
    generated_psps_eri_permutation(
        permutation, i, j, k, l, a, b, c, d);
    if (!generated_psps_unique_permutation(
            permutation, i, j, k, l, a, b, c, d)) continue;
    const std::size_t ab = generated_psps_matrix_index(a, b, n);
    const std::size_t ac = generated_psps_matrix_index(a, c, n);
    const std::size_t cd = generated_psps_matrix_index(c, d, n);
    const std::size_t bd = generated_psps_matrix_index(b, d, n);
    if constexpr (Unrestricted) {
      const double total_ab = density[task.spin_offset + ab] +
          density[task.spin_offset + matrix_size + ab];
      const double total_cd = density[task.spin_offset + cd] +
          density[task.spin_offset + matrix_size + cd];
      coefficient += 0.5 * total_ab * total_cd;
      coefficient -= 0.5 *
          (density[task.spin_offset + ac] * density[task.spin_offset + bd] +
           density[task.spin_offset + matrix_size + ac] *
               density[task.spin_offset + matrix_size + bd]);
    } else {
      coefficient +=
          0.5 * density[task.density_offset + ab] *
              density[task.density_offset + cd] -
          0.25 * density[task.density_offset + ac] *
              density[task.density_offset + bd];
    }
  }
  return coefficient;
}

__device__ __forceinline__ GeneratedPspsWeightedCoulomb
generated_psps_contract_weighted_coulomb(
    double rho, double x, double y, double z, const double (&boys)[4],
    double h0, double hx, double hy, double hz,
    double hxx, double hxy, double hxz,
    double hyy, double hyz, double hzz) {
  const double twice_rho = 2.0 * rho;
  const double twice_rho_squared = twice_rho * twice_rho;
  const double twice_rho_cubed = twice_rho_squared * twice_rho;
  const double c0 = boys[0];
  const double cx = -twice_rho * x * boys[1];
  const double cy = -twice_rho * y * boys[1];
  const double cz = -twice_rho * z * boys[1];
  const double cxx =
      twice_rho_squared * x * x * boys[2] - twice_rho * boys[1];
  const double cxy = twice_rho_squared * x * y * boys[2];
  const double cxz = twice_rho_squared * x * z * boys[2];
  const double cyy =
      twice_rho_squared * y * y * boys[2] - twice_rho * boys[1];
  const double cyz = twice_rho_squared * y * z * boys[2];
  const double czz =
      twice_rho_squared * z * z * boys[2] - twice_rho * boys[1];
  const double cxxx =
      -twice_rho_cubed * x * x * x * boys[3] +
      3.0 * twice_rho_squared * x * boys[2];
  const double cxxy =
      -twice_rho_cubed * x * x * y * boys[3] +
      twice_rho_squared * y * boys[2];
  const double cxxz =
      -twice_rho_cubed * x * x * z * boys[3] +
      twice_rho_squared * z * boys[2];
  const double cxyy =
      -twice_rho_cubed * x * y * y * boys[3] +
      twice_rho_squared * x * boys[2];
  const double cxyz = -twice_rho_cubed * x * y * z * boys[3];
  const double cxzz =
      -twice_rho_cubed * x * z * z * boys[3] +
      twice_rho_squared * x * boys[2];
  const double cyyy =
      -twice_rho_cubed * y * y * y * boys[3] +
      3.0 * twice_rho_squared * y * boys[2];
  const double cyyz =
      -twice_rho_cubed * y * y * z * boys[3] +
      twice_rho_squared * z * boys[2];
  const double cyzz =
      -twice_rho_cubed * y * z * z * boys[3] +
      twice_rho_squared * y * boys[2];
  const double czzz =
      -twice_rho_cubed * z * z * z * boys[3] +
      3.0 * twice_rho_squared * z * boys[2];

  GeneratedPspsWeightedCoulomb result{};
  result.c0 = c0;
  result.cx = cx;
  result.cy = cy;
  result.cz = cz;
  result.value =
      h0 * c0 + hx * cx + hy * cy + hz * cz +
      hxx * cxx + hxy * cxy + hxz * cxz +
      hyy * cyy + hyz * cyz + hzz * czz;
  result.chain[0] =
      h0 * cx + hx * cxx + hy * cxy + hz * cxz +
      hxx * cxxx + hxy * cxxy + hxz * cxxz +
      hyy * cxyy + hyz * cxyz + hzz * cxzz;
  result.chain[1] =
      h0 * cy + hx * cxy + hy * cyy + hz * cyz +
      hxx * cxxy + hxy * cxyy + hxz * cxyz +
      hyy * cyyy + hyz * cyyz + hzz * cyzz;
  result.chain[2] =
      h0 * cz + hx * cxz + hy * cyz + hz * czz +
      hxx * cxxz + hxy * cxyz + hxz * cxzz +
      hyy * cyyz + hyz * cyzz + hzz * czzz;
  return result;
}

template <bool Unrestricted>
__device__ __forceinline__ void generated_psps_force_task(
    const GeneratedPspsShellTask* tasks,
    const GeneratedPspsPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedPspsVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    std::size_t task_index) {
  const GeneratedPspsShellTask& task = tasks[task_index];
  const std::size_t n = static_cast<std::size_t>(task.matrix_order);
  const std::size_t first_p_ao = static_cast<std::size_t>(task.ao_begin[0]);
  const std::size_t first_s_ao = static_cast<std::size_t>(task.ao_begin[1]);
  const std::size_t second_p_ao = static_cast<std::size_t>(task.ao_begin[2]);
  const std::size_t second_s_ao = static_cast<std::size_t>(task.ao_begin[3]);
  const bool same_shell_pair = task.shell_pair[0] == task.shell_pair[1];
  double component_weight[9]{};
  bool any_component = false;

#pragma unroll
  for (unsigned first_axis = 0; first_axis < 3U; ++first_axis) {
#pragma unroll
    for (unsigned second_axis = 0; second_axis < 3U; ++second_axis) {
      if (same_shell_pair && first_axis < second_axis) continue;
      const std::size_t i = first_p_ao + first_axis;
      const std::size_t j = first_s_ao;
      const std::size_t k = second_p_ao + second_axis;
      const std::size_t l = second_s_ao;
      if (schwarz_bounds != nullptr &&
          schwarz_bounds[
              task.density_offset + generated_psps_matrix_index(i, j, n)] *
              schwarz_bounds[
                  task.density_offset +
                  generated_psps_matrix_index(k, l, n)] <
              screening_tolerance) {
        continue;
      }
      const double density_coefficient =
          generated_psps_density_coefficient<Unrestricted>(
              task, i, j, k, l, density);
      if (density_coefficient == 0.0) continue;
      const double angular_coefficient =
          ao_coefficients[task.ao_coefficient_begin[0] + first_axis] *
          ao_coefficients[task.ao_coefficient_begin[1]] *
          ao_coefficients[task.ao_coefficient_begin[2] + second_axis] *
          ao_coefficients[task.ao_coefficient_begin[3]];
      component_weight[first_axis * 3U + second_axis] =
          density_coefficient * angular_coefficient;
      any_component = true;
    }
  }
  if (!any_component) return;

  const GeneratedPspsVec3 first = atom_positions[task.atom[0]];
  const GeneratedPspsVec3 second = atom_positions[task.atom[1]];
  const GeneratedPspsVec3 third = atom_positions[task.atom[2]];
  const GeneratedPspsVec3 fourth = atom_positions[task.atom[3]];
  const bool first_pair_reversed =
      (task.reversed_shell_pair_mask & 1U) != 0U;
  const bool second_pair_reversed =
      (task.reversed_shell_pair_mask & 2U) != 0U;
  double gradient[3][3]{};

  const std::int64_t first_pair_begin =
      primitive_pair_offsets[task.shell_pair[0]];
  const std::int64_t first_pair_end =
      primitive_pair_offsets[task.shell_pair[0] + 1U];
  const std::int64_t second_pair_begin =
      primitive_pair_offsets[task.shell_pair[1]];
  const std::int64_t second_pair_end =
      primitive_pair_offsets[task.shell_pair[1] + 1U];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end; ++first_primitive) {
    const GeneratedPspsPrimitivePairData first_pair =
        primitive_pairs[first_primitive];
    const double p = first_pair.exponent_sum;
    const double mu = first_pair.reduced_exponent;
    const double inverse_two_p = 0.5 / p;
    const GeneratedPspsVec3 product_p = first_pair.product_center;
    const double pa[3] = {
        product_p.x - first.x,
        product_p.y - first.y,
        product_p.z - first.z,
    };
    const double first_product_scale = first_pair_reversed
        ? first_pair.second_product_scale
        : first_pair.first_product_scale;
    const double second_product_scale = first_pair_reversed
        ? first_pair.first_product_scale
        : first_pair.second_product_scale;
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end; ++second_primitive) {
      const GeneratedPspsPrimitivePairData second_pair =
          primitive_pairs[second_primitive];
      const double q = second_pair.exponent_sum;
      const double nu = second_pair.reduced_exponent;
      const double inverse_two_q = 0.5 / q;
      const double rho = p * q / (p + q);
      const GeneratedPspsVec3 product_q = second_pair.product_center;
      const double qc[3] = {
          product_q.x - third.x,
          product_q.y - third.y,
          product_q.z - third.z,
      };
      const double x = product_p.x - product_q.x;
      const double y = product_p.y - product_q.y;
      const double z = product_p.z - product_q.z;
      double boys[4];
      boys_values<3>(rho * (x * x + y * y + z * z), boys);

      const double row[3] = {
          component_weight[0] * qc[0] + component_weight[1] * qc[1] +
              component_weight[2] * qc[2],
          component_weight[3] * qc[0] + component_weight[4] * qc[1] +
              component_weight[5] * qc[2],
          component_weight[6] * qc[0] + component_weight[7] * qc[1] +
              component_weight[8] * qc[2],
      };
      const double column[3] = {
          component_weight[0] * pa[0] + component_weight[3] * pa[1] +
              component_weight[6] * pa[2],
          component_weight[1] * pa[0] + component_weight[4] * pa[1] +
              component_weight[7] * pa[2],
          component_weight[2] * pa[0] + component_weight[5] * pa[1] +
              component_weight[8] * pa[2],
      };
      const double h0 = pa[0] * row[0] + pa[1] * row[1] + pa[2] * row[2];
      const double hx = inverse_two_p * row[0] - inverse_two_q * column[0];
      const double hy = inverse_two_p * row[1] - inverse_two_q * column[1];
      const double hz = inverse_two_p * row[2] - inverse_two_q * column[2];
      const double second_scale = -inverse_two_p * inverse_two_q;
      const GeneratedPspsWeightedCoulomb coulomb =
          generated_psps_contract_weighted_coulomb(
              rho, x, y, z, boys, h0, hx, hy, hz,
              second_scale * component_weight[0],
              second_scale * (component_weight[1] + component_weight[3]),
              second_scale * (component_weight[2] + component_weight[6]),
              second_scale * component_weight[4],
              second_scale * (component_weight[5] + component_weight[7]),
              second_scale * component_weight[8]);

      const double first_explicit[3] = {
          row[0] * coulomb.c0 -
              inverse_two_q *
                  (component_weight[0] * coulomb.cx +
                   component_weight[1] * coulomb.cy +
                   component_weight[2] * coulomb.cz),
          row[1] * coulomb.c0 -
              inverse_two_q *
                  (component_weight[3] * coulomb.cx +
                   component_weight[4] * coulomb.cy +
                   component_weight[5] * coulomb.cz),
          row[2] * coulomb.c0 -
              inverse_two_q *
                  (component_weight[6] * coulomb.cx +
                   component_weight[7] * coulomb.cy +
                   component_weight[8] * coulomb.cz),
      };
      const double second_explicit[3] = {
          column[0] * coulomb.c0 +
              inverse_two_p *
                  (component_weight[0] * coulomb.cx +
                   component_weight[3] * coulomb.cy +
                   component_weight[6] * coulomb.cz),
          column[1] * coulomb.c0 +
              inverse_two_p *
                  (component_weight[1] * coulomb.cx +
                   component_weight[4] * coulomb.cy +
                   component_weight[7] * coulomb.cz),
          column[2] * coulomb.c0 +
              inverse_two_p *
                  (component_weight[2] * coulomb.cx +
                   component_weight[5] * coulomb.cy +
                   component_weight[8] * coulomb.cz),
      };
      const double third_product_scale = second_pair_reversed
          ? second_pair.second_product_scale
          : second_pair.first_product_scale;
      const double product_scale[3] = {
          first_product_scale, second_product_scale, -third_product_scale};
      const double shift_scale[3] = {
          first_product_scale - 1.0,
          second_product_scale,
          third_product_scale - 1.0,
      };
      const double prefactor =
          first_pair.weighted_coefficient *
          second_pair.weighted_coefficient *
          34.986836655249725 / (p * q * sqrt(p + q));

#pragma unroll
      for (unsigned center = 0; center < 3U; ++center) {
#pragma unroll
        for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {
          const double pair_coefficient_derivative =
              shift_scale[center] *
              (center < 2U ? first_explicit[coordinate]
                           : second_explicit[coordinate]);
          const double first_difference =
              generated_psps_axis(first, coordinate) -
              generated_psps_axis(second, coordinate);
          const double third_difference =
              generated_psps_axis(third, coordinate) -
              generated_psps_axis(fourth, coordinate);
          const double decay_derivative = center < 2U
              ? (center == 0U ? -2.0 * mu : 2.0 * mu) * first_difference
              : -2.0 * nu * third_difference;
          gradient[center][coordinate] += prefactor *
              (pair_coefficient_derivative +
               product_scale[center] * coulomb.chain[coordinate] +
               coulomb.value * decay_derivative);
        }
      }
    }
  }

  double center_gradient[4][3];
#pragma unroll
  for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {
    center_gradient[0][coordinate] = gradient[0][coordinate];
    center_gradient[1][coordinate] = gradient[1][coordinate];
    center_gradient[2][coordinate] = gradient[2][coordinate];
    center_gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] -
        gradient[2][coordinate];
  }
#pragma unroll
  for (unsigned center = 0; center < 4U; ++center) {
    bool first_occurrence = true;
#pragma unroll
    for (unsigned previous = 0; previous < 4U; ++previous) {
      if (previous >= center) break;
      first_occurrence = first_occurrence &&
          task.atom[previous] != task.atom[center];
    }
    if (!first_occurrence) continue;
#pragma unroll
    for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {
      double derivative = 0.0;
#pragma unroll
      for (unsigned source = 0; source < 4U; ++source) {
        if (task.atom[source] == task.atom[center]) {
          derivative += center_gradient[source][coordinate];
        }
      }
      if (derivative != 0.0) {
        atomicAdd(
            forces + static_cast<std::size_t>(task.atom[center]) * 3U +
                coordinate,
            -derivative);
      }
    }
  }
}

template <bool Unrestricted>
__device__ __forceinline__ void generated_psps_force_persistent(
    const GeneratedPspsShellTask* tasks,
    const GeneratedPspsPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedPspsVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {
  const unsigned warp_lane = threadIdx.x % 32U;
  while (true) {
    std::uint32_t task_begin = 0U;
    if (warp_lane == 0U) task_begin = atomicAdd(task_head, 32U);
    task_begin = __shfl_sync(0xffffffffU, task_begin, 0);
    if (task_begin >= *task_count) return;
    const std::uint32_t task_index = task_begin + warp_lane;
    if (task_index < *task_count) {
      generated_psps_force_task<Unrestricted>(
          tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
          atom_positions, screening_tolerance, schwarz_bounds, density,
          forces,
          static_cast<std::size_t>(*task_offset + task_index));
    }
  }
}

extern "C" __global__ __launch_bounds__(kGeneratedPspsBlockThreads, 1)
void generated_psps_shell_class_force_rhf_persistent_kernel(
    const GeneratedPspsShellTask* tasks,
    const GeneratedPspsPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedPspsVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {
  generated_psps_force_persistent<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}

extern "C" __global__ __launch_bounds__(kGeneratedPspsBlockThreads, 1)
void generated_psps_shell_class_force_uhf_persistent_kernel(
    const GeneratedPspsShellTask* tasks,
    const GeneratedPspsPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedPspsVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* forces,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {
  generated_psps_force_persistent<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, forces,
      task_offset, task_count, task_head);
}
"""


def emit_psps_weighted_force_cuda() -> str:
    """Return the deterministic thread-per-task ``psps`` CUDA worker."""

    return _PSPS_WEIGHTED_FORCE_CUDA

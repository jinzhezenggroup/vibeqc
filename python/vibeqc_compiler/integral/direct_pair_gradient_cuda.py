"""Compiler-owned CUDA emission for the Direct-HF high-order force consumer.

The native Direct runtime owns queue traversal and scheduling only.  This module
owns order-4/5/6 source canonicalization, primitive contraction, subset/Wick pair
algebra, and complete primitive ERI-gradient composition.  Emission preserves the
qualified arithmetic and reduction order while retiring the native scientific body.
"""

from __future__ import annotations


def emit_direct_high_order_pair_gradient_header() -> str:
    """Emit compiler-owned order-4/5/6 Direct force-consumer CUDA."""

    return r"""#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_eri_order4.cuh"
#include "scf/cuda/direct_native_gradient_types.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/direct_native_high_order_coulomb.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Generated from the compiler-owned high-order primitive-gradient lowering.
// Do not edit this build artifact: change
// python/vibeqc_compiler/integral/direct_pair_gradient_cuda.py instead.
// Native Direct-HF retains queue/runtime scheduling only; the order-4/5/6
// canonicalization and primitive-contraction consumer is emitted here as well.
namespace vibeqc::scf::cuda_execution {

/** One sparse coefficient term and its first-center gradient. */
struct HighOrderPairGradientTerm {
  unsigned derivative_state;
  double coefficient;
  double first_center[3];
};

template <unsigned PairOrder>
struct HighOrderPairGradientGeometry {
  static_assert(PairOrder <= 6);
  static constexpr unsigned QuantumStorage = PairOrder == 0 ? 1 : PairOrder;
  unsigned axes[QuantumStorage];
  double shifts[QuantumStorage];
  double first_center_shift_gradients[QuantumStorage];
  double inverse_two_exponent;
};

/** Add one Wick matching to one derivative subset and its gradient. */
template <unsigned PairOrder>
__device__ inline void add_high_order_wick_matching_term(
    HighOrderPairGradientTerm& term, const HighOrderPairGradientGeometry<PairOrder>& geometry,
    unsigned subset, unsigned removed, unsigned contraction_count) {
  static_assert(PairOrder <= 6);
  if ((subset & removed) != 0) return;
  double inverse_factor = 1.0;
  const unsigned inverse_count = contraction_count + static_cast<unsigned>(__popc(subset));
  for (unsigned factor = 0; factor < inverse_count; ++factor) {
    inverse_factor *= geometry.inverse_two_exponent;
  }

  double coefficient = inverse_factor;
  for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
    const unsigned bit = 1U << quantum;
    if (((subset | removed) & bit) == 0) {
      coefficient *= geometry.shifts[quantum];
    }
  }
  term.coefficient += coefficient;

  // Differentiate the same surviving product while its factors are hot.
  // The old path revisited every matching once for coefficients and three
  // more times for Cartesian gradients. Accumulating by the differentiated
  // quantum avoids that repeated combinatorial walk and keeps the gradient
  // sparse in its quantum's Cartesian axis.
  for (unsigned differentiated = 0; differentiated < PairOrder; ++differentiated) {
    const unsigned differentiated_bit = 1U << differentiated;
    if (((subset | removed) & differentiated_bit) != 0) continue;
    double derivative = inverse_factor * geometry.first_center_shift_gradients[differentiated];
    for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
      const unsigned bit = 1U << quantum;
      if (quantum == differentiated || ((subset | removed) & bit) != 0) {
        continue;
      }
      derivative *= geometry.shifts[quantum];
    }
    term.first_center[geometry.axes[differentiated]] += derivative;
  }
}

/**
 * Generate the exact subset/Wick pair expansion through angular order six.
 *
 * One contraction removes a same-axis quantum pair. Three disjoint
 * contractions are the highest possible matching at order six. Expanding
 * every surviving quantum into either its center shift or Hermite derivative
 * covers the complete Gaussian product recurrence without a dense
 * coefficient workspace. Pair masks are ordered to visit every disjoint Wick
 * matching exactly once.
 */
template <unsigned PairOrder>
__device__ inline HighOrderPairGradientGeometry<PairOrder> make_high_order_pair_gradient_geometry(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second) {
  static_assert(PairOrder <= 6);
  HighOrderPairGradientGeometry<PairOrder> geometry{};
  if constexpr (PairOrder != 0) {
    const double exponent = alpha + beta;
    const Vec3<double> product = product_center(alpha, first, beta, second);
    geometry.inverse_two_exponent = 0.5 / exponent;
    unsigned quantum_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
      for (unsigned quantum = 0; quantum < angular_axis(angular_first, axis); ++quantum) {
        geometry.axes[quantum_count] = static_cast<unsigned>(axis);
        geometry.shifts[quantum_count] = vec_axis(product, axis) - vec_axis(first, axis);
        geometry.first_center_shift_gradients[quantum_count] = alpha / exponent - 1.0;
        ++quantum_count;
      }
      for (unsigned quantum = 0; quantum < angular_axis(angular_second, axis); ++quantum) {
        geometry.axes[quantum_count] = static_cast<unsigned>(axis);
        geometry.shifts[quantum_count] = vec_axis(product, axis) - vec_axis(second, axis);
        geometry.first_center_shift_gradients[quantum_count] = alpha / exponent;
        ++quantum_count;
      }
    }
  }
  return geometry;
}

/** Generate one exact subset/Wick coefficient and its center gradient. */
template <unsigned PairOrder>
__device__ inline HighOrderPairGradientTerm make_high_order_pair_gradient_term(
    const HighOrderPairGradientGeometry<PairOrder>& geometry, unsigned subset) {
  HighOrderPairGradientTerm term{};
  if constexpr (PairOrder == 0) {
    // The scalar pair has one unit term and no center derivative. Keeping it
    // out of the combinatorial loops also avoids zero-trip unsigned-loop
    // diagnostics in CUDA's template instantiation.
    term.coefficient = 1.0;
  } else {
    for (unsigned quantum = 0; quantum < PairOrder; ++quantum) {
      if ((subset & (1U << quantum)) != 0) {
        term.derivative_state += fourth_order_derivative_state(geometry.axes[quantum]);
      }
    }
    add_high_order_wick_matching_term(term, geometry, subset, 0U, 0U);
    for (unsigned first_quantum = 0; first_quantum < PairOrder; ++first_quantum) {
      for (unsigned second_quantum = first_quantum + 1; second_quantum < PairOrder;
           ++second_quantum) {
        if (geometry.axes[first_quantum] != geometry.axes[second_quantum]) {
          continue;
        }
        const unsigned first_pair = (1U << first_quantum) | (1U << second_quantum);
        add_high_order_wick_matching_term(term, geometry, subset, first_pair, 1U);
        for (unsigned third_quantum = 0; third_quantum < PairOrder; ++third_quantum) {
          for (unsigned fourth_quantum = third_quantum + 1; fourth_quantum < PairOrder;
               ++fourth_quantum) {
            const unsigned second_pair = (1U << third_quantum) | (1U << fourth_quantum);
            if ((first_pair & second_pair) != 0 || first_pair >= second_pair ||
                geometry.axes[third_quantum] != geometry.axes[fourth_quantum]) {
              continue;
            }
            add_high_order_wick_matching_term(term, geometry, subset, first_pair | second_pair, 2U);
            if constexpr (PairOrder == 6) {
              for (unsigned fifth_quantum = 0; fifth_quantum < PairOrder; ++fifth_quantum) {
                for (unsigned sixth_quantum = fifth_quantum + 1; sixth_quantum < PairOrder;
                     ++sixth_quantum) {
                  const unsigned third_pair = (1U << fifth_quantum) | (1U << sixth_quantum);
                  if (((first_pair | second_pair) & third_pair) != 0 || second_pair >= third_pair ||
                      geometry.axes[fifth_quantum] != geometry.axes[sixth_quantum]) {
                    continue;
                  }
                  add_high_order_wick_matching_term(term, geometry, subset,
                                                    first_pair | second_pair | third_pair, 3U);
                }
              }
            }
          }
        }
      }
    }
  }
  return term;
}

/** Evaluate all-center derivatives of one canonical order-four to-six primitive. */
template <unsigned FirstPairOrder, unsigned SecondPairOrder>
__device__ inline void primitive_eri_order456_gradient(
    double alpha, const Vec3<double>& first, const Angular& angular_first, double beta,
    const Vec3<double>& second, const Angular& angular_second, double gamma,
    const Vec3<double>& third, const Angular& angular_third, double delta,
    const Vec3<double>& fourth, const Angular& angular_fourth, double (&gradient)[4][3]) {
  constexpr unsigned AngularOrder = FirstPairOrder + SecondPairOrder;
  constexpr unsigned CoulombOrder = AngularOrder + 1;
  static_assert(AngularOrder == 4 || AngularOrder == 5 || AngularOrder == 6);
  static_assert(FirstPairOrder >= SecondPairOrder);
  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<double> product_p = product_center(alpha, first, beta, second);
  const Vec3<double> product_q = product_center(gamma, third, delta, fourth);
  const Vec3<double> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };
  const HighOrderPairGradientGeometry<FirstPairOrder> first_geometry =
      make_high_order_pair_gradient_geometry<FirstPairOrder>(alpha, first, angular_first, beta,
                                                             second, angular_second);
  const HighOrderPairGradientGeometry<SecondPairOrder> second_geometry =
      make_high_order_pair_gradient_geometry<SecondPairOrder>(gamma, third, angular_third, delta,
                                                              fourth, angular_fourth);
  double boys[AngularOrder + 2];
  boys_values<AngularOrder + 1>(rho * distance_squared(product_p, product_q), boys);
  const HighOrderCoulombWorkspace<CoulombOrder> coulomb_workspace =
      make_high_order_coulomb_workspace<CoulombOrder>(rho, product_difference);
  const double first_product_scale = alpha / p;
  const double second_product_scale = beta / p;
  const double third_product_scale = -gamma / q;
  double value = 0.0;
  double value_gradient[3][3]{};
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;
  // Canonical pair ordering keeps the second expansion small (at most eight
  // terms through total order six). Materialize those terms once; generate
  // each larger first-pair term immediately before consuming it so its
  // coefficient and gradient do not create another full local array.
  HighOrderPairGradientTerm second_items[SecondTermCount];
  for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
    second_items[second_term] = make_high_order_pair_gradient_term(second_geometry, second_term);
  }
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    const HighOrderPairGradientTerm first_item =
        make_high_order_pair_gradient_term(first_geometry, first_term);
    for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
      const HighOrderPairGradientTerm& second_item = second_items[second_term];
      const double sign =
          (fourth_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      const unsigned derivative_state = first_item.derivative_state + second_item.derivative_state;
      const double coulomb =
          high_order_coulomb<CoulombOrder>(derivative_state, rho, coulomb_workspace, boys);
      const double coefficient = sign * first_item.coefficient * second_item.coefficient;
      value += coefficient * coulomb;
      for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
        const double first_coefficient_gradient =
            sign * first_item.first_center[coordinate] * second_item.coefficient;
        const double second_coefficient_gradient =
            sign * first_item.coefficient * second_item.first_center[coordinate];
        const double scaled_coulomb_derivative =
            coefficient * high_order_coulomb<CoulombOrder>(
                              derivative_state + fourth_order_derivative_state(coordinate), rho,
                              coulomb_workspace, boys);
        value_gradient[0][coordinate] +=
            first_coefficient_gradient * coulomb + first_product_scale * scaled_coulomb_derivative;
        value_gradient[1][coordinate] += -first_coefficient_gradient * coulomb +
                                         second_product_scale * scaled_coulomb_derivative;
        value_gradient[2][coordinate] +=
            second_coefficient_gradient * coulomb + third_product_scale * scaled_coulomb_derivative;
      }
    }
  }

  const double pair_decay =
      exp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  const double prefactor = 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay;
  for (unsigned center = 0; center < 3; ++center) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      double decay_derivative = 0.0;
      if (center < 2) {
        const double difference = vec_axis(first, coordinate) - vec_axis(second, coordinate);
        decay_derivative = (center == 0 ? -2.0 * mu : 2.0 * mu) * difference;
      } else {
        const double difference = vec_axis(third, coordinate) - vec_axis(fourth, coordinate);
        decay_derivative = -2.0 * nu * difference;
      }
      gradient[center][coordinate] =
          prefactor * (value_gradient[center][coordinate] + value * decay_derivative);
    }
  }
  for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
    gradient[3][coordinate] =
        -gradient[0][coordinate] - gradient[1][coordinate] - gradient[2][coordinate];
  }
}

/** Canonicalize and contract all-center gradients for total angular order 4. */
__device__ inline CartesianQuartetGradient contracted_eri_cartesian_source_order4_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  const Angular angular[4] = {
      direct_ao_angular(batch, slots[0].ao),
      direct_ao_angular(batch, slots[1].ao),
      direct_ao_angular(batch, slots[2].ao),
      direct_ao_angular(batch, slots[3].ao),
  };
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] + batch.shell_angular[slots[1].shell];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient * batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] * batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          if (first_pair_order == 4) {
            primitive_eri_order456_gradient<4, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else if (first_pair_order == 3) {
            primitive_eri_order456_gradient<3, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else {
            primitive_eri_order456_gradient<2, 2>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          }
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

/** Canonicalize and contract all-center gradients for total angular order 5. */
__device__ inline CartesianQuartetGradient contracted_eri_cartesian_source_order5_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  const Angular angular[4] = {
      direct_ao_angular(batch, slots[0].ao),
      direct_ao_angular(batch, slots[1].ao),
      direct_ao_angular(batch, slots[2].ao),
      direct_ao_angular(batch, slots[3].ao),
  };
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] + batch.shell_angular[slots[1].shell];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient * batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] * batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          if (first_pair_order == 5) {
            primitive_eri_order456_gradient<5, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else if (first_pair_order == 4) {
            primitive_eri_order456_gradient<4, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else {
            primitive_eri_order456_gradient<3, 2>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          }
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

/** Canonicalize and contract all-center gradients for total angular order 6. */
__device__ inline CartesianQuartetGradient contracted_eri_cartesian_source_order6_gradient(
    const DeviceBatch& batch, std::int32_t system, std::int32_t i, std::int32_t j, std::int32_t k,
    std::int32_t l) {
  struct SourceSlot {
    std::int64_t ao;
    std::int32_t shell;
    unsigned original;
  };
  const std::int64_t base = static_cast<std::int64_t>(system) * batch.direct_nbf;
  SourceSlot slots[4] = {
      {base + i, batch.direct_ao_shells[base + i], 0},
      {base + j, batch.direct_ao_shells[base + j], 1},
      {base + k, batch.direct_ao_shells[base + k], 2},
      {base + l, batch.direct_ao_shells[base + l], 3},
  };
  if (batch.shell_angular[slots[0].shell] < batch.shell_angular[slots[1].shell]) {
    const SourceSlot swap = slots[0];
    slots[0] = slots[1];
    slots[1] = swap;
  }
  if (batch.shell_angular[slots[2].shell] < batch.shell_angular[slots[3].shell]) {
    const SourceSlot swap = slots[2];
    slots[2] = slots[3];
    slots[3] = swap;
  }
  const unsigned first_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[0].shell], batch.shell_angular[slots[1].shell]);
  const unsigned second_pair_class = direct_shell_pair_class_cuda(
      batch.shell_angular[slots[2].shell], batch.shell_angular[slots[3].shell]);
  if (first_pair_class < second_pair_class) {
    const SourceSlot first_swap = slots[0];
    slots[0] = slots[2];
    slots[2] = first_swap;
    const SourceSlot second_swap = slots[1];
    slots[1] = slots[3];
    slots[3] = second_swap;
  }

  const Vec3<double> positions[4] = {
      atom_position<double>(batch, batch.shell_atoms[slots[0].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[1].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[2].shell], -1),
      atom_position<double>(batch, batch.shell_atoms[slots[3].shell], -1),
  };
  const Angular angular[4] = {
      direct_ao_angular(batch, slots[0].ao),
      direct_ao_angular(batch, slots[1].ao),
      direct_ao_angular(batch, slots[2].ao),
      direct_ao_angular(batch, slots[3].ao),
  };
  const double angular_coefficient =
      batch.direct_ao_coefficients[slots[0].ao] * batch.direct_ao_coefficients[slots[1].ao] *
      batch.direct_ao_coefficients[slots[2].ao] * batch.direct_ao_coefficients[slots[3].ao];
  const unsigned first_pair_order =
      batch.shell_angular[slots[0].shell] + batch.shell_angular[slots[1].shell];
  CartesianQuartetGradient result{};
  for (std::int64_t a = batch.shell_primitive_offsets[slots[0].shell];
       a < batch.shell_primitive_offsets[slots[0].shell + 1]; ++a) {
    for (std::int64_t b = batch.shell_primitive_offsets[slots[1].shell];
         b < batch.shell_primitive_offsets[slots[1].shell + 1]; ++b) {
      for (std::int64_t c = batch.shell_primitive_offsets[slots[2].shell];
           c < batch.shell_primitive_offsets[slots[2].shell + 1]; ++c) {
        for (std::int64_t d = batch.shell_primitive_offsets[slots[3].shell];
             d < batch.shell_primitive_offsets[slots[3].shell + 1]; ++d) {
          const double weight = angular_coefficient * batch.primitive_coefficients[a] *
                                batch.primitive_coefficients[b] * batch.primitive_coefficients[c] *
                                batch.primitive_coefficients[d];
          double primitive_gradient[4][3];
          if (first_pair_order == 6) {
            primitive_eri_order456_gradient<6, 0>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else if (first_pair_order == 5) {
            primitive_eri_order456_gradient<5, 1>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else if (first_pair_order == 4) {
            primitive_eri_order456_gradient<4, 2>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          } else {
            primitive_eri_order456_gradient<3, 3>(
                batch.primitive_exponents[a], positions[0], angular[0],
                batch.primitive_exponents[b], positions[1], angular[1],
                batch.primitive_exponents[c], positions[2], angular[2],
                batch.primitive_exponents[d], positions[3], angular[3], primitive_gradient);
          }
          for (unsigned center = 0; center < 4; ++center) {
            for (unsigned coordinate = 0; coordinate < 3; ++coordinate) {
              result.center[slots[center].original][coordinate] +=
                  weight * primitive_gradient[center][coordinate];
            }
          }
        }
      }
    }
  }
  return result;
}

}  // namespace vibeqc::scf::cuda_execution
"""

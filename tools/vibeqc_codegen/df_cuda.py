"""Bounded CUDA lowering of raw DF values through one- to five-root Rys rules.

The operator/signature records own root counts and auxiliary roles. Scalar
Gaussian moment DAGs are lowered by the common expression emitter; audited
Rys tables come from the existing code generator. This module owns neither
metric factorization nor RI-J/K contractions.
"""

from __future__ import annotations

from itertools import product

from .cuda import CudaEmitter
from .df_values import build_df_axis_moment, build_df_value_ir
from .ir_serialization import integral_to_payload
from .rys import (
    emit_rys2_roots_cuda,
    emit_rys3_roots_cuda,
    emit_rys4_roots_cuda,
    emit_rys5_roots_cuda,
)


def df_program_inventory() -> dict:
    """Expose every supported operator signature and its exact quadrature order."""
    programs = []
    for family, count in (("coulomb_metric", 2), ("three_center_eri", 3)):
        for angular in product(range(4), repeat=count):
            nroots = sum(angular) // 2 + 1
            integral = build_df_value_ir(family, angular, recurrence=f"rys{nroots}")
            programs.append(integral_to_payload(integral))
    return {
        "schema": "vibeqc.df_values",
        "version": 1,
        "precision": "fp64",
        "programs": programs,
    }


def emit_df_axis_cuda() -> str:
    """Lower the complete bounded one-axis moment family with structural CSE.

    One noinline helper serves all roots and coordinate axes. Pruning each
    branch before emission avoids a runtime three-dimensional recurrence
    workspace and prevents unrelated moments from extending register liveness.
    """
    lines = [
        "__device__ __noinline__ double axis_moment(unsigned a, unsigned b, unsigned c,",
        "    double mean_0, double mean_1, double mean_2,",
        "    double variance_x, double covariance_xy, double variance_y) {",
        "  switch (a * 16U + b * 4U + c) {",
    ]
    for a, b, c in product(range(4), repeat=3):
        graph, root = build_df_axis_moment(a, b, c)
        emitter = CudaEmitter(graph, {})
        emitter.emit((root,))
        lines.extend(
            [
                f"    case {a * 16 + b * 4 + c}U: {{",
                *emitter.lines,
                f"      return {emitter.reference(root)};",
                "    }",
            ]
        )
    lines.extend(
        [
            "  }",
            '  return nan("");  // Unsupported angular input is never silently truncated.',
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def emit_df_values_cuda() -> str:
    """Emit one native/fixture header evaluating unnormalized primitive M and A."""
    # Constructing the inventory validates every mathematical root count before
    # writing executable code, including the otherwise easily omitted ss/sp end.
    df_program_inventory()
    tables = "\n".join(
        emitter(symbol_prefix=f"df_rys{n}")
        for n, emitter in enumerate(
            (
                emit_rys2_roots_cuda,
                emit_rys3_roots_cuda,
                emit_rys4_roots_cuda,
                emit_rys5_roots_cuda,
            ),
            2,
        )
    )
    prefix = r"""// Generated raw DF Coulomb values; regenerate with tools/generate_df_kernels.py.
#ifndef VIBEQC_GENERATED_DF_VALUES_CUH
#define VIBEQC_GENERATED_DF_VALUES_CUH
#include <cuda_runtime.h>
#include <cmath>

namespace vibeqc::scf::generated_df {
struct Vec3 { double x, y, z; };
struct Angular { unsigned x, y, z; };

/** Single-root rule integrates the ss/sp polynomial moments exactly. */
__device__ __forceinline__ void df_rys1_roots(double argument, double* rw) {
  double f0, f1;
  if (argument < 0.5) {
    // Direct series retains F1 accuracy where upward recurrence cancels.
    double term = 1.0;
    f0 = 1.0; f1 = 1.0 / 3.0;
    for (unsigned k = 1; k < 24; ++k) {
      term *= -argument / k;
      f0 += term / (2 * k + 1);
      f1 += term / (2 * k + 3);
    }
  } else {
    f0 = 0.88622692545275801365 / sqrt(argument) * erf(sqrt(argument));
    f1 = (f0 - exp(-argument)) / (2.0 * argument);
  }
  rw[0] = f1 / f0;
  rw[1] = f0;
}
"""
    suffix = r"""
/** Contract three scalar Gaussian moments for each exact Rys quadrature root. */
template <unsigned Roots>
__device__ __noinline__ double value(
    double p, double q, const Vec3& pa, const Vec3& pb, const Vec3& difference,
    const Angular& a, const Angular& b, const Angular& c) {
  static_assert(Roots >= 1 && Roots <= 5);
  const double rho = p * q / (p + q);
  const double argument = rho * (difference.x * difference.x +
      difference.y * difference.y + difference.z * difference.z);
  double rw[2 * Roots];
  if constexpr (Roots == 1) df_rys1_roots(argument, rw);
  if constexpr (Roots == 2) df_rys2_roots(argument, rw, 1U);
  if constexpr (Roots == 3) df_rys3_roots(argument, rw, 1U);
  if constexpr (Roots == 4) df_rys4_roots(argument, rw, 1U);
  if constexpr (Roots == 5) df_rys5_roots(argument, rw, 1U);
  double result = 0.0;
#pragma unroll 1
  for (unsigned root = 0; root < Roots; ++root) {
    const double root_over_sum = rw[2 * root] / (p + q);
    const double first_scale = root_over_sum * q;
    const double second_scale = root_over_sum * p;
    const double xx = 0.5 / p * (1.0 - first_scale);
    const double xy = 0.5 * root_over_sum;
    const double yy = 0.5 / q * (1.0 - second_scale);
    const double x = axis_moment(a.x, b.x, c.x,
        pa.x - difference.x * first_scale, pb.x - difference.x * first_scale,
        difference.x * second_scale, xx, xy, yy);
    const double y = axis_moment(a.y, b.y, c.y,
        pa.y - difference.y * first_scale, pb.y - difference.y * first_scale,
        difference.y * second_scale, xx, xy, yy);
    const double z = axis_moment(a.z, b.z, c.z,
        pa.z - difference.z * first_scale, pb.z - difference.z * first_scale,
        difference.z * second_scale, xx, xy, yy);
    result += rw[2 * root + 1] * x * y * z;
  }
  return 34.986836655249725694 / (p * q * sqrt(p + q)) * result;
}

__device__ __forceinline__ unsigned order(const Angular& a) { return a.x + a.y + a.z; }

/** (P|Q): two positive-exponent auxiliary charge distributions, with no overlap decay. */
__device__ __forceinline__ double metric(
    double alpha, const Vec3& A, const Angular& a,
    double gamma, const Vec3& C, const Angular& c) {
  const Vec3 zero{0.0, 0.0, 0.0};
  const Vec3 difference{A.x - C.x, A.y - C.y, A.z - C.z};
  const Angular b{0, 0, 0};
  const unsigned total = order(a) + order(c);
  if (order(a) > 3 || order(c) > 3) return nan("");
  if (total <= 1) return value<1>(alpha, gamma, zero, zero, difference, a, b, c);
  if (total <= 3) return value<2>(alpha, gamma, zero, zero, difference, a, b, c);
  if (total <= 5) return value<3>(alpha, gamma, zero, zero, difference, a, b, c);
  return value<4>(alpha, gamma, zero, zero, difference, a, b, c);
}

/** (mu nu|P): orbital-product Gaussian decay is applied once, outside the Rys moment. */
__device__ __forceinline__ double three_center(
    double alpha, const Vec3& A, const Angular& a,
    double beta, const Vec3& B, const Angular& b,
    double gamma, const Vec3& C, const Angular& c) {
  const double p = alpha + beta;
  const Vec3 ab{A.x - B.x, A.y - B.y, A.z - B.z};
  const Vec3 pa{-beta / p * ab.x, -beta / p * ab.y, -beta / p * ab.z};
  const Vec3 pb{alpha / p * ab.x, alpha / p * ab.y, alpha / p * ab.z};
  const Vec3 difference{A.x - C.x + pa.x, A.y - C.y + pa.y, A.z - C.z + pa.z};
  const double decay = exp(-alpha * beta / p * (ab.x * ab.x + ab.y * ab.y + ab.z * ab.z));
  const unsigned total = order(a) + order(b) + order(c);
  if (order(a) > 3 || order(b) > 3 || order(c) > 3) return nan("");
  double result;
  if (total <= 1) result = value<1>(p, gamma, pa, pb, difference, a, b, c);
  else if (total <= 3) result = value<2>(p, gamma, pa, pb, difference, a, b, c);
  else if (total <= 5) result = value<3>(p, gamma, pa, pb, difference, a, b, c);
  else if (total <= 7) result = value<4>(p, gamma, pa, pb, difference, a, b, c);
  else result = value<5>(p, gamma, pa, pb, difference, a, b, c);
  return decay * result;
}
}  // namespace vibeqc::scf::generated_df
#endif
"""
    return prefix + tables + emit_df_axis_cuda() + suffix

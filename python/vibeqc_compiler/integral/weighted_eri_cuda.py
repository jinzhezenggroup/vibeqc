"""CUDA scalar lowering of bounded external-weight ERI derivative programs.

The callable consumes shared primitive geometry and already normalized external
weights. Runtime contraction loops, native task queues, density adapters, and
resident-pair policies belong to the caller, so replacing a handwritten
primitive expression does not discard its validated scheduling structure.
"""

from __future__ import annotations

import typing

from .cuda import CudaEmitter
from .expr import AlgebraForm, AlgebraFusion, AlgebraOrdering, RematerializationPolicy
from .weighted_eri import (
    WeightedEriKernel,
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)


def emit_weighted_eri_function(
    kernel: WeightedEriKernel,
    name: str,
    *,
    inline_single_use: typing.Any = False,
    backend: typing.Any = "cuda",
    packed_weights: typing.Any = False,
    include_value: bool = True,
    gradient_centers: tuple[int, ...] = (0, 1, 2, 3),
    result_type: str = "Gradient",
    ordering: AlgebraOrdering = AlgebraOrdering.TOPOLOGICAL,
    fusion: AlgebraFusion = AlgebraFusion.SEPARATE,
) -> str:
    """Emit one selected weighted result with shared scalar CSE.

    A component subset yields its additive contribution only. Every required
    geometry field is mapped by meaning rather than a density/task ABI; the
    same helper therefore serves HF adapters and arbitrary external weights.
    Packed weights follow component_indices order, bounding a sparse through-f
    helper's input to its selected subset rather than a full shell weight array.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("weighted scalar emission supports cpu or cuda")
    if not name.isascii() or not name.isidentifier():
        raise ValueError("CUDA helper name must be an ASCII identifier")
    if not result_type.isascii() or not result_type.isidentifier():
        raise ValueError("weighted result type must be an ASCII identifier")
    if len(set(gradient_centers)) != len(gradient_centers) or any(
        type(center) is not int or not 0 <= center < 4 for center in gradient_centers
    ):
        raise ValueError("weighted gradient centers must be unique center indices")
    source_roots = []
    assignments = []
    if include_value:
        source_roots.append(kernel.value)
        assignments.append("value")
    for output_center, center in enumerate(gradient_centers):
        for axis, value in enumerate(kernel.gradients[center]):
            source_roots.append(value)
            assignments.append(f"center[{output_center}][{axis}]")
    if not source_roots:
        raise ValueError("weighted scalar emission requires at least one output root")
    graph, roots = kernel.graph.apply_algebra_form(
        tuple(source_roots), AlgebraForm.FACTORED_NARY
    )
    policy = RematerializationPolicy(
        name="weighted_single_use", inline_single_use=inline_single_use
    )
    plan = graph.materialization_plan(roots, policy, ordering, fusion)
    variables = {
        name: f"geometry.{name}"
        for name in ("inverse_two_p", "inverse_two_q", "rho", "prefactor")
    }
    for axis, label in enumerate("xyz"):
        variables[f"difference_{label}"] = f"geometry.difference[{axis}]"
        for slot, prefix in enumerate(("pa", "pb", "qc", "qd")):
            variables[f"{prefix}_{label}"] = f"geometry.shifts[{slot}][{axis}]"
        for center, prefix in enumerate(("first", "second", "third", "fourth")):
            variables[f"decay_{prefix}_{label}"] = f"geometry.decay[{center}][{axis}]"
    for center, prefix in enumerate(("first", "second", "third", "fourth")):
        variables[f"{prefix}_product_scale"] = f"geometry.product_scales[{center}]"
    for order in range(kernel.integral.maximum_coulomb_order + 1):
        variables[f"boys_{order}"] = f"geometry.boys[{order}]"
    for packed, index in enumerate(kernel.component_indices):
        offset = packed if packed_weights else index
        variables[f"component_weight_{index}"] = f"component_weights[{offset}]"
    emitter = CudaEmitter(graph, variables, plan)
    emitter.emit(roots)
    output_kind = "derivatives" if include_value else "gradient-only"
    lines = [
        f"/** Unscreened {kernel.spec.name} external-weight {output_kind}; {len(kernel.component_indices)} components. */",
        f"{'__device__ __forceinline__' if backend == 'cuda' else 'inline'} {result_type} {name}(const Geometry& geometry, const double* component_weights) {{",
        *emitter.lines,
        f"  {result_type} result{{}};",
    ]
    if kernel.integral.operator.range_separated:
        radial = kernel.integral.operator.coulomb_kernel
        # The geometry-factored helper consumes modified moments supplied by
        # its caller. Retain exact operator identity even when the arithmetic
        # DAG is shared with full Coulomb; legacy native streams reject this IR.
        lines.insert(
            0,
            f"/** Requires {radial.family.value} moments; omega={radial.omega.hex()} inverse bohr, held fixed. */",
        )
    for assignment, value in zip(assignments, roots, strict=True):
        lines.append(f"  result.{assignment} = {emitter.reference(value)};")
    lines.extend(["  return result;", "}"])
    return "\n".join(lines) + "\n"


def emit_weighted_eri_header(
    functions: tuple[tuple[WeightedEriKernel, str], ...],
    *,
    inline_single_use: typing.Any = False,
    backend: typing.Any = "cuda",
    packed_weights: typing.Any = False,
) -> str:
    """Wrap bounded helpers in one shared geometry/result interface.

    Geometry includes fields through the public f/f/f/f derivative order.
    An inlined helper reads only its reachable fields; callers need initialize
    those fields only. Coefficients and Cartesian normalization are multiplied
    into weights/prefactor once by the existing primitive contraction layer.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("weighted scalar emission supports cpu or cuda")
    names = [name for _, name in functions]
    if len(names) != len(set(names)):
        raise ValueError("generated weighted helper names must be unique")
    prefix = r"""// Generated by tools/generate_weighted_eri_kernels.py; do not edit.
#ifndef VIBEQC_GENERATED_WEIGHTED_ERI_CUH
#define VIBEQC_GENERATED_WEIGHTED_ERI_CUH
#include <cuda_runtime.h>
#include <cmath>
namespace vibeqc::scf::generated_weighted_eri {
/** Primitive pair data; product scales are positive exponent fractions. */
struct Geometry {
  double inverse_two_p, inverse_two_q, rho;
  double difference[3], shifts[4][3], product_scales[4], decay[4][3];
  double prefactor;
  double boys[14];
};
/** Nuclear derivatives in original shell-center order, before atom accumulation. */
struct Gradient { double value; double center[4][3]; };
"""
    if backend == "cpu":
        prefix = prefix.replace("#include <cuda_runtime.h>\n", "")
    body = "\n".join(
        emit_weighted_eri_function(
            kernel,
            name,
            inline_single_use=inline_single_use,
            backend=backend,
            packed_weights=packed_weights,
        )
        for kernel, name in functions
    )
    return (
        prefix + body + "}  // namespace vibeqc::scf::generated_weighted_eri\n#endif\n"
    )


def emit_psss_weighted_header(*, inline_single_use: typing.Any = False) -> str:
    """Generate the first native migration candidate without replacing queues."""
    return emit_weighted_eri_header(
        ((build_weighted_eri_kernel(build_weighted_eri_ir((1, 0, 0, 0))), "psss"),),
        inline_single_use=inline_single_use,
    )


def emit_low_order_weighted_header(*, inline_single_use: typing.Any = False) -> str:
    """Generate native low-order helpers with force-only Direct-HF specializations."""
    ssss = build_weighted_eri_kernel(build_weighted_eri_ir((0, 0, 0, 0)))
    psss = build_weighted_eri_kernel(build_weighted_eri_ir((1, 0, 0, 0)))
    full = emit_weighted_eri_header(
        ((psss, "psss"),),
        inline_single_use=inline_single_use,
    )
    marker = "}  // namespace vibeqc::scf::generated_weighted_eri\n#endif\n"
    if not full.endswith(marker):
        raise ValueError("weighted ERI header footer changed unexpectedly")
    specialized_result = (
        "/** Independent-center force result; recovered center is reconstructed by the caller. */\n"
        "struct IndependentGradient { double center[3][3]; };\n"
    )
    psss_force = emit_weighted_eri_function(
        psss,
        "psss_force",
        inline_single_use=inline_single_use,
        include_value=False,
        gradient_centers=(0, 1, 2),
        result_type="IndependentGradient",
        ordering=AlgebraOrdering.PRESSURE_AWARE,
    )
    ssss_force = emit_weighted_eri_function(
        ssss,
        "ssss_force",
        inline_single_use=inline_single_use,
        include_value=False,
        gradient_centers=(0, 1, 2),
        result_type="IndependentGradient",
    )
    return full[: -len(marker)] + specialized_result + psss_force + ssss_force + marker

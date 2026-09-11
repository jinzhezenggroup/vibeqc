"""CUDA scalar lowering of bounded external-weight ERI derivative programs.

The callable consumes shared primitive geometry and already normalized external
weights. Runtime contraction loops, native task queues, density adapters, and
resident-pair policies belong to the caller, so replacing a handwritten
primitive expression does not discard its validated scheduling structure.
"""

from __future__ import annotations

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
    inline_single_use=False,
    backend="cuda",
    packed_weights=False,
) -> str:
    """Emit one complete center-gradient result with shared scalar CSE.

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
    roots = (kernel.value, *(value for row in kernel.gradients for value in row))
    graph, roots = kernel.graph.apply_algebra_form(roots, AlgebraForm.FACTORED_NARY)
    policy = RematerializationPolicy(
        name="weighted_single_use", inline_single_use=inline_single_use
    )
    plan = graph.materialization_plan(
        roots, policy, AlgebraOrdering.TOPOLOGICAL, AlgebraFusion.SEPARATE
    )
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
    lines = [
        f"/** Unscreened {kernel.spec.name} external-weight derivatives; {len(kernel.component_indices)} components. */",
        f"{'__device__ __forceinline__' if backend == 'cuda' else 'inline'} Gradient {name}(const Geometry& geometry, const double* component_weights) {{",
        *emitter.lines,
        "  Gradient result{};",
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
    lines.append(f"  result.value = {emitter.reference(roots[0])};")
    for i, value in enumerate(roots[1:]):
        lines.append(
            f"  result.center[{i // 3}][{i % 3}] = {emitter.reference(value)};"
        )
    lines.extend(["  return result;", "}"])
    return "\n".join(lines) + "\n"


def emit_weighted_eri_header(
    functions: tuple[tuple[WeightedEriKernel, str], ...],
    *,
    inline_single_use=False,
    backend="cuda",
    packed_weights=False,
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


def emit_psss_weighted_header(*, inline_single_use=False) -> str:
    """Generate the first native migration candidate without replacing queues."""
    return emit_weighted_eri_header(
        ((build_weighted_eri_kernel(build_weighted_eri_ir((1, 0, 0, 0))), "psss"),),
        inline_single_use=inline_single_use,
    )

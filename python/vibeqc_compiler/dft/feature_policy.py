"""Shared compiler-owned density bilinears for CPU and CUDA D/C consumers."""

import typing

from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter


def emit_feature_policy(*, device: typing.Any = False) -> typing.Any:
    """Emit rho/gradient/tau and sigma with identical occupation conventions.

    D supplies AO jets and their density products; C supplies weighted orbital
    jets in both slots. Runtime controls traversal and requested-output masks.
    The host XC consumer uses the same roots as bounded dense/local CUDA.
    """
    qualifier = "__device__" if device else "inline"
    lines = []
    graph = Graph()
    phi = graph.variable("phi")
    dp = tuple(graph.variable(f"derivative[{i}]") for i in range(3))
    work = tuple(graph.variable(f"work[{i}]") for i in range(4))
    roots = (
        phi * work[0],
        *(2 * d * work[0] for d in dp),
        sum((d * w / 2 for d, w in zip(dp, work[1:], strict=True)), graph.constant(0)),
    )
    lines.append(
        f"{qualifier} void add_features(double phi, const double* derivative, const double* work, double* accum, unsigned mask) {{"
    )
    # The orbital route supplies phi=Psi, derivative=grad(Psi), work=Psi jets.
    # The same compiler-owned bilinears then give the occupation-weighted
    # orbital identities, with no second native scientific formula.
    for mask, indices in ((1, (0,)), (6, (1, 2, 3)), (8, (4,))):
        emitter = ScalarCEmitter(graph, {})
        emitter.emit(tuple(roots[i] for i in indices))
        lines.append(f"  if (mask & {mask}) {{")
        lines.extend(emitter.lines)
        lines.extend(f"  accum[{i}] += {emitter.reference(roots[i])};" for i in indices)
        lines.append("  }")
    lines.append("}")
    graph = Graph()
    gradients = [[graph.variable(f"g[{s}][{k}]") for k in range(3)] for s in range(2)]
    roots = tuple(
        sum((gradients[a][k] * gradients[b][k] for k in range(3)), graph.constant(0))
        for a, b in ((0, 0), (0, 1), (1, 1))
    )
    emitter = ScalarCEmitter(graph, {})
    emitter.emit(roots)
    lines.append(f"{qualifier} void sigma(const double (&g)[2][3], double* output) {{")
    lines.extend(emitter.lines)
    lines.extend(
        f"  output[{i}] = {emitter.reference(r)};" for i, r in enumerate(roots)
    )
    lines.extend(("}", ""))
    return "\n".join(lines)

"""A conservative absolute force bound for the compiler's SSS derivative IR.

This bounded first domain has no Cartesian/spherical expansion ambiguity: each
s shell has one public component. The caller supplies the actual folded external
response weight and normalized primitive coefficients. Higher angular classes
retain strict evaluation until separately derived bounds are available.
"""

from fractions import Fraction

from .cuda import CudaEmitter
from .df_derivatives_cuda import emit_df_geometry_cuda
from .df_rys_shell import build_df_rys_sss_ir
from .expr import Graph, Node


def sss_force_bound_ir():
    """Apply the triangle inequality to all six independent derivative channels.

    For nonnegative T, F0(T)<=1 and F1(T)/F0(T)<=1/3. Replacing each signed
    monomial coefficient and geometry input by its magnitude gives an upper
    bound on |dA|+|dB|. Translation gives dC=-(dA+dB), so their sum also bounds
    every atomic component, including coincident/shared centers. This is a
    derivative bound, not an integral or energy screening estimate.
    """
    source, outputs = build_df_rys_sss_ir()
    graph = Graph()
    cloned = {}
    for identifier in source.topological_order(outputs):
        node = source.nodes[identifier]
        if node.operation == "constant":
            cloned[identifier] = graph.constant(abs(node.payload))
        elif node.operation == "variable":
            if node.payload == "root":
                cloned[identifier] = graph.constant(Fraction(1, 3))
            elif node.payload == "root_weight":
                cloned[identifier] = graph.constant(1)
            else:
                cloned[identifier] = graph.variable(node.payload)
        else:
            if node.operation not in ("add", "multiply"):
                raise ValueError(
                    "SSS bound requires an audited polynomial derivative graph"
                )
            cloned[identifier] = graph._intern(
                Node(
                    node.operation,
                    tuple(cloned[child].identifier for child in node.arguments),
                    node.payload,
                )
            )
    # A factor of two leaves a wide FP64 rounding margin. The analytical bound
    # does not certify unrelated force approximations or SCF/metric errors.
    return graph, 2 * graph.sum(cloned[value.identifier] for value in outputs)


def emit_sss_force_screening_cuda():
    """Emit the cheap Gaussian geometry and bound before Boys/root evaluation."""
    geometry = emit_df_geometry_cuda(
        "prepare_screen_geometry", moments="(void)total; (void)work;"
    ).replace("__device__", "static __device__")
    graph, output = sss_force_bound_ir()
    variables = {
        "weight": "fabs(weight)",
        "alpha": "alpha",
        "beta": "beta",
        "prefactor": "fabs(g.prefactor)",
        "sx": "g.sx",
    }
    variables.update(
        {
            f"{field}_{axis}": f"fabs(g.{field}[{axis}])"
            for field in ("pa", "pb", "dx")
            for axis in range(3)
        }
    )
    emitter = CudaEmitter(graph, variables)
    emitter.emit((output,))
    return "\n".join(
        [
            "// Generated SSS derivative bound; no Boys or Rys evaluation.",
            "#pragma once",
            '#include "generated_df_derivatives.cuh"',
            "namespace vibeqc::scf::generated_df_derivatives {",
            geometry,
            "__device__ __forceinline__ double sss_force_bound(double alpha,Vec3 A,double beta,Vec3 B,",
            "    double gamma,Vec3 C,double weight) {",
            "  Geometry g; prepare_screen_geometry(alpha,A,beta,B,gamma,C,0,g);",
            *emitter.lines,
            f"  const double bound={emitter.reference(output)};",
            "  return bound;",
            "}",
            "} // namespace vibeqc::scf::generated_df_derivatives",
            "",
        ]
    )

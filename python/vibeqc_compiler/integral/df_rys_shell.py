"""First incremental Rys derivative shell lowering: SSS only.

Mathematics comes from the existing one-axis Gaussian-moment IR. The prototype
contracts six orbital-center derivatives directly at the one quadrature node;
auxiliary translation recovery remains in the existing generated finish().
Broader classes must be added only after this first slice is GPU-qualified.
"""

from .cuda import CudaEmitter
from .df_derivatives_cuda import emit_df_geometry_cuda
from .df_values import build_df_axis_moment
from .expr import Graph, Node

RYS_SHELL_CLASSES = ((0, 0, 0),)


def shell_rys_work_model(angular):
    """Count emitted root/moment work, independent of GPU instruction counts."""
    if tuple(angular) not in RYS_SHELL_CLASSES:
        raise ValueError("Rys shell class is not generated")
    return {
        "rys_roots": 1,
        "recurrence_states": 6,
        "axis_polynomial_calls": 0,
        "specialized_prepare_axis_calls": 0,
        "cache_coefficient_values": 0,
        "component_convolution_iterations": [0],
    }


def emit_df_rys_policy_cpp():
    """Expose generated availability without importing CUDA into host policy."""
    mask = sum(1 << (16 * a + 4 * b + c) for a, b, c in RYS_SHELL_CLASSES)
    return f"""// Generated derivative-lowering capability; no automatic promotion yet.
#pragma once
#include <cstdint>
namespace vibeqc::scf::generated_df_shell {{
inline constexpr std::uint64_t rys_available_mask={mask}ULL;
inline constexpr std::uint64_t rys_qualified_mask=0;
template<unsigned A,unsigned B,unsigned C>
inline constexpr bool rys_available=(rys_available_mask & (1ULL<<(16*A+4*B+C)))!=0;
}} // namespace vibeqc::scf::generated_df_shell
"""


def build_df_rys_sss_ir():
    """Lower raised orbital Gaussians through the shared root-dependent moments.

    SSS needs only degree-one moments, so a single Gaussian quadrature node
    exactly integrates its first derivative. Normalization and public response
    folding remain outside the primitive IR. Six moments, one per independent
    center/axis, are produced; no axis polynomial or coefficient convolution is
    evaluated. The source graph is cloned with root-affine means, not rebuilt
    as a separate native recurrence.
    """
    graph = Graph()
    root, sx = graph.variable("root"), graph.variable("sx")
    factor = (
        graph.variable("weight")
        * graph.variable("prefactor")
        * graph.variable("root_weight")
    )
    outputs = []
    for center, exponent in enumerate(("alpha", "beta")):
        powers = [0, 0, 0]
        powers[center] += 1
        source, value = build_df_axis_moment(*powers, internal_derivative=True)
        for axis in range(3):
            shift = graph.variable(f"dx_{axis}") * sx * root
            replacements = {
                "mean_0": graph.variable(f"pa_{axis}") - shift,
                "mean_1": graph.variable(f"pb_{axis}") - shift,
            }
            cloned = {}
            for identifier in source.topological_order((value,)):
                node = source.nodes[identifier]
                if node.operation == "variable":
                    cloned[identifier] = replacements[node.payload]
                elif node.operation == "constant":
                    cloned[identifier] = graph.clone_constant(node)
                else:
                    cloned[identifier] = graph._intern(
                        Node(
                            node.operation,
                            tuple(cloned[child].identifier for child in node.arguments),
                            node.payload,
                        )
                    )
            # Differentiating an s Gaussian raises its power once; its lowering
            # coefficient is zero. No runtime differentiation is introduced.
            outputs.append(
                factor * 2 * graph.variable(exponent) * cloned[value.identifier]
            )
    return graph, tuple(outputs)


def emit_df_rys_shell_cuda():
    """Emit the first shell prototype against the common geometry/finish API.

    The production dispatcher instantiates this header only for available
    Rys classes. Automatic policy remains polynomial until force and endpoint
    qualification justifies a generated promotion mask.
    A dummy cache extent keeps the common native array declaration well-formed;
    prepare() never references it, so CUDA can eliminate that shared allocation.
    """
    geometry = emit_df_geometry_cuda(
        "prepare_geometry_rys",
        moments="if(work) *work={}; (void)total; generated_df_rys::roots<Roots>(rho*distance,g.f,g.f+Roots);",
    ).replace("__device__", "static __device__")
    graph, outputs = build_df_rys_sss_ir()
    variables = {
        "weight": "weight",
        "alpha": "alpha",
        "beta": "beta",
        "root": "g.f[0]",
        "root_weight": "g.f[1]",
        "prefactor": "g.prefactor",
        "sx": "g.sx",
    }
    variables.update(
        {
            f"{field}_{axis}": f"g.{field}[{axis}]"
            for field in ("pa", "pb", "dx")
            for axis in range(3)
        }
    )
    emitter = CudaEmitter(graph, variables)
    emitter.emit(outputs)
    lines = [
        r"""// First compiler-owned Rys derivative shell; generated from shared moment IR.
#ifndef VIBEQC_GENERATED_DF_RYS_SHELL_CUH
#define VIBEQC_GENERATED_DF_RYS_SHELL_CUH
#include "generated_df_rys.cuh"
#include "generated_df_rys_policy.hpp"
#include "generated_df_shell_derivatives.cuh"
namespace vibeqc::scf::generated_df_derivatives {
template<unsigned Roots>""",
        geometry,
        r"""
} // namespace vibeqc::scf::generated_df_derivatives
namespace vibeqc::scf::generated_df_shell {
template<unsigned A,unsigned B,unsigned C> struct RysShell;
template<> struct RysShell<0,0,0> : Shell<0,0,0> {
  static constexpr unsigned nroots=1;
  static constexpr unsigned axis_size=1;
  static constexpr unsigned cache_coefficient_values=0;
  static constexpr unsigned polynomial_calls=0,specialized_axis_calls=0;
  static constexpr unsigned recurrence_states_per_primitive=6;
  __device__ static unsigned convolution_work(unsigned) { return 0; }
  __device__ static void prepare(const scalar::Geometry&,double*,unsigned,unsigned) {}
  __device__ static void accumulate(unsigned,double alpha,double beta,
      const scalar::Geometry& g,const double*,double weight,double* out) {""",
    ]
    lines.extend(emitter.lines)
    lines.extend(
        f"    out[{axis}]+={emitter.reference(value)};"
        for axis, value in enumerate(outputs)
    )
    lines.extend(
        ("  }", "};", "} // namespace vibeqc::scf::generated_df_shell", "#endif", "")
    )
    return "\n".join(lines)

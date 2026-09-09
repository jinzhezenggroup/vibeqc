"""CUDA lowering of pruned, shared one-electron S/T/V scalar DAGs.

The native contraction layer owns normalized coefficients and output layouts.
This module emits a bounded s/p/d/f primitive family, with pair geometry hoisted
out of the nuclear loop and S/T lowered together for common-subexpression reuse.
"""

from functools import cache
from itertools import product

from .cuda import CudaEmitter
from .expr import Graph, Node
from .ir_serialization import integral_to_payload
from .one_electron_values import (
    build_one_electron_component_kernel,
    build_one_electron_value_ir,
)
from .shell_spec import cartesian_components


def one_electron_program_inventory():
    """Record scientific signatures separately from runtime/promotion evidence."""
    return {
        "schema": "vibeqc.one_electron_values",
        "version": 1,
        "precision": "fp64",
        "schedules": ["thread", "shell_warp"],
        "programs": [
            integral_to_payload(build_one_electron_value_ir(family, angular))
            for family in ("overlap", "kinetic", "nuclear_attraction")
            for angular in product(range(4), repeat=2)
        ],
    }


def _geometry_boundary(kernel, roots):
    """Cut hoisted geometry roots out of a copy of the executable DAG.

    Substitution before topological scheduling also removes their ancestors;
    renaming emitted temporaries alone would leave redundant exponentials and
    divisions inside every nuclear-center invocation.
    """
    target = Graph()
    replacements = {expr.identifier: name for name, expr in kernel.pair_geometry}

    @cache
    def visit(identifier):
        if identifier in replacements:
            return target.variable("pair." + replacements[identifier])
        node = kernel.graph.nodes[identifier]
        if node.operation == "variable":
            name = str(node.payload)
            if name.startswith("boys_"):
                return target.variable(f"boys[{name[5:]}]")
            return target.variable(name if name.startswith("c_") else "pair." + name)
        if node.operation == "constant":
            return target.clone_constant(node)
        # Preserve the already simplified algebra, interning shared ancestors
        # in the target graph just as the scientific builder does.
        return target._intern(
            Node(
                node.operation,
                tuple(visit(i).identifier for i in node.arguments),
                node.payload,
            )
        )

    return target, tuple(visit(root.identifier) for root in roots)


def _emit_pair_geometry():
    kernel = build_one_electron_component_kernel(
        build_one_electron_value_ir("overlap", (0, 0)), ("", "")
    )
    fields = ["alpha", "beta"] + [
        f"{center}_{axis}" for center in "ab" for axis in "xyz"
    ]
    fields += [name for name, _ in kernel.pair_geometry]
    lines = ["struct PairGeometry {", "  double " + ", ".join(fields) + ";", "};"]
    lines += [
        "__device__ __forceinline__ PairGeometry make_pair(",
        "    double alpha, double beta, double a_x, double a_y, double a_z,",
        "    double b_x, double b_y, double b_z) {",
        "  PairGeometry pair{};",
    ]
    for field in fields[:8]:
        lines.append(f"  pair.{field} = {field};")
    emitter = CudaEmitter(kernel.graph, {})
    for name, expr in kernel.pair_geometry:
        emitter.emit_assignment(expr, f"pair.{name}")
    return "\n".join(lines + emitter.lines + ["  return pair;", "}"])


def _emit_operator_helpers(attraction):
    name = "attraction" if attraction else "overlap_kinetic"
    return_type = "double" if attraction else "ST"
    arguments = "const PairGeometry& pair, unsigned component"
    if attraction:
        arguments += ", double c_x, double c_y, double c_z"
    lines = []
    for angular in product(range(4), repeat=2):
        suffix = f"{angular[0]}{angular[1]}"
        lines += [
            f"__device__ __noinline__ {return_type} {name}_{suffix}({arguments}) {{",
            "  switch (component) {",
        ]
        for index, components in enumerate(
            product(*(cartesian_components(l) for l in angular))
        ):
            graph = Graph()
            if attraction:
                kernel = build_one_electron_component_kernel(
                    build_one_electron_value_ir("nuclear_attraction", angular),
                    components,
                    graph=graph,
                )
                target, roots = _geometry_boundary(
                    kernel, (kernel.boys_argument, kernel.value)
                )
            else:
                s = build_one_electron_component_kernel(
                    build_one_electron_value_ir("overlap", angular),
                    components,
                    graph=graph,
                )
                kernel = build_one_electron_component_kernel(
                    build_one_electron_value_ir("kinetic", angular),
                    components,
                    graph=graph,
                )
                target, roots = _geometry_boundary(kernel, (s.value, kernel.value))
            emitter = CudaEmitter(target, {})
            lines.append(f"    case {index}U: {{")
            if attraction:
                emitter.emit((roots[0],))
                lines += emitter.lines
                emitter.lines.clear()
                lines += [
                    f"      double boys[{kernel.boys_count}];",
                    f"      boys_values<{kernel.boys_count - 1}>({emitter.reference(roots[0])}, boys);",
                ]
                emitter.emit((roots[1],))
                result = emitter.reference(roots[1])
            else:
                emitter.emit(roots)
                result = (
                    "{" + ", ".join(emitter.reference(root) for root in roots) + "}"
                )
            lines += emitter.lines + [f"      return {result};", "    }"]
        invalid = 'nan("")' if attraction else 'ST{nan(""), nan("")}'
        lines += ["  }", f"  return {invalid};", "}"]
    lines += [
        f"__device__ __forceinline__ {return_type} {name}(",
        "    const PairGeometry& pair, unsigned first, unsigned second"
        + (", double c_x, double c_y, double c_z) {" if attraction else ") {"),
        "  const unsigned a = first < 1 ? 0 : first < 4 ? 1 : first < 10 ? 2 : 3;",
        "  const unsigned b = second < 1 ? 0 : second < 4 ? 1 : second < 10 ? 2 : 3;",
        "  const unsigned offsets[] = {0, 1, 4, 10};",
        "  const unsigned counts[] = {1, 3, 6, 10};",
        "  const unsigned component = (first - offsets[a]) * counts[b] + second - offsets[b];",
        "  if (first >= 20 || second >= 20) return " + invalid + ";",
        "  switch (a * 4U + b) {",
    ]
    for a, b in product(range(4), repeat=2):
        args = "pair, component" + (", c_x, c_y, c_z" if attraction else "")
        lines.append(f"    case {a * 4 + b}U: return {name}_{a}{b}({args});")
    lines += ["  }", f"  return {invalid};", "}"]
    return "\n".join(lines)


def emit_one_electron_values_cuda():
    """Emit primitive S/T and signed unit-charge V; charge is applied by consumer."""
    prefix = r"""// Generated by tools/generate_one_electron_kernels.py; do not edit.
#ifndef VIBEQC_GENERATED_ONE_ELECTRON_VALUES_CUH
#define VIBEQC_GENERATED_ONE_ELECTRON_VALUES_CUH
#include <cuda_runtime.h>
#include <cmath>
namespace vibeqc::scf::generated_one_electron {
struct ST { double overlap, kinetic; };

/** Positive-term series plus downward recurrence avoids cancellation at small T.
 * F_m(T) = exp(-T) sum_k (2T)^k / [(2m+1)(2m+3)...(2m+2k+1)].
 * Above 20, upward recurrence through order six is well conditioned. */
template<unsigned Order>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  static_assert(Order <= 6);
  const double decay = exp(-argument);
  if (argument < 20.0) {
    double term = 1.0 / (2 * Order + 1);
    double sum = term;
    for (unsigned k = 1; k < 160; ++k) {
      term *= 2.0 * argument / (2 * Order + 2 * k + 1);
      sum += term;
      if (term < 1.0e-17 * sum) break;
    }
    values[Order] = decay * sum;
    for (unsigned n = Order; n > 0; --n)
      values[n - 1] = (2.0 * argument * values[n] + decay) / (2 * n - 1);
  } else {
    values[0] = 0.88622692545275801365 * erf(sqrt(argument)) / sqrt(argument);
    for (unsigned n = 1; n <= Order; ++n)
      values[n] = ((2 * n - 1) * values[n - 1] - decay) / (2.0 * argument);
  }
}
"""
    index = [
        "/** Public Cartesian order, shared by all native basis expansion terms. */",
        "__device__ __forceinline__ unsigned component_index(unsigned x, unsigned y, unsigned z) {",
        "  switch (x * 16U + y * 4U + z) {",
    ]
    for i, component in enumerate(c for l in range(4) for c in cartesian_components(l)):
        x, y, z = (component.count(axis) for axis in "xyz")
        index.append(f"    case {x * 16 + y * 4 + z}U: return {i}U;")
    index += ["  }", "  return 20U;", "}"]
    return "\n".join(
        [
            prefix,
            _emit_pair_geometry(),
            "\n".join(index),
            _emit_operator_helpers(False),
            _emit_operator_helpers(True),
            "}  // namespace vibeqc::scf::generated_one_electron",
            "#endif",
            "",
        ]
    )

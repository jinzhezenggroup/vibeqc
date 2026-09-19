"""CPU primitive first derivatives from the existing shared scientific graphs.

This finite source generator emits only requested Cartesian component kernels.
Runtime records, normalization, allocation, contraction and atom scatter belong
to consumers. It imports neither the public runtime nor any independent oracle.
"""

from functools import lru_cache

from .expr import Graph
from .one_electron_derivatives import (
    build_one_electron_derivative_ir,
    build_one_electron_derivative_kernel,
)
from .scalar_c import ScalarCEmitter
from .weighted_eri import build_weighted_eri_ir, build_weighted_eri_kernel
from .weighted_eri_native import emit_weighted_eri_primitive_header


def _scalar_function(name, graph, roots, variables, prefix=()):
    emitter = ScalarCEmitter(graph, variables)
    emitter.emit(roots)
    return "\n".join(
        [
            f"static bool {name}(const double* e, const double* c, double* out) {{",
            *prefix,
            *emitter.lines,
            *(
                f"  out[{i}] = {emitter.reference(root)};"
                for i, root in enumerate(roots)
            ),
            "  return true;",
            "}",
        ]
    )


@lru_cache(maxsize=4)
def emit_first_derivative_cpu(requests):
    """Emit ordered ``(operator, components)`` kernels and a checked dispatcher.

    S/T/V use ordinary unnormalized primitives; attraction has unit positive
    nuclear charge with its physical negative sign in the graph. Nuclear pair
    repulsion uses e[0:2] as charges. ERIs use full-range Coulomb and unit graph
    weights. The runtime record weight is the sole external multiplicity.
    """
    requests = tuple(requests)
    if not requests or len(requests) != len(set(requests)):
        raise ValueError("first derivative kernels require unique nonempty requests")
    parts = [
        '#include "integrals/first_derivative_runtime.hpp"',
        '#include "integrals/range_moments.hpp"',
    ]
    eri = []
    for i, (operator, components) in enumerate(requests):
        name = f"primitive_{i}"
        if operator == "four_center_eri":
            ir = build_weighted_eri_ir(tuple(map(len, components)))
            from .shell_spec import ShellClassSpec

            spec = ShellClassSpec(
                "".join("spdf"[l] for l in ir.signature.angular), ir.signature.angular
            )
            kernel = build_weighted_eri_kernel(ir, (spec.components.index(components),))
            eri.append((kernel, name))
            continue
        variables = {
            f"{'abc'[j]}_{axis}": f"c[{3 * j + k}]"
            for j in range(3)
            for k, axis in enumerate("xyz")
        }
        prefix = []
        if operator == "nuclear":
            if components:
                raise ValueError("nuclear pair has no AO components")
            graph = Graph()
            delta = [graph.variable(f"a_{a}") - graph.variable(f"b_{a}") for a in "xyz"]
            energy = (
                graph.variable("za")
                * graph.variable("zb")
                * graph.power(graph.sum(d * d for d in delta), -0.5)
            )
            roots = tuple(
                graph.differentiate(energy, graph.variable(f"{center}_{axis}"))
                for center in "ab"
                for axis in "xyz"
            )
            variables.update(za="e[0]", zb="e[1]")
        else:
            ir = build_one_electron_derivative_ir(operator, tuple(map(len, components)))
            kernel = build_one_electron_derivative_kernel(ir, components)
            graph = kernel.graph
            roots = tuple(root for center in kernel.gradients for root in center)
            variables.update(alpha="e[0]", beta="e[1]")
            if kernel.boys_argument is not None:
                emitter = ScalarCEmitter(graph, variables)
                emitter.emit((kernel.boys_argument,))
                prefix = [
                    f"double boys[{kernel.boys_count}]{{}};",
                    "{",
                    *emitter.lines,
                    (
                        f"if (!vibeqc::integrals::range_moments({kernel.boys_count - 1}, "
                        f"{emitter.reference(kernel.boys_argument)}, 1, "
                        "vibeqc::integrals::CoulombRange::Full, 0, boys)) return false;"
                    ),
                    "}",
                ]
                variables.update(
                    {f"boys_{j}": f"boys[{j}]" for j in range(kernel.boys_count)}
                )
        parts.append(_scalar_function(name, graph, roots, variables, prefix))
    if eri:
        parts.append(emit_weighted_eri_primitive_header(tuple(eri), backend="cpu"))
        for _, name in eri:
            parts.append(f"""static bool {name}(const double* e, const double* c, double* out) {{
  vibeqc::scf::generated_weighted_eri::Gradient result{{}};
  const double weight = 1;
  if (!vibeqc::scf::generated_weighted_eri::{name}_primitive(e, c, &weight, result)) return false;
  for (unsigned j = 0; j < 12; ++j) out[j] = result.center[j/3][j%3];
  return true;
}}""")
    parts += [
        'extern "C" int vibeqc_first_derivative_cpu(unsigned kind, const double* records,',
        "    std::size_t count, double* output) {",
        "  switch (kind) {",
    ]
    parts += [
        f"case {i}: return vibeqc::integrals::first_derivative_records(records, count, output, primitive_{i});"
        for i in range(len(requests))
    ]
    parts += ["default: return 1;", "}", "}"]
    return "\n".join(parts) + "\n"

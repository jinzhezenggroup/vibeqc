"""CPU primitive first derivatives from the existing shared scientific graphs.

This finite source generator emits only requested Cartesian component kernels.
Runtime records, normalization, allocation, contraction and atom scatter belong
to consumers. It imports neither the public runtime nor any independent oracle.
"""

import typing
from functools import lru_cache

from .expr import Graph
from .one_electron_derivatives import (
    build_one_electron_derivative_ir,
    build_one_electron_derivative_kernel,
)
from .scalar_c import ScalarCEmitter
from .weighted_eri import build_weighted_eri_ir, build_weighted_eri_kernel
from .weighted_eri_native import emit_weighted_eri_primitive_header


def _scalar_function(
    name: typing.Any,
    graph: typing.Any,
    roots: typing.Any,
    variables: typing.Any,
    prefix: typing.Any = (),
    qualifier: typing.Any = "static",
) -> typing.Any:
    emitter = ScalarCEmitter(graph, variables)
    emitter.emit(roots)
    return "\n".join(
        [
            f"{qualifier} bool {name}(const double* e, const double* c, double* out) {{",
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
def emit_first_derivative_cpu(requests: typing.Any) -> typing.Any:
    """Emit ordered ``(operator, components)`` kernels and a checked dispatcher.

    S/T/V use ordinary unnormalized primitives; attraction has unit positive
    nuclear charge with its physical negative sign in the graph. Nuclear pair
    repulsion uses e[0:2] as charges. ERIs use full-range Coulomb and unit graph
    weights. The runtime record weight is the sole external multiplicity.
    """
    return _emit_first_derivative(requests, backend="cpu")


@lru_cache(maxsize=4)
def emit_first_derivative_cuda(
    requests: typing.Any, *, symbol: str = "first_derivative"
) -> typing.Any:
    """Lower the identical primitive graphs to a device-only dispatcher.

    Allocation, primitive records, weighting and atom reduction belong to a
    bounded native consumer. No host primitive implementation is emitted.
    """
    return _emit_first_derivative(requests, backend="cuda", symbol=symbol)


def _emit_first_derivative(
    requests: typing.Any, *, backend: typing.Any, symbol: str = "first_derivative"
) -> typing.Any:
    requests = tuple(requests)
    if not requests or len(requests) != len(set(requests)):
        raise ValueError("first derivative kernels require unique nonempty requests")
    if backend == "cuda" and (
        type(symbol) is not str
        or not symbol
        or not symbol.isascii()
        or not (symbol[0].isalpha() or symbol[0] == "_")
        or any(not (character.isalnum() or character == "_") for character in symbol)
    ):
        raise ValueError("CUDA first derivative dispatcher requires a C identifier")
    qualifier = "static" if backend == "cpu" else "__device__ __noinline__"
    primitive_prefix = "" if symbol == "first_derivative" else f"{symbol}_"
    parts = [
        (
            '#include "integrals/first_derivative_runtime.hpp"'
            if backend == "cpu"
            else "#include <cuda_runtime.h>"
        ),
        '#include "integrals/range_moments.hpp"',
    ]
    eri = []
    for i, (operator, components) in enumerate(requests):
        name = f"{primitive_prefix}primitive_{i}"
        if operator == "four_center_eri":
            if len(components) != 4:
                raise ValueError("four-center ERI requires four component labels")
            angular = (
                len(components[0]),
                len(components[1]),
                len(components[2]),
                len(components[3]),
            )
            ir = build_weighted_eri_ir(angular)
            from .shell_spec import ShellClassSpec

            spec = ShellClassSpec("".join("spdf"[l] for l in angular), angular)
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
        parts.append(_scalar_function(name, graph, roots, variables, prefix, qualifier))
    if eri:
        parts.append(emit_weighted_eri_primitive_header(tuple(eri), backend=backend))
        for _, name in eri:
            parts.append(f"""{qualifier} bool {name}(const double* e, const double* c, double* out) {{
  vibeqc::scf::generated_weighted_eri::Gradient result{{}};
  const double weight = 1;
  if (!vibeqc::scf::generated_weighted_eri::{name}_primitive(e, c, &weight, result)) return false;
  for (unsigned j = 0; j < 12; ++j) out[j] = result.center[j/3][j%3];
  return true;
}}""")
    if backend == "cuda":
        parts += [
            f"__device__ bool {symbol}(unsigned kind, const double* e, const double* c, double* out) {{",
            "switch (kind) {",
            *(
                f"case {i}: return {primitive_prefix}primitive_{i}(e, c, out);"
                for i in range(len(requests))
            ),
            "default: return false;",
            "}",
            "}",
        ]
        return "\n".join(parts) + "\n"
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

"""Opt-in scalar S/T/V/DF lowering through g; independent of AOT catalogs.

One compilation unit evaluates one explicit Cartesian primitive component.
Inputs are positive exponents followed by operator centers in xyz order.
Outputs are the raw value followed by first center/xyz derivatives. They are
unnormalized, unscreened atomic-unit quantities; contraction and arbitrary
fixed weights belong to the caller. No HF/DF CUDA method promotion is implied.
"""

from .df_derivatives import build_df_derivative_kernel
from .expr import AlgebraForm, AlgebraOrdering, RematerializationPolicy
from .ir import OperatorFamily
from .one_electron_derivatives import build_one_electron_derivative_kernel
from .scalar_c import ScalarCEmitter


def emit_bounded_component(integral, components, *, backend="cpu"):
    """Emit one first-derivative component using the existing scientific DAG.

    There is deliberately no implicit full-shell default. Spherical consumers
    must transform normalized Cartesian results (and pull weights back). CPU
    exports ``evaluate(inputs, outputs)``; CUDA exposes that device function
    for a caller-owned launch. Callers validate finite inputs and positive
    exponents before invocation, as for the existing primitive CUDA kernels.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("bounded component backend must be cpu or cuda")
    if integral.derivative is None or integral.derivative.order != 1:
        raise ValueError("bounded components require first-derivative raw IR")
    from .blocks import RawBlock

    if len(integral.contractions) != 1 or not isinstance(
        integral.contractions[0], RawBlock
    ):
        raise ValueError(
            "bounded components expose raw values/gradients; contract weights outside"
        )
    df = integral.operator.family in (
        OperatorFamily.COULOMB_METRIC,
        OperatorFamily.THREE_CENTER_ERI,
    )
    kernel = (
        build_df_derivative_kernel if df else build_one_electron_derivative_kernel
    )(integral, components)
    graph = kernel.graph
    roots = (kernel.value,) + tuple(x for axes in kernel.gradients for x in axes)
    if len(tuple(graph.topological_order(roots))) > 30000:
        raise ValueError("component exceeds the 30000-node scalar compilation budget")
    exponent_count = len(integral.signature.shells)
    names = (
        [f"exponent_{i}" for i in range(exponent_count)]
        + [f"center_{i}_{axis}" for i in integral.operator.centers for axis in "xyz"]
        if df
        else ["alpha", "beta"]
        + [f"{'abc'[i]}_{axis}" for i in integral.operator.centers for axis in "xyz"]
    )
    variables = {name: f"inputs[{i}]" for i, name in enumerate(names)}
    variables.update({f"boys_{i}": f"boys[{i}]" for i in range(kernel.boys_count)})
    emitter = ScalarCEmitter(graph, variables)
    qualifier = 'extern "C"' if backend == "cpu" else "__device__ __noinline__"
    lines = [
        "#include <cmath>",
        f"{qualifier} void evaluate(const double* inputs, double* outputs) {{",
    ]
    if kernel.boys_argument is not None:
        emitter.emit((kernel.boys_argument,))
        lines += emitter.lines
        emitter.lines.clear()
        lines += [
            f"  const double t = {emitter.reference(kernel.boys_argument)};",
            f"  double boys[{kernel.boys_count}];",
            f"  if (t < {kernel.boys_count + 16}.0) {{",
            f"    for (unsigned n = 0; n < {kernel.boys_count}; ++n) {{",
            "      double term = 1.0 / (2.0*n + 1.0), sum = term;",
            "      for (unsigned k = 1; k < 512; ++k) {",
            "        term *= 2.0*t / (2.0*n + 2.0*k + 1.0);",
            "        sum += term;",
            "        if (term <= sum * 2e-16) break;",
            "      }",
            "      boys[n] = exp(-t) * sum;",
            "    }",
            "  } else {",
            "    boys[0] = 0.5 * sqrt(3.14159265358979323846/t) * erf(sqrt(t));",
            f"    for (unsigned n = 1; n < {kernel.boys_count}; ++n)",
            "      boys[n] = ((2.0*n-1.0)*boys[n-1] - exp(-t))/(2.0*t);",
            "  }",
        ]
    if backend == "cuda":
        # Recompute per-coordinate ancestors rather than keeping every gradient
        # root live. Each noinline helper is a separate optimization unit; the
        # caller keeps only a scalar result and the shared Boys array live.
        helpers = ["#include <cmath>"]
        for i, root in enumerate(roots):
            scalar_graph, scalar_roots = graph.apply_algebra_form(
                (root,), AlgebraForm.BINARY
            )
            plan = scalar_graph.materialization_plan(
                scalar_roots,
                RematerializationPolicy.inline_single_use_values(),
                AlgebraOrdering.PRESSURE_AWARE,
            )
            scalar = ScalarCEmitter(scalar_graph, variables, plan)
            scalar.emit(scalar_roots)
            helpers += [
                f"__device__ __noinline__ double component_output_{i}(const double* inputs, const double* boys) {{",
                *scalar.lines,
                f"  return {scalar.reference(scalar_roots[0])};",
                "}",
            ]
        boys = "boys" if kernel.boys_argument is not None else "nullptr"
        lines += [
            f"  outputs[{i}] = component_output_{i}(inputs, {boys});"
            for i in range(len(roots))
        ]
        return "\n".join(helpers + lines + ["}", ""])
    emitter.emit(roots)
    lines += emitter.lines
    lines += [
        f"  outputs[{i}] = {emitter.reference(root)};" for i, root in enumerate(roots)
    ]
    return "\n".join(lines + ["}", ""])

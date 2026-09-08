"""Consumer-pruned scalar differentiation and interpretable array execution."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_codegen.expr import AlgebraForm

from .expressions import energy_expression
from .spec import UnsupportedXC


def output_set(spec, order):
    """Energy, feature gradient, then packed upper-triangle feature Hessian."""
    if type(order) is not int or order not in (0, 1, 2):
        raise UnsupportedXC("XC supports derivative orders 0, 1 and 2")
    result = [()]
    if order >= 1:
        result.extend((i,) for i in range(len(spec.features)))
    if order >= 2:
        result.extend(combinations_with_replacement(range(len(spec.features)), 2))
    return tuple(result)


def validate_features(spec, features, *, order=2):
    """Validate interior-v1 without changing any feature or derivative.

    Positive densities span 24 decades, spin fractions reach 1e-10 and reduced
    gradients reach 1e6. Outside this finite audited domain callers receive an
    explicit unsupported result. Only order-zero vacuum energy is defined.
    """
    raw = np.asarray(features)
    if np.iscomplexobj(raw) or raw.ndim != 2 or raw.shape[0] != len(spec.features):
        raise ValueError("XC features require real [feature, point] arrays")
    x = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(x)):
        raise UnsupportedXC("nonfinite XC input")
    if spec.spin == "polarized":
        ra, rb, aa, ab, bb, ta, tb = x
    else:
        n, sigma, tau = x
        ra = rb = n / 2
        aa = ab = bb = sigma / 4
        ta = tb = tau / 2
    vacuum = (ra == 0) & (rb == 0)
    if np.any(vacuum) and (order != 0 or np.any(x[:, vacuum] != 0)):
        raise UnsupportedXC(
            "vacuum supports only zero-feature energy; derivatives are undefined"
        )
    if (
        np.any(ra < 0)
        or np.any(rb < 0)
        or np.any(aa < 0)
        or np.any(bb < 0)
        or np.any(ta < 0)
        or np.any(tb < 0)
    ):
        raise UnsupportedXC("negative density, same-spin sigma or tau")
    # Cauchy-Schwarz constrains physically attainable cross-spin gradients.
    # The tolerance admits roundoff from dot products without clipping them.
    bound = np.sqrt(aa) * np.sqrt(bb)
    if np.any(np.abs(ab) > bound * (1 + 16 * np.finfo(float).eps)):
        raise UnsupportedXC("sigma Gram matrix is not positive semidefinite")
    n = ra + rb
    active = ~vacuum
    if np.any((n[active] < 1e-12) | (n[active] > 1e12)):
        raise UnsupportedXC("total density outside interior-v1 [1e-12, 1e12]")
    if np.any(np.minimum(ra[active], rb[active]) / n[active] < 1e-10):
        raise UnsupportedXC("spin fraction outside interior-v1 [1e-10, 1-1e-10]")
    if "sigma" in spec.ingredients:
        for density, sigma in ((ra, aa), (rb, bb)):
            if np.any(np.sqrt(sigma[active]) / density[active] ** (4 / 3) > 1e6):
                raise UnsupportedXC("reduced gradient exceeds interior-v1 1e6")
    return x, active


def pack_grid_features(spec, values):
    """Map DFT01 spin features; unpolarized conversion requires equal spins."""
    rho, sigma, tau = (np.asarray(values[k]) for k in ("rho", "sigma", "tau"))
    if (
        rho.ndim != 2
        or rho.shape[0] != 2
        or sigma.shape != (3, rho.shape[1])
        or tau.shape != rho.shape
    ):
        raise ValueError("invalid DFT01 feature layout")
    if spec.spin == "polarized":
        return np.concatenate((rho, sigma, tau))
    if not (
        np.array_equal(rho[0], rho[1])
        and np.array_equal(tau[0], tau[1])
        and np.array_equal(sigma[0], sigma[1])
        and np.array_equal(sigma[1], sigma[2])
    ):
        raise UnsupportedXC(
            "unpolarized features require identical spin densities/gradients/tau"
        )
    return np.stack(
        (rho.sum(axis=0), sigma[0] + 2 * sigma[1] + sigma[2], tau.sum(axis=0))
    )


@dataclass(frozen=True)
class XCProgram:
    """One immutable output contract with scalar DAG and reproducible identity."""

    spec: object
    graph: object
    roots: tuple
    outputs: tuple[tuple[int, ...], ...]
    optimization: str
    expression_hash: str

    @property
    def order(self):
        return max(map(len, self.outputs))

    def evaluate(self, features):
        """Interpret the generated DAG on CPU; no autograd or Libxc dependency."""
        x, active = validate_features(self.spec, features, order=self.order)
        result = np.zeros((len(self.outputs), x.shape[1]))
        if not np.any(active):
            return result
        variables = dict(zip(self.spec.features, x[:, active], strict=True))
        values = {}
        # Shared intermediates are evaluated once for the requested roots.
        # This is a diagnostic interpreter, not the CUDA execution fallback.
        with np.errstate(all="raise"):
            for index in self.graph.topological_order(self.roots):
                node = self.graph.nodes[index]
                args = [values[i] for i in node.arguments]
                if node.operation == "constant":
                    value = float(node.payload)
                elif node.operation == "variable":
                    value = variables[node.payload]
                elif node.operation == "add":
                    value = sum(args)
                elif node.operation == "multiply":
                    value = 1
                    for arg in args:
                        value = value * arg
                elif node.operation == "reciprocal":
                    value = 1 / args[0]
                elif node.operation == "power":
                    value = args[0] ** float(node.payload)
                elif node.operation in ("exp", "log", "log1p", "expm1"):
                    value = getattr(np, node.operation)(args[0])
                else:
                    raise UnsupportedXC(f"unsupported XC primitive {node.operation!r}")
                values[index] = value
        for row, root in enumerate(self.roots):
            result[row, active] = values[root.identifier]
        if not np.all(np.isfinite(result)):
            raise ArithmeticError("nonfinite XC output")
        return result

    def unpack(self, result):
        """Return only complete requested tensors; never fill absent derivatives."""
        result = np.asarray(result)
        if result.ndim != 2 or result.shape[0] != len(self.outputs):
            raise ValueError("output shape does not match program")
        rows = dict(zip(self.outputs, result, strict=True))
        answer = {"energy_density": rows[()]} if () in rows else {}
        size = len(self.spec.features)
        if all((i,) in rows for i in range(size)):
            answer["gradient"] = np.stack([rows[(i,)] for i in range(size)])
        if all(
            (i, j) in rows for i, j in combinations_with_replacement(range(size), 2)
        ):
            answer["hessian"] = np.stack(
                [rows[(min(i, j), max(i, j))] for i in range(size) for j in range(size)]
            ).reshape(size, size, result.shape[1])
        return answer


def build_program(spec, *, order=2, outputs=None, optimization="after"):
    """Differentiate only consumer roots, using the shared algebra passes.

    ``before`` enables the independent optimize/differentiate ordering gate.
    Directed mixed partial requests are allowed so symmetry can be checked by
    deriving both orders, instead of merely mirroring a packed Hessian.
    """
    available = output_set(spec, order)
    requested = available if outputs is None else tuple(tuple(v) for v in outputs)
    if not requested or len(set(requested)) != len(requested):
        raise UnsupportedXC("outputs must be nonempty and unique")
    if any(
        len(v) > order
        or any(type(i) is not int or not 0 <= i < len(spec.features) for i in v)
        for v in requested
    ):
        raise UnsupportedXC("unsupported derivative output")
    if optimization not in ("none", "before", "after"):
        raise UnsupportedXC("unsupported optimization order")
    graph, energy, variables = energy_expression(spec)
    if optimization == "before":
        graph, (energy,) = graph.apply_algebra_form(
            (energy,), AlgebraForm.FACTORED_NARY
        )
        variables = tuple(graph.variable(name) for name in spec.features)
    derivatives = {(): energy}
    for output in requested:
        for depth in range(1, len(output) + 1):
            key = output[:depth]
            if key not in derivatives:
                derivatives[key] = graph.differentiate(
                    derivatives[key[:-1]], variables[key[-1]]
                )
    roots = tuple(derivatives[v] for v in requested)
    if optimization == "after":
        graph, roots = graph.apply_algebra_form(roots, AlgebraForm.FACTORED_NARY)
    graph, roots = graph.lower_small_integer_powers(roots)
    reachable = graph.topological_order(roots)
    indices = {index: i for i, index in enumerate(reachable)}
    payload = {
        "spec": spec.to_payload(),
        "outputs": requested,
        "optimization": optimization,
        "nodes": [
            (
                graph.nodes[i].operation,
                [indices[j] for j in graph.nodes[i].arguments],
                str(graph.nodes[i].payload),
            )
            for i in reachable
        ],
        "roots": [indices[r.identifier] for r in roots],
    }
    return XCProgram(
        spec, graph, roots, requested, optimization, canonical_hash(payload)
    )

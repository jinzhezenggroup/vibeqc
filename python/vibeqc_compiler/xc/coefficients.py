"""Generate compact AO bilinear coefficients from the ingredient differential.

The shared Graph differentiates sum_i v_i z_i with respect to rho, grad rho
and tau. This is the pullback of the scalar functional feature gradient.
Directional differentiation of those same roots generates response coefficients
including both the feature Hessian and the changing sigma Jacobian.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.integral.expr import AlgebraForm, Graph


@dataclass(frozen=True)
class CoefficientProgram:
    """Scalar-DAG roots for one compact rho/gradient/kinetic AO pullback."""

    graph: object
    roots: tuple
    labels: tuple
    spin: str
    family: str
    kinetic: bool
    response: bool
    before: dict
    after: dict

    def bind(self, gradient, v, *, delta_gradient=None, delta_v=None):
        """Bind the same validated point ABI for diagnostic and native execution."""
        v = immutable(v)
        size = 7 if self.spin == "polarized" else 3
        if v.ndim != 2 or v.shape[0] != size:
            raise ValueError("invalid XC feature gradient")
        spins, npoint = (2 if self.spin == "polarized" else 1), v.shape[1]

        def bind_gradient(values, prefix):
            if self.family == "lda":
                return {}
            values = immutable(values)
            shape = (2, npoint, 3) if spins == 2 else (npoint, 3)
            if values.shape != shape:
                raise ValueError("invalid density-gradient point/spin layout")
            values = values if spins == 2 else values[None]
            return {
                f"{prefix}{s}_{k}": values[s, :, k]
                for s in range(spins)
                for k in range(3)
            }

        variables = {f"v{i}": value for i, value in enumerate(v)}
        variables.update(bind_gradient(gradient, "g"))
        if self.response:
            dv = immutable(delta_v, shape=v.shape)
            variables.update({f"dv{i}": value for i, value in enumerate(dv)})
            variables.update(bind_gradient(delta_gradient, "dg"))
        elif delta_gradient is not None or delta_v is not None:
            raise ValueError("direction inputs require a response coefficient program")
        return variables, npoint

    def unpack(self, values, npoint):
        """Map generated coefficient roots to their compact bilinear labels."""
        spins = 2 if self.spin == "polarized" else 1
        result = {"rho": np.zeros((spins, npoint))}
        if self.family == "gga":
            result["gradient"] = np.zeros((spins, npoint, 3))
        if self.kinetic:
            result["tau"] = np.zeros((spins, npoint))
        for label, value in zip(self.labels, values, strict=True):
            spin, kind, axis = label
            if axis is None:
                result[kind][spin] = value
            else:
                result[kind][spin, :, axis] = value
        return {key: immutable(value) for key, value in result.items()}

    def evaluate(self, gradient, v, *, delta_gradient=None, delta_v=None):
        """Interpret diagnostic point coefficients with no AO-pair expansion."""
        variables, npoint = self.bind(
            gradient, v, delta_gradient=delta_gradient, delta_v=delta_v
        )
        return self.unpack(
            evaluate_array_graph(self.graph, self.roots, variables), npoint
        )


@lru_cache(maxsize=16)
def coefficient_program(spin, family="gga", *, kinetic=False, response=False):
    """Derive and factor the ingredient pullback using the shared scalar AD.

    ``kinetic=True`` preserves the original generic coefficient helper's tau-half
    contract for synthetic linear-form tests. Audited LDA/GGA consumers request
    False so they neither evaluate tau nor materialize kinetic coefficients.
    """
    if spin not in ("polarized", "unpolarized") or family not in ("lda", "gga"):
        raise ValueError("unsupported coefficient spin/family")
    if type(kinetic) is not bool or type(response) is not bool:
        raise ValueError("coefficient flags must be boolean")
    graph = Graph()
    spins = 2 if spin == "polarized" else 1
    rho = [graph.variable(f"r{s}") for s in range(spins)]
    gradient = [[graph.variable(f"g{s}_{k}") for k in range(3)] for s in range(spins)]
    tau = [graph.variable(f"t{s}") for s in range(spins)]
    sigma = [
        sum((gradient[a][k] * gradient[b][k] for k in range(3)), graph.constant(0))
        for a, b in (((0, 0), (0, 1), (1, 1)) if spins == 2 else ((0, 0),))
    ]
    ingredients = [*rho, *sigma, *tau]
    v = [graph.variable(f"v{i}") for i in range(len(ingredients))]
    active = list(range(spins))
    if family == "gga":
        active.extend(range(spins, len(ingredients) - spins))
    if kinetic:
        active.extend(range(len(ingredients) - spins, len(ingredients)))
    differential = sum((v[i] * ingredients[i] for i in active), graph.constant(0))
    roots, labels = [], []
    for s in range(spins):
        roots.append(graph.differentiate(differential, rho[s]))
        labels.append((s, "rho", None))
        if family == "gga":
            for k in range(3):
                roots.append(graph.differentiate(differential, gradient[s][k]))
                labels.append((s, "gradient", k))
        if kinetic:
            # tau's bilinear is one-half grad(phi_mu) dot grad(phi_nu).
            roots.append(graph.differentiate(differential, tau[s]) / 2)
            labels.append((s, "tau", None))
    if response:
        seed = graph.variable("direction")
        leaves = {f"v{i}": graph.variable(f"dv{i}") for i in range(len(v))}
        leaves.update(
            {
                f"g{s}_{k}": graph.variable(f"dg{s}_{k}")
                for s in range(spins)
                for k in range(3)
            }
        )
        roots = [graph.differentiate(root, seed, leaves) for root in roots]
    before = graph.analyze_ssa(roots).to_payload()
    graph, roots = graph.apply_algebra_form(roots, AlgebraForm.FACTORED_NARY)
    after = graph.analyze_ssa(roots).to_payload()
    return CoefficientProgram(
        graph, roots, tuple(labels), spin, family, kinetic, response, before, after
    )


@dataclass(frozen=True)
class AOJetPullbackProgram:
    """Generated derivatives of compact symmetric AO-pair bilinears."""

    graph: object
    roots: tuple
    family: str
    before: dict
    after: dict

    def bind(self, coefficients, work):
        """Bind D-contracted AO jets and already weighted point coefficients.

        Differentiating both AO legs before binding x/y to the same symmetric
        D-contracted jets produces the correct factor two without a separate
        hand-coded geometric formula. One bounded point-by-AO panel is used.
        """
        work = immutable(work)
        jets = 1 if self.family == "lda" else 4
        if work.ndim != 3 or work.shape[0] != jets:
            raise ValueError("AO pullback requires its minimal contracted jet domain")
        shape = work.shape[1:]
        rho = immutable(coefficients["rho"], shape=(shape[0],))
        values = {"c0": np.broadcast_to(rho[:, None], shape).reshape(-1)}
        if jets > 1:
            gradient = immutable(coefficients["gradient"], shape=(shape[0], 3))
            values.update(
                {
                    f"c{k + 1}": np.broadcast_to(gradient[:, k, None], shape).reshape(
                        -1
                    )
                    for k in range(3)
                }
            )
        for j in range(jets):
            values[f"x{j}"] = values[f"y{j}"] = work[j].reshape(-1)
        return values, shape

    def unpack(self, values, shape):
        return immutable(
            np.stack(
                [
                    np.broadcast_to(value, (shape[0] * shape[1],)).reshape(shape)
                    for value in values
                ]
            )
        )

    def evaluate(self, coefficients, work):
        values, shape = self.bind(coefficients, work)
        return self.unpack(evaluate_array_graph(self.graph, self.roots, values), shape)


@lru_cache(maxsize=2)
def jet_pullback_program(family):
    """Generate AO-jet adjoints from the same compact potential bilinears.

    The full symmetric-D trace contains rho*x0*y0 and, for GGA, each
    c_k*(x_k*y0+x0*y_k). Graph AD owns differentiation of both AO legs;
    geometry consumers supply only center/point translation and weight sources.
    """
    if family not in ("lda", "gga"):
        raise ValueError("AO pullbacks support LDA/GGA only")
    graph = Graph()
    jets = 1 if family == "lda" else 4
    x = [graph.variable(f"x{j}") for j in range(jets)]
    y = [graph.variable(f"y{j}") for j in range(jets)]
    c = [graph.variable(f"c{j}") for j in range(jets)]
    bilinear = c[0] * x[0] * y[0]
    for j in range(1, jets):
        bilinear += c[j] * (x[j] * y[0] + x[0] * y[j])
    roots = [
        graph.differentiate(bilinear, x[j]) + graph.differentiate(bilinear, y[j])
        for j in range(jets)
    ]
    before = graph.analyze_ssa(roots).to_payload()
    graph, roots = graph.apply_algebra_form(roots, AlgebraForm.FACTORED_NARY)
    return AOJetPullbackProgram(
        graph, roots, family, before, graph.analyze_ssa(roots).to_payload()
    )

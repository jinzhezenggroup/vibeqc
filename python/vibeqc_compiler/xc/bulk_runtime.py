"""Generic runtime descriptors for pointwise-qualified bulk Libxc Graphs.

This module bridges the #912 bulk mathematical inventory into the existing XC
execution ABI. It deliberately does not grant production-domain, molecular SCF,
force, response, or public-method capability.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.expr import AlgebraForm, Expr, Graph

from . import libxc_bulk
from .libxc_bulk_capabilities import functional_capability
from .spec import UnsupportedXC

_SUPPORTED_RUNTIME_INGREDIENTS = frozenset(("rho", "sigma", "tau"))
PRODUCTION_CANDIDATE_DOMAIN = "libxc-bulk-production-candidate/v1"
PRODUCTION_DENSITY_CANDIDATE_DOMAIN = "libxc-bulk-production-candidate/v2"
_RUNTIME_DOMAINS = frozenset(
    (
        libxc_bulk.BULK_SEMANTICS,
        PRODUCTION_CANDIDATE_DOMAIN,
        PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    )
)


@dataclass(frozen=True)
class BulkRuntimeSpec:
    """Runtime-facing semantic contract for one imported Libxc registration."""

    identifier: str
    family: str
    spin: str
    features: tuple[str, ...]
    ingredients: tuple[str, ...]
    capability_identity: str
    source_identity: str
    domain: str = libxc_bulk.BULK_SEMANTICS
    density_threshold: float | None = None

    def to_payload(self) -> dict[str, typing.Any]:
        """Return semantic identity without implying production admission."""
        payload = {
            "identifier": self.identifier,
            "family": self.family,
            "spin": self.spin,
            "features": list(self.features),
            "ingredients": list(self.ingredients),
            "domain": self.domain,
            "capability_identity": self.capability_identity,
            "source_identity": self.source_identity,
            "qualification": "pointwise-validated",
            "runtime_candidate": True,
            "production_admitted": False,
        }
        if self.density_threshold is not None:
            payload["density_threshold"] = self.density_threshold
        return payload

    def validate_features(
        self,
        features: typing.Any,
        *,
        order: typing.Any = 1,
        copy: typing.Any = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate the exact versioned bulk runtime-candidate domain."""
        if type(order) is not int or order not in (0, 1, 2):
            raise UnsupportedXC("bulk XC supports derivative orders 0, 1 and 2")
        if type(copy) is not bool:
            raise ValueError("copy must be bool")
        raw = np.asarray(features)
        if np.iscomplexobj(raw) or raw.ndim != 2 or raw.shape[0] != len(self.features):
            raise ValueError("bulk XC features require real [feature, point] arrays")
        if copy:
            x = np.array(raw, dtype=np.float64, order="C", copy=True)
        else:
            if raw.dtype != np.float64 or not raw.flags.c_contiguous:
                raise ValueError(
                    "zero-copy bulk XC features require contiguous float64"
                )
            x = raw
        if not np.all(np.isfinite(x)):
            raise UnsupportedXC("nonfinite bulk XC input")

        rows = dict(zip(self.features, x, strict=True))
        density_names = ("rho_a", "rho_b") if self.spin == "polarized" else ("rho",)
        density_boundary = self.domain == PRODUCTION_DENSITY_CANDIDATE_DOMAIN
        density_invalid = (
            any(np.any(rows[name] < 0) for name in density_names)
            if density_boundary
            else any(np.any(rows[name] <= 0) for name in density_names)
        )
        if density_invalid:
            requirement = "nonnegative" if density_boundary else "strictly positive"
            raise UnsupportedXC(
                f"bulk Libxc runtime candidate requires {requirement} density"
            )
        total_density = sum(rows[name] for name in density_names)
        if density_boundary:
            if self.density_threshold is None:
                raise ValueError(
                    "production density candidate requires a density threshold"
                )
            active = total_density >= self.density_threshold
        else:
            active = np.ones(x.shape[1], dtype=bool)

        if "sigma" in self.ingredients:
            if self.spin == "polarized":
                aa = rows["sigma_aa"]
                ab = rows["sigma_ab"]
                bb = rows["sigma_bb"]
                sigma_invalid = (
                    (aa < 0) | (bb < 0)
                    if self.domain
                    in (
                        PRODUCTION_CANDIDATE_DOMAIN,
                        PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
                    )
                    else (aa <= 0) | (bb <= 0)
                )
                if np.any(sigma_invalid):
                    requirement = (
                        "nonnegative"
                        if self.domain
                        in (
                            PRODUCTION_CANDIDATE_DOMAIN,
                            PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
                        )
                        else "positive"
                    )
                    raise UnsupportedXC(
                        "bulk Libxc runtime candidate requires "
                        f"{requirement} same-spin sigma"
                    )
                bound = np.sqrt(aa) * np.sqrt(bb)
                if np.any(np.abs(ab) > bound * (1 + 16 * np.finfo(float).eps)):
                    raise UnsupportedXC(
                        "bulk XC sigma Gram matrix is not positive semidefinite"
                    )
            else:
                sigma = rows["sigma"]
                sigma_invalid = (
                    sigma < 0
                    if self.domain
                    in (
                        PRODUCTION_CANDIDATE_DOMAIN,
                        PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
                    )
                    else sigma <= 0
                )
                if np.any(sigma_invalid):
                    requirement = (
                        "nonnegative"
                        if self.domain
                        in (
                            PRODUCTION_CANDIDATE_DOMAIN,
                            PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
                        )
                        else "positive"
                    )
                    raise UnsupportedXC(
                        f"bulk Libxc runtime candidate requires {requirement} sigma"
                    )

        if "tau" in self.ingredients:
            tau_names = ("tau_a", "tau_b") if self.spin == "polarized" else ("tau",)
            tau_invalid = (
                any(np.any(rows[name] < 0) for name in tau_names)
                if density_boundary
                else any(np.any(rows[name] <= 0) for name in tau_names)
            )
            if tau_invalid:
                requirement = "nonnegative" if density_boundary else "strictly positive"
                raise UnsupportedXC(
                    f"bulk Libxc runtime candidate requires {requirement} tau"
                )

        return x, active


@dataclass(frozen=True)
class BulkRuntimeProgram:
    """One bulk Libxc Graph lowered into the common XC execution contract."""

    spec: BulkRuntimeSpec
    graph: Graph
    roots: tuple[Expr, ...]
    outputs: tuple[tuple[int, ...], ...]
    optimization: str
    expression_hash: str

    @property
    def order(self) -> int:
        return max(map(len, self.outputs))

    def validate_features(
        self, features: typing.Any, *, copy: typing.Any = True
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.spec.validate_features(features, order=self.order, copy=copy)

    def evaluate(self, features: typing.Any) -> np.ndarray:
        """Interpret the exact generated Graph without a runtime Libxc call."""
        x, active = self.validate_features(features)
        result = np.zeros((len(self.outputs), x.shape[1]), dtype=np.float64)
        if np.any(active):
            variables = dict(zip(self.spec.features, x[:, active], strict=True))
            values = evaluate_array_graph(self.graph, self.roots, variables)
            for row, value in enumerate(values):
                result[row, active] = np.broadcast_to(value, (int(np.sum(active)),))
        if not np.all(np.isfinite(result)):
            raise ArithmeticError("nonfinite bulk XC runtime output")
        return result

    def unpack(self, result: typing.Any) -> dict[str, np.ndarray]:
        result = np.asarray(result)
        if result.ndim != 2 or result.shape[0] != len(self.outputs):
            raise ValueError("output shape does not match bulk XC program")
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


def _output_set(size: int, order: int) -> tuple[tuple[int, ...], ...]:
    if type(order) is not int or order not in (0, 1, 2):
        raise UnsupportedXC("bulk XC supports derivative orders 0, 1 and 2")
    result: list[tuple[int, ...]] = [()]
    if order >= 1:
        result.extend((i,) for i in range(size))
    if order >= 2:
        result.extend(combinations_with_replacement(range(size), 2))
    return tuple(result)


def build_bulk_runtime_program(
    name: str,
    *,
    spin: str = "polarized",
    order: int = 1,
    outputs: typing.Any = None,
    optimization: str = "after",
    domain: str = libxc_bulk.BULK_SEMANTICS,
) -> BulkRuntimeProgram:
    """Build one pointwise-qualified runtime candidate without promoting it.

    Only rho/sigma/tau registrations enter this bridge. The default preserves
    the strictly-positive interior domain. The versioned production candidate
    v1 additionally admits physical zero sigma. v2 also admits nonnegative
    density/tau and applies the pinned Libxc outer total-density screening before
    Graph evaluation. Empty-spin and zero-tau active points are still evaluated
    without clipping so the independent campaign decides whether their E/vxc
    behavior is actually valid. Production-domain and molecular capability remain
    evidence gates owned by downstream lanes.
    """
    if domain not in _RUNTIME_DOMAINS:
        raise UnsupportedXC(f"unsupported bulk runtime domain {domain!r}")
    capability = functional_capability(name)
    ingredients = capability.required_ingredients
    unsupported = set(ingredients) - _SUPPORTED_RUNTIME_INGREDIENTS
    if unsupported:
        raise UnsupportedXC(
            "bulk runtime ingredients are unsupported: "
            + ", ".join(sorted(unsupported))
        )
    if spin not in capability.spin_layouts:
        raise UnsupportedXC(f"bulk runtime spin layout is unsupported: {spin!r}")

    bulk = libxc_bulk.build_bulk_program(capability.name, spin=spin)
    density_threshold = None
    if domain == PRODUCTION_DENSITY_CANDIDATE_DOMAIN:
        catalog = libxc_bulk.read_catalog()
        record = next(
            item for item in catalog["registrations"] if item["name"] == capability.name
        )
        density_threshold = float(record["bindings"]["p_a_dens_threshold"])
        if not np.isfinite(density_threshold) or density_threshold < 0.0:
            raise UnsupportedXC("bulk Libxc density threshold is invalid")
    # Project the runtime ABI, not the imported expression. Flags alone are not
    # proof that an input is dead; never replace a reachable variable with zero.
    families = {"rho": "rho", "sigma": "sigma", "lapl": "laplacian", "tau": "tau"}
    reachable_inputs = set(bulk.graph.topological_order((bulk.energy,)))
    selected = []
    for feature, variable in zip(bulk.features, bulk.variables, strict=True):
        family = families.get(feature.split("_", 1)[0])
        if family in ingredients:
            selected.append((feature, variable))
        elif variable.identifier in reachable_inputs:
            raise UnsupportedXC(
                f"bulk runtime cannot discard reachable feature: {feature}"
            )
    features = tuple(feature for feature, _ in selected)
    variables = tuple(variable for _, variable in selected)
    spec = BulkRuntimeSpec(
        identifier=capability.name,
        family=capability.family,
        spin=spin,
        features=features,
        ingredients=ingredients,
        capability_identity=capability.identity,
        source_identity=bulk.identity,
        domain=domain,
        density_threshold=density_threshold,
    )
    available = _output_set(len(spec.features), order)
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

    graph, energy = bulk.graph, bulk.energy
    if optimization == "before":
        graph, (energy,) = graph.apply_algebra_form(
            (energy,), AlgebraForm.FACTORED_NARY
        )
        variables = tuple(graph.variable(feature) for feature in spec.features)

    derivatives: dict[tuple[int, ...], Expr] = {(): energy}
    for output in requested:
        for depth in range(1, len(output) + 1):
            key = output[:depth]
            if key not in derivatives:
                derivatives[key] = graph.differentiate(
                    derivatives[key[:-1]], variables[key[-1]]
                )
    roots = tuple(derivatives[output] for output in requested)
    if optimization == "after":
        graph, roots = graph.apply_algebra_form(roots, AlgebraForm.FACTORED_NARY)
    graph, roots = graph.lower_small_integer_powers(roots)

    reachable = graph.topological_order(roots)
    indices = {identifier: i for i, identifier in enumerate(reachable)}
    expression_hash = canonical_hash(
        {
            "schema": "vibeqc.bulk-xc-runtime.v1",
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
            "roots": [indices[root.identifier] for root in roots],
        }
    )
    return BulkRuntimeProgram(
        spec=spec,
        graph=graph,
        roots=roots,
        outputs=requested,
        optimization=optimization,
        expression_hash=expression_hash,
    )

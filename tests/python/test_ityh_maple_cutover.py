"""Production cutover gates for pinned Libxc ITYH Maple exchange."""

from fractions import Fraction

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Expr, Graph
from vibeqc_compiler.xc import ityh_maple
from vibeqc_compiler.xc.ityh_maple import ityh_exchange, ityh_maple_provenance
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.spec import FunctionalSpec

POLARIZED = ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb", "tau_a", "tau_b")
UNPOLARIZED = ("rho", "sigma", "tau")


def _spec(spin: str) -> FunctionalSpec:
    return FunctionalSpec(
        "GGA_X_ITYH_PRODUCTION",
        (("GGA_X_ITYH", Fraction(1)),),
        spin=spin,
        range_omega=Fraction(33, 100),
    )


def _roots(graph: Graph, energy: Expr, variables: tuple[Expr, ...]) -> tuple[Expr, ...]:
    first = tuple(graph.differentiate(energy, variable) for variable in variables)
    second = tuple(
        graph.differentiate(first[i], variables[j])
        for i in range(len(variables))
        for j in range(i, len(variables))
    )
    return (energy, *first, *second)


@pytest.mark.parametrize(
    ("spin", "point"),
    (
        ("unpolarized", np.array((0.5, 0.031, 0.0))),
        ("polarized", np.array((0.3, 0.2, 0.015, 0.003, 0.010, 0.0, 0.0))),
    ),
)
def test_production_ityh_matches_dedicated_maple_adapter(
    spin: str, point: np.ndarray
) -> None:
    spec = _spec(spin)
    names = POLARIZED if spin == "polarized" else UNPOLARIZED
    graph = Graph()
    variables = tuple(graph.variable(name) for name in names)
    energy = ityh_exchange(graph, spec, variables)
    values = evaluate_array_graph(
        graph, _roots(graph, energy, variables), dict(zip(names, point, strict=True))
    )
    adapter = np.asarray([float(np.asarray(value)) for value in values])
    production = build_program(spec, order=2).evaluate(point[:, None])[:, 0]
    np.testing.assert_allclose(production, adapter, rtol=2e-13, atol=2e-13)


def test_rsh_dispatch_calls_ityh_maple_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def replacement(
        graph: Graph, spec: FunctionalSpec, variables: tuple[Expr, ...]
    ) -> Expr:
        nonlocal called
        called = True
        return graph.constant(0)

    monkeypatch.setattr(ityh_maple, "ityh_exchange", replacement)
    program = build_program(_spec("unpolarized"), order=0)
    assert called
    np.testing.assert_array_equal(
        program.evaluate(np.array([[0.5], [0.031], [0.0]])), 0.0
    )


def test_ityh_identity_records_maple_provenance() -> None:
    spec = _spec("polarized")
    provenance = spec.to_payload()["expression_provenance"]
    direct = ityh_maple_provenance(spec.components, spec.range_omega)
    assert direct is not None
    assert provenance["kind"] == "libxc-maple"
    assert set(provenance["components"]) == {"GGA_X_ITYH"}
    assert provenance["components"]["GGA_X_ITYH"] == direct["components"]["GGA_X_ITYH"]
    assert len(provenance["adapter_sha256"]) == 64
    assert len(provenance["importer_sha256"]) == 64


def test_cutover_adapter_is_pinned_in_scientific_source_registry() -> None:
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    registry = json.loads((root / "upstream/manifest.json").read_text())
    inputs = registry["products"]["libxc-xc-admission"]["canonical_inputs"]
    relative = "python/vibeqc_compiler/xc/ityh_maple.py"
    assert (
        inputs[relative] == hashlib.sha256((root / relative).read_bytes()).hexdigest()
    )

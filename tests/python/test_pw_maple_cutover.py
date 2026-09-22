"""Production cutover gates for pinned Libxc PW/PW-mod correlation."""

from fractions import Fraction

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Expr, Graph
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.pw_maple import pw_correlation, pw_maple_provenance
from vibeqc_compiler.xc.spec import FunctionalSpec

POLARIZED = ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb", "tau_a", "tau_b")
UNPOLARIZED = ("rho", "sigma", "tau")


def _roots(graph: Graph, energy: Expr, variables: tuple[Expr, ...]) -> tuple[Expr, ...]:
    first = tuple(graph.differentiate(energy, variable) for variable in variables)
    second = tuple(
        graph.differentiate(first[i], variables[j])
        for i in range(len(variables))
        for j in range(i, len(variables))
    )
    return (energy, *first, *second)


@pytest.mark.parametrize(
    ("name", "modified"),
    (("LDA_C_PW", False), ("LDA_C_PW_MOD", True)),
)
@pytest.mark.parametrize(
    ("spin", "point"),
    (
        ("unpolarized", np.array((0.5, 0.0, 0.0))),
        ("polarized", np.array((0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0))),
    ),
)
def test_production_pw_matches_dedicated_maple_adapter(
    name: str, modified: bool, spin: str, point: np.ndarray
) -> None:
    spec = FunctionalSpec(name, ((name, Fraction(1)),), spin=spin)
    names = POLARIZED if spin == "polarized" else UNPOLARIZED
    graph = Graph()
    variables = tuple(graph.variable(feature) for feature in names)
    energy = pw_correlation(graph, spec, variables, modified=modified)
    values = evaluate_array_graph(
        graph, _roots(graph, energy, variables), dict(zip(names, point, strict=True))
    )
    adapter = np.asarray([float(np.asarray(value)) for value in values])
    production = build_program(spec, order=2).evaluate(point[:, None])[:, 0]
    np.testing.assert_allclose(production, adapter, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize(
    ("name", "modified"),
    (("LDA_C_PW", False), ("LDA_C_PW_MOD", True)),
)
def test_pw_identity_records_maple_provenance(name: str, modified: bool) -> None:
    spec = FunctionalSpec(name, ((name, Fraction(1)),), spin="polarized")
    provenance = spec.to_payload()["expression_provenance"]
    direct = pw_maple_provenance(spec.components)
    assert direct is not None
    assert provenance["kind"] == "libxc-maple"
    assert set(provenance["components"]) == {name}
    assert provenance["components"][name] == direct["components"][name]
    assert len(provenance["adapter_sha256"]) == 64
    assert len(provenance["importer_sha256"]) == 64

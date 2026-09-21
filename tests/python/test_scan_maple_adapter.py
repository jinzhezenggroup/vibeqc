"""Cutover qualification for the production-ready SCAN/r2SCAN Maple adapter."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.cuda import CudaEmitter
from vibeqc_compiler.integral.expr import Expr, Graph
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.xc import libxc_maple, scan_maple
from vibeqc_compiler.xc.fixtures import load_fixture
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.scan_maple import (
    SCAN_COMPONENTS,
    scan_component,
    scan_maple_provenance,
)
from vibeqc_compiler.xc.spec import FunctionalSpec, functional

from tools.vibeqc_validation.schema import block_error

POLARIZED_FEATURES = (
    "rho_a",
    "rho_b",
    "sigma_aa",
    "sigma_ab",
    "sigma_bb",
    "tau_a",
    "tau_b",
)
UNPOLARIZED_FEATURES = ("rho", "sigma", "tau")


def _feature_roots(
    graph: Graph, energy: Expr, variables: tuple[Expr, ...]
) -> tuple[Expr, ...]:
    first = tuple(graph.differentiate(energy, variable) for variable in variables)
    second = tuple(
        graph.differentiate(first[i], variables[j])
        for i in range(len(variables))
        for j in range(i, len(variables))
    )
    return (energy, *first, *second)


def _adapter_component(
    name: str, spin: str
) -> tuple[Graph, tuple[Expr, ...], tuple[Expr, ...]]:
    graph = Graph()
    feature_names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES
    variables = tuple(graph.variable(item) for item in feature_names)
    spec = FunctionalSpec(
        f"{name}_MAPLE_PRODUCTION",
        ((name, Fraction(1)),),
        spin=spin,
    )
    energy = scan_component(graph, spec, variables, name)
    return graph, _feature_roots(graph, energy, variables), variables


@pytest.mark.parametrize("name", SCAN_COMPONENTS)
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
@pytest.mark.parametrize("domain", ["typical", "boundary"])
def test_scan_adapter_matches_independent_fixture(
    name: str, spin: str, domain: str
) -> None:
    metadata, features, expected, _ = load_fixture(name, spin=spin, domain=domain)
    feature_names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES
    inputs = dict(zip(feature_names, features, strict=True))

    graph, roots, _ = _adapter_component(name, spin)
    actual_values = evaluate_array_graph(graph, roots, inputs)
    actual = np.stack(
        [np.broadcast_to(value, expected.shape[1:]) for value in actual_values],
        axis=0,
    ).astype(float, copy=False)

    report = block_error(actual, expected, **metadata[f"{domain}_tolerance"])
    assert np.isfinite(actual).all()
    assert report["passed"], report

    # Keep the tight ordinary-interior gate against the independent fixture,
    # not a new direct dependency on the retired handwritten implementation.
    if domain == "typical":
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize(
    ("family", "components"),
    [
        ("SCAN", ("MGGA_X_SCAN", "MGGA_C_SCAN")),
        ("R2SCAN", ("MGGA_X_R2SCAN", "MGGA_C_R2SCAN")),
    ],
)
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_scan_adapter_composes_the_same_family_graph(
    family: str, components: tuple[str, str], spin: str
) -> None:
    graph = Graph()
    feature_names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES
    variables = tuple(graph.variable(item) for item in feature_names)
    spec = functional(family, spin=spin)
    energy = graph.sum(
        scan_component(graph, spec, variables, name) for name in components
    )
    roots = _feature_roots(graph, energy, variables)

    _, features, expected, _ = load_fixture(family, spin=spin)
    inputs = dict(zip(feature_names, features, strict=True))
    actual = np.asarray(evaluate_array_graph(graph, roots, inputs), dtype=float)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("name", SCAN_COMPONENTS)
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_production_expression_uses_scan_maple_adapter(name: str, spin: str) -> None:
    spec = FunctionalSpec(
        f"{name}_PRODUCTION",
        ((name, Fraction(1)),),
        spin=spin,
    )
    adapter_graph, adapter_roots, _ = _adapter_component(name, spin)
    feature_names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES
    _, features, _, _ = load_fixture(name, spin=spin)
    inputs = dict(zip(feature_names, features, strict=True))

    production = build_program(spec, order=2).evaluate(features)
    adapter = np.asarray(
        evaluate_array_graph(adapter_graph, adapter_roots, inputs), dtype=float
    )
    np.testing.assert_allclose(production, adapter, rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize(
    ("family", "components"),
    [
        ("SCAN", {"MGGA_X_SCAN", "MGGA_C_SCAN"}),
        ("R2SCAN", {"MGGA_X_R2SCAN", "MGGA_C_R2SCAN"}),
    ],
)
def test_scan_functional_identity_records_maple_provenance(
    family: str, components: set[str]
) -> None:
    provenance = functional(family).to_payload()["expression_provenance"]

    assert provenance["kind"] == "libxc-maple"
    assert set(provenance["components"]) == components
    assert provenance["importer_semantics"].startswith("libxc-maple-graph/")


@pytest.mark.parametrize("name", SCAN_COMPONENTS)
def test_scan_adapter_emits_scalar_c_and_cuda(name: str) -> None:
    graph, roots, _ = _adapter_component(name, "polarized")
    variables = {name: name for name in POLARIZED_FEATURES}

    scalar = ScalarCEmitter(graph, variables)
    scalar.emit(roots)
    cuda = CudaEmitter(graph, variables)
    cuda.emit(roots)

    assert scalar.lines
    assert cuda.lines
    assert any(
        graph.nodes[index].operation == "select_le"
        for index in graph.topological_order(roots)
    )


@pytest.mark.parametrize("name", SCAN_COMPONENTS)
def test_scan_maple_provenance_records_importer_and_source_identity(name: str) -> None:
    provenance = scan_maple_provenance(((name, Fraction(1)),))

    assert provenance is not None
    assert provenance["kind"] == "libxc-maple"
    assert provenance["importer_semantics"].startswith("libxc-maple-graph/")
    assert set(provenance["components"]) == {name}
    record = provenance["components"][name]
    assert len(record["source_sha256"]) == 64
    assert len(record["transitive_sha256"]) == 64
    assert len(provenance["adapter_sha256"]) == 64
    assert len(provenance["importer_sha256"]) == 64


@pytest.mark.parametrize("owner", ["adapter", "importer"])
def test_scan_maple_provenance_tracks_local_math_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str
) -> None:
    module = scan_maple if owner == "adapter" else libxc_maple
    snapshot = tmp_path / Path(module.__file__).name
    snapshot.write_bytes(Path(module.__file__).read_bytes())
    monkeypatch.setattr(module, "__file__", str(snapshot))

    before = scan_maple_provenance((("MGGA_X_R2SCAN", Fraction(1)),))
    assert before is not None
    snapshot.write_bytes(snapshot.read_bytes() + b"\n# changed implementation\n")
    after = scan_maple_provenance((("MGGA_X_R2SCAN", Fraction(1)),))
    assert after is not None
    field = "adapter_sha256" if owner == "adapter" else "importer_sha256"
    assert after[field] != before[field]

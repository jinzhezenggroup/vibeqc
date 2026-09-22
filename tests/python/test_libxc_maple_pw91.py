"""Qualified Libxc Maple import for the PW91 exchange/correlation family."""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.cuda import CudaEmitter
from vibeqc_compiler.integral.expr import Expr, Graph
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.xc import rsh_expressions
from vibeqc_compiler.xc.libxc_maple import (
    IMPORTER_SEMANTICS,
    MapleModule,
    import_maple_file,
)
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.pw91_maple import pw91_component, pw91_maple_provenance
from vibeqc_compiler.xc.rsh_expressions import energy_expression
from vibeqc_compiler.xc.spec import FunctionalSpec

ROOT = Path(__file__).resolve().parents[2]
LIBXC_ROOT = ROOT / "upstream/libxc/7.0.0"
LIBXC_MANIFEST_ROOT = ROOT / "manifests/libxc/7.0.0"
MANIFEST = json.loads((LIBXC_MANIFEST_ROOT / "manifest.json").read_text())
PW91_MANIFEST = json.loads((LIBXC_MANIFEST_ROOT / "rsh-manifest.json").read_text())
FIXTURE = json.loads((ROOT / "tests/data/xc/pw91-hessian.json").read_text())
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


def _coordinates(
    graph: Graph, spin: str
) -> tuple[
    tuple[Expr, ...],
    Expr,
    Expr,
    Expr,
    Expr,
    Expr,
    Expr,
    Expr,
    Expr,
]:
    if spin == "polarized":
        variables = tuple(graph.variable(name) for name in POLARIZED_FEATURES)
        rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, _, _ = variables
        density = rho_a + rho_b
        zeta = (rho_a - rho_b) / density
        total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
        channel_a, channel_b = rho_a, rho_b
        xs_a = sigma_aa.pow(0.5) * channel_a.pow(-4.0 / 3.0)
        xs_b = sigma_bb.pow(0.5) * channel_b.pow(-4.0 / 3.0)
    else:
        variables = tuple(graph.variable(name) for name in UNPOLARIZED_FEATURES)
        density, sigma, _ = variables
        zeta = graph.constant(0)
        total_sigma = sigma
        channel_a = channel_b = density / 2
        xs_a = (sigma / 4).pow(0.5) * channel_a.pow(-4.0 / 3.0)
        xs_b = (sigma / 4).pow(0.5) * channel_b.pow(-4.0 / 3.0)

    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)
    return variables, density, rs, zeta, xt, xs_a, xs_b, channel_a, channel_b


def _imported_component(
    name: str, spin: str
) -> tuple[MapleModule, Graph, tuple[Expr, ...], tuple[Expr, ...]]:
    graph = Graph()
    (
        variables,
        density,
        rs,
        zeta,
        xt,
        xs_a,
        xs_b,
        channel_a,
        channel_b,
    ) = _coordinates(graph, spin)
    if name == "GGA_X_PW91":
        module = import_maple_file(
            LIBXC_ROOT,
            "gga_x_pw91.mpl",
            defines={"gga_x_pw91_params"},
        )
        cx = 3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
        energy = -cx * channel_a.pow(4.0 / 3.0) * module.call(
            graph, "pw91_f", xs_a
        ) - cx * channel_b.pow(4.0 / 3.0) * module.call(graph, "pw91_f", xs_b)
    elif name == "GGA_C_PW91":
        module = import_maple_file(
            LIBXC_ROOT,
            "gga_c_pw91.mpl",
            support_files=("util.mpl",),
        )
        energy = density * module.call(graph, "f", rs, zeta, xt, 0, 0)
    else:
        raise ValueError(name)
    return module, graph, _feature_roots(graph, energy, variables), variables


def _manual_component(name: str, spin: str) -> tuple[Graph, tuple[Expr, ...]]:
    spec = FunctionalSpec(
        f"{name}_MAPLE_REFERENCE",
        ((name, Fraction(1)),),
        spin=spin,
    )
    graph, energy, variables = energy_expression(spec)
    return graph, _feature_roots(graph, energy, variables)


def _fixture_case(name: str, spin: str) -> dict[str, object]:
    return next(
        case
        for case in FIXTURE["cases"]
        if case["name"] == name and case["spin"] == spin
    )


def _evaluate_fixture(
    graph: Graph,
    roots: tuple[Expr, ...],
    names: tuple[str, ...],
    features: np.ndarray,
) -> np.ndarray:
    values = evaluate_array_graph(
        graph,
        roots,
        dict(zip(names, features, strict=True)),
    )
    count = features.shape[1]
    return np.stack([np.broadcast_to(value, (count,)) for value in values]).astype(
        float, copy=False
    )


def test_pw91_importer_semantics_and_source_provenance() -> None:
    assert IMPORTER_SEMANTICS == "libxc-maple-graph/v11"

    exchange, _, _, _ = _imported_component("GGA_X_PW91", "polarized")
    correlation, _, _, _ = _imported_component("GGA_C_PW91", "polarized")

    assert (
        dict(exchange.source_hashes)["gga_x_pw91.mpl"]
        == PW91_MANIFEST["files"]["gga_x_pw91.mpl"]["sha256"]
    )
    assert {"gga_x_pw91_params"} <= set(exchange.defines)
    for source in ("gga_c_pw91.mpl", "lda_c_pw.mpl", "util.mpl"):
        manifest = PW91_MANIFEST if source == "gga_c_pw91.mpl" else MANIFEST
        assert (
            dict(correlation.source_hashes)[source]
            == manifest["files"][source]["sha256"]
        )


@pytest.mark.parametrize("name", ["GGA_X_PW91", "GGA_C_PW91"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_imported_pw91_matches_audited_dag_and_independent_libxc(
    name: str, spin: str
) -> None:
    case = _fixture_case(name, spin)
    features = np.asarray(case["features"], dtype=float)
    expected = np.asarray(case["expected"], dtype=float)
    names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES

    _, imported_graph, imported_roots, _ = _imported_component(name, spin)
    manual_graph, manual_roots = _manual_component(name, spin)
    imported = _evaluate_fixture(imported_graph, imported_roots, names, features)
    manual = _evaluate_fixture(manual_graph, manual_roots, names, features)

    np.testing.assert_allclose(imported, manual, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(imported, expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("name", ["GGA_X_PW91", "GGA_C_PW91"])
def test_imported_pw91_emits_scalar_c_and_cuda(name: str) -> None:
    _, graph, roots, _ = _imported_component(name, "polarized")
    variables = {feature: feature for feature in POLARIZED_FEATURES}

    scalar = ScalarCEmitter(graph, variables)
    scalar.emit(roots)
    cuda = CudaEmitter(graph, variables)
    cuda.emit(roots)

    assert scalar.lines
    assert cuda.lines
    assert len(roots) == 36
    if name == "GGA_X_PW91":
        operations = {
            graph.nodes[index].operation for index in graph.topological_order(roots)
        }
        assert "asinh" in operations


def _production_adapter_component(
    name: str, spin: str
) -> tuple[Graph, tuple[Expr, ...], tuple[Expr, ...]]:
    graph = Graph()
    feature_names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES
    variables = tuple(graph.variable(item) for item in feature_names)
    spec = FunctionalSpec(
        f"{name}_PRODUCTION_MAPLE",
        ((name, Fraction(1)),),
        spin=spin,
    )
    energy = pw91_component(graph, spec, variables, name)
    return graph, _feature_roots(graph, energy, variables), variables


@pytest.mark.parametrize("name", ["GGA_X_PW91", "GGA_C_PW91"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_production_pw91_matches_dedicated_maple_adapter(name: str, spin: str) -> None:
    case = _fixture_case(name, spin)
    features = np.asarray(case["features"], dtype=float)
    feature_names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES
    spec = FunctionalSpec(
        f"{name}_PRODUCTION",
        ((name, Fraction(1)),),
        spin=spin,
    )
    production = build_program(spec, order=2).evaluate(features)
    graph, roots, _ = _production_adapter_component(name, spin)
    adapter = _evaluate_fixture(graph, roots, feature_names, features)
    np.testing.assert_allclose(production, adapter, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize(
    ("name", "attribute"),
    [
        ("GGA_X_PW91", "imported_pw91_exchange"),
        ("GGA_C_PW91", "imported_pw91_correlation"),
    ],
)
def test_rsh_production_dispatch_calls_pw91_maple_adapter(
    name: str, attribute: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def replacement(
        graph: Graph, spec: FunctionalSpec, variables: tuple[Expr, ...]
    ) -> Expr:
        nonlocal called
        called = True
        return graph.constant(0)

    monkeypatch.setattr(rsh_expressions, attribute, replacement)
    spec = FunctionalSpec(
        f"{name}_DISPATCH",
        ((name, Fraction(1)),),
        spin="unpolarized",
    )
    graph, energy, _ = rsh_expressions.energy_expression(spec)
    assert called
    assert graph.node(energy).operation == "constant"


@pytest.mark.parametrize("name", ["GGA_X_PW91", "GGA_C_PW91"])
def test_pw91_functional_identity_records_maple_provenance(name: str) -> None:
    spec = FunctionalSpec(
        f"{name}_PROVENANCE",
        ((name, Fraction(1)),),
        spin="polarized",
    )
    provenance = spec.to_payload()["expression_provenance"]
    direct = pw91_maple_provenance(spec.components)
    assert direct is not None
    assert provenance["kind"] == "libxc-maple"
    assert provenance["importer_semantics"] == IMPORTER_SEMANTICS
    assert set(provenance["components"]) == {name}
    assert provenance["components"][name] == direct["components"][name]
    assert len(provenance["adapter_sha256"]) == 64
    assert len(provenance["importer_sha256"]) == 64

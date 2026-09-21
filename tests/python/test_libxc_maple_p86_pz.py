"""Qualified Libxc Maple imports for the P86/PZ correlation family."""

from __future__ import annotations

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
from vibeqc_compiler.xc.libxc_maple import (
    IMPORTER_SEMANTICS,
    MapleImportError,
    MapleModule,
    import_maple_file,
    import_maple_source,
)
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.spec import FunctionalSpec

ROOT = Path(__file__).resolve().parents[2]
LIBXC_ROOT = ROOT / "external/libxc-7.0.0"
MANIFEST = json.loads((LIBXC_ROOT / "rsh-manifest.json").read_text())
FIXTURE = json.loads((ROOT / "tests/data/xc/p86-hessian.json").read_text())
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
RS_FACTOR = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
P86_BINDINGS = {
    "params_a_malpha": "0.023266",
    "params_a_mbeta": "0.000007389",
    "params_a_mgamma": "8.723",
    "params_a_mdelta": "0.472",
    "params_a_aa": "0.001667",
    "params_a_bb": "0.002568",
    "params_a_ftilde": "0.19195",
    "RS_FACTOR": repr(RS_FACTOR),
}


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


def _coordinates(graph: Graph, spin: str) -> tuple[tuple[Expr, ...], Expr, Expr, Expr]:
    if spin == "polarized":
        variables = tuple(graph.variable(name) for name in POLARIZED_FEATURES)
        rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, _, _ = variables
        density = rho_a + rho_b
        zeta = (rho_a - rho_b) / density
        total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    else:
        variables = tuple(graph.variable(name) for name in UNPOLARIZED_FEATURES)
        density, sigma, _ = variables
        zeta = graph.constant(0)
        total_sigma = sigma
    rs = graph.approximate_constant(RS_FACTOR) * density.pow(-1.0 / 3.0)
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)
    return variables, density, zeta, rs, xt


def _imported_component(
    name: str, spin: str
) -> tuple[MapleModule, Graph, tuple[Expr, ...], tuple[Expr, ...]]:
    graph = Graph()
    variables, density, zeta, rs, xt = _coordinates(graph, spin)
    if name == "LDA_C_PZ":
        module = import_maple_file(
            LIBXC_ROOT,
            "lda_c_pz.mpl",
            defines=("lda_c_pz_params",),
            support_files=("lda_c_pz.c", "util.mpl"),
        )
        epsilon = module.call(graph, "f", rs, zeta)
    elif name == "GGA_C_P86":
        module = import_maple_file(
            LIBXC_ROOT,
            "gga_c_p86.mpl",
            bindings=P86_BINDINGS,
            support_files=("gga_c_p86.c", "lda_c_pz.c", "util.mpl"),
        )
        epsilon = module.call(graph, "f", rs, zeta, xt, 0, 0)
    else:
        raise AssertionError(f"unknown test component {name}")
    energy = density * epsilon
    return module, graph, _feature_roots(graph, energy, variables), variables


def _evaluate(
    graph: Graph,
    roots: tuple[Expr, ...],
    names: tuple[str, ...],
    features: np.ndarray,
) -> np.ndarray:
    values = evaluate_array_graph(graph, roots, dict(zip(names, features, strict=True)))
    return np.stack(
        [np.broadcast_to(value, features.shape[1:]) for value in values], axis=0
    ).astype(float, copy=False)


def test_maple_line_continuation_is_narrow_and_versioned() -> None:
    assert IMPORTER_SEMANTICS == "libxc-maple-graph/v9"
    graph = Graph()
    x = graph.variable("x")
    module = import_maple_source("f := x -> x + \\\n      1:")
    assert graph.evaluate(module.call(graph, "f", x), {"x": 2.0}) == 3.0
    with pytest.raises(MapleImportError, match="unsupported Maple expression"):
        import_maple_source(r"f := x -> x \\ 1:")


def test_p86_pz_import_pins_transitive_source_and_parameters() -> None:
    pz, _, _, _ = _imported_component("LDA_C_PZ", "polarized")
    p86, _, _, _ = _imported_component("GGA_C_P86", "polarized")
    for module, names in (
        (pz, ("lda_c_pz.mpl", "lda_c_pz.c", "util.mpl")),
        (
            p86,
            (
                "gga_c_p86.mpl",
                "gga_c_p86.c",
                "lda_c_pz.mpl",
                "lda_c_pz.c",
                "util.mpl",
            ),
        ),
    ):
        hashes = dict(module.source_hashes)
        for source in names:
            assert hashes[source] == MANIFEST["files"][source]["sha256"]
    assert "lda_c_pz_params" in pz.defines
    assert "lda_c_pz_params" in p86.defines
    assert dict(p86.bindings) == {
        name: str(Fraction(value)) for name, value in P86_BINDINGS.items()
    }

    other = import_maple_file(
        LIBXC_ROOT,
        "gga_c_p86.mpl",
        bindings={**P86_BINDINGS, "params_a_ftilde": "0.192"},
        support_files=("gga_c_p86.c", "lda_c_pz.c", "util.mpl"),
    )
    assert other.transitive_sha256 != p86.transitive_sha256


@pytest.mark.parametrize(
    "case",
    [case for case in FIXTURE["cases"] if case["name"] in {"LDA_C_PZ", "GGA_C_P86"}],
    ids=lambda case: case["name"] + "-" + case["spin"],
)
def test_production_p86_pz_matches_imported_graph_through_feature_hessian(
    case: dict,
) -> None:
    name, spin = case["name"], case["spin"]
    names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES
    features = np.asarray(case["features"], dtype=float)
    _, imported_graph, imported_roots, _ = _imported_component(name, spin)
    imported = _evaluate(imported_graph, imported_roots, names, features)
    spec = FunctionalSpec(
        f"{name}_PRODUCTION",
        ((name, Fraction(1)),),
        spin=spin,
    )
    production = build_program(spec, order=2).evaluate(features)
    np.testing.assert_allclose(production, imported, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize(
    "case",
    [case for case in FIXTURE["cases"] if case["name"] in {"LDA_C_PZ", "GGA_C_P86"}],
    ids=lambda case: case["name"] + "-" + case["spin"],
)
def test_production_p86_pz_matches_independent_libxc_hessian(case: dict) -> None:
    name, spin = case["name"], case["spin"]
    features = np.asarray(case["features"], dtype=float)
    expected = np.asarray(case["expected"], dtype=float)
    spec = FunctionalSpec(
        f"{name}_PRODUCTION",
        ((name, Fraction(1)),),
        spin=spin,
    )
    actual = build_program(spec, order=2).evaluate(features)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize(
    "case",
    [case for case in FIXTURE["cases"] if case["name"] in {"LDA_C_PZ", "GGA_C_P86"}],
    ids=lambda case: case["name"] + "-" + case["spin"],
)
def test_imported_p86_pz_matches_independent_libxc_hessian(case: dict) -> None:
    assert FIXTURE["schema"] == "vibeqc.independent-semilocal-hessian.v1"
    assert FIXTURE["pyscf"] == "2.14.0"
    assert FIXTURE["libxc"] == "7.0.0"
    name, spin = case["name"], case["spin"]
    names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES
    features = np.asarray(case["features"], dtype=float)
    expected = np.asarray(case["expected"], dtype=float)
    _, graph, roots, _ = _imported_component(name, spin)
    actual = _evaluate(graph, roots, names, features)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("name", ["LDA_C_PZ", "GGA_C_P86"])
def test_imported_p86_pz_emit_existing_scalar_c_and_cuda(name: str) -> None:
    _, graph, roots, _ = _imported_component(name, "polarized")
    variables = {feature: feature for feature in POLARIZED_FEATURES}
    scalar = ScalarCEmitter(graph, variables)
    scalar.emit(roots)
    cuda = CudaEmitter(graph, variables)
    cuda.emit(roots)
    assert all(scalar.reference(root) for root in roots)
    assert scalar.lines
    assert cuda.lines

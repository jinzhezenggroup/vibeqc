"""Qualified Libxc Maple import for the LYP correlation family."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Expr, Graph
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.xc.libxc_maple import MapleModule, import_maple_file
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.spec import FunctionalSpec

ROOT = Path(__file__).resolve().parents[2]
LIBXC_ROOT = ROOT / "upstream/libxc/7.0.0"
LIBXC_MANIFEST_ROOT = ROOT / "manifests/libxc/7.0.0"
MANIFEST = json.loads((LIBXC_MANIFEST_ROOT / "rsh-manifest.json").read_text())
FIXTURE = json.loads((ROOT / "tests/data/xc/lyp-hessian.json").read_text())
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
LYP_BINDINGS = {
    "params_a_a": "0.04918",
    "params_a_b": "0.132",
    "params_a_c": "0.2533",
    "params_a_d": "0.349",
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


def _imported_lyp(
    spin: str,
) -> tuple[MapleModule, Graph, tuple[Expr, ...], tuple[Expr, ...]]:
    graph = Graph()
    if spin == "polarized":
        variables = tuple(graph.variable(name) for name in POLARIZED_FEATURES)
        rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, _, _ = variables
        density = rho_a + rho_b
        zeta = (rho_a - rho_b) / density
    else:
        variables = tuple(graph.variable(name) for name in UNPOLARIZED_FEATURES)
        density, sigma, _ = variables
        rho_a = rho_b = density / 2
        sigma_aa = sigma_ab = sigma_bb = sigma / 4
        zeta = graph.constant(0)

    rr = density.pow(-1.0 / 3.0)
    total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)
    xs0 = sigma_aa.pow(0.5) * rho_a.pow(-4.0 / 3.0)
    xs1 = sigma_bb.pow(0.5) * rho_b.pow(-4.0 / 3.0)
    module = import_maple_file(
        LIBXC_ROOT,
        "gga_c_lyp.mpl",
        bindings=LYP_BINDINGS,
        support_files=("gga_c_lyp.c",),
    )
    energy = density * module.call(graph, "f_lyp_rr", rr, zeta, xt, xs0, xs1)
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


def test_lyp_import_pins_maple_and_parameter_sources() -> None:
    module, _, _, _ = _imported_lyp("polarized")
    hashes = dict(module.source_hashes)
    for name in ("gga_c_lyp.mpl", "gga_c_lyp.c"):
        assert hashes[name] == MANIFEST["files"][name]["sha256"]
    assert dict(module.bindings) == {
        name: str(Fraction(value)) for name, value in LYP_BINDINGS.items()
    }
    assert module.include_edges == ()

    changed = {**LYP_BINDINGS, "params_a_d": "0.350"}
    other = import_maple_file(
        LIBXC_ROOT,
        "gga_c_lyp.mpl",
        bindings=changed,
        support_files=("gga_c_lyp.c",),
    )
    assert other.transitive_sha256 != module.transitive_sha256


def test_lyp_fixture_records_independent_generator_identity() -> None:
    generator = ROOT / "tools/generate_semilocal_hessian_cases.py"
    assert FIXTURE["schema"] == "vibeqc.independent-semilocal-hessian.v1"
    assert FIXTURE["pyscf"] == "2.14.0"
    assert FIXTURE["libxc"] == "7.0.0"
    assert (
        FIXTURE["generator_sha256"]
        == hashlib.sha256(generator.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["spin"])
def test_production_lyp_matches_imported_graph_through_feature_hessian(
    case: dict,
) -> None:
    spin = case["spin"]
    names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES
    features = np.asarray(case["features"], dtype=float)
    _, imported_graph, imported_roots, _ = _imported_lyp(spin)
    imported = _evaluate(imported_graph, imported_roots, names, features)
    spec = FunctionalSpec(
        "GGA_C_LYP_PRODUCTION",
        (("GGA_C_LYP", Fraction(1)),),
        spin=spin,
    )
    production = build_program(spec, order=2).evaluate(features)
    np.testing.assert_allclose(production, imported, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["spin"])
def test_production_lyp_matches_independent_libxc_hessian(case: dict) -> None:
    spin = case["spin"]
    features = np.asarray(case["features"], dtype=float)
    expected = np.asarray(case["expected"], dtype=float)
    spec = FunctionalSpec(
        "GGA_C_LYP_PRODUCTION",
        (("GGA_C_LYP", Fraction(1)),),
        spin=spin,
    )
    actual = build_program(spec, order=2).evaluate(features)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["spin"])
def test_imported_lyp_matches_independent_libxc_hessian(case: dict) -> None:
    spin = case["spin"]
    names = POLARIZED_FEATURES if spin == "polarized" else UNPOLARIZED_FEATURES
    features = np.asarray(case["features"], dtype=float)
    expected = np.asarray(case["expected"], dtype=float)
    _, graph, roots, _ = _imported_lyp(spin)
    actual = _evaluate(graph, roots, names, features)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)


def test_imported_lyp_emits_existing_scalar_c_and_cuda_backends() -> None:
    _, graph, roots, _ = _imported_lyp("polarized")
    variables = {name: name for name in POLARIZED_FEATURES}
    scalar = ScalarCEmitter(graph, variables)
    scalar.emit(roots)
    cuda = ScalarCEmitter(graph, variables)
    cuda.emit(roots)
    assert all(scalar.reference(root) for root in roots)
    assert scalar.lines
    assert cuda.lines

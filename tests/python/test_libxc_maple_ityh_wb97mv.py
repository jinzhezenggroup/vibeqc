"""Qualified Libxc 7.0.0 Maple imports for ITYH and omegaB97M-V."""

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
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.xc.libxc_maple import MapleModule, import_maple_file
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.spec import FunctionalSpec

ROOT = Path(__file__).resolve().parents[2]
LIBXC_ROOT = ROOT / "external/libxc-7.0.0"
ZETA_THRESHOLD = "2.220446049250313e-16"
UNPOLARIZED = ("rho", "sigma", "tau")
POLARIZED = (
    "rho_a",
    "rho_b",
    "sigma_aa",
    "sigma_ab",
    "sigma_bb",
    "tau_a",
    "tau_b",
)
WB97MV_BINDINGS = {
    "p_a_zeta_threshold": ZETA_THRESHOLD,
    "p_a_dens_threshold": "1e-13",
    "p_a_cam_omega": "0.3",
    "params_a_c_x": ("0.85", "1.007", "0.259"),
    "params_a_c_ss": ("0.443", "-1.437", "-4.535", "-3.39", "4.278"),
    "params_a_c_os": ("1.0", "1.358", "2.924", "-8.812", "-1.39", "9.142"),
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


def _coordinates(
    graph: Graph, spin: str
) -> tuple[tuple[Expr, ...], Expr, Expr, Expr, Expr, Expr, Expr, Expr]:
    names = POLARIZED if spin == "polarized" else UNPOLARIZED
    variables = tuple(graph.variable(name) for name in names)
    if spin == "polarized":
        rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b = variables
        density = rho_a + rho_b
        zeta = (rho_a - rho_b) / density
    else:
        density, sigma, tau = variables
        rho_a = rho_b = density / 2
        sigma_aa = sigma_ab = sigma_bb = sigma / 4
        tau_a = tau_b = tau / 2
        zeta = graph.constant(0)

    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    xt = (sigma_aa + 2 * sigma_ab + sigma_bb).pow(0.5) * density.pow(-4.0 / 3.0)
    xs_a = sigma_aa.pow(0.5) * rho_a.pow(-4.0 / 3.0)
    xs_b = sigma_bb.pow(0.5) * rho_b.pow(-4.0 / 3.0)
    ts_a = tau_a * rho_a.pow(-5.0 / 3.0)
    ts_b = tau_b * rho_b.pow(-5.0 / 3.0)
    return variables, density, zeta, rs, xt, xs_a, xs_b, ts_a, ts_b


def _imported_ityh(
    spin: str,
) -> tuple[MapleModule, Graph, tuple[Expr, ...], tuple[Expr, ...]]:
    graph = Graph()
    variables, density, zeta, rs, xt, xs_a, xs_b, _, _ = _coordinates(graph, spin)
    module = import_maple_file(
        LIBXC_ROOT,
        "gga_x_ityh.mpl",
        bindings={
            "p_a_zeta_threshold": ZETA_THRESHOLD,
            "p_a_dens_threshold": "1e-14",
            "p_a_cam_omega": Fraction(33, 100),
        },
        support_files=("util.mpl",),
    )
    energy = density * module.call(graph, "f", rs, zeta, xt, xs_a, xs_b)
    return module, graph, _feature_roots(graph, energy, variables), variables


def _imported_wb97mv(
    spin: str,
) -> tuple[MapleModule, Graph, tuple[Expr, ...], tuple[Expr, ...]]:
    graph = Graph()
    variables, density, zeta, rs, xt, xs_a, xs_b, ts_a, ts_b = _coordinates(graph, spin)
    module = import_maple_file(
        LIBXC_ROOT,
        "hyb_mgga_xc_wb97mv.mpl",
        bindings=WB97MV_BINDINGS,
        support_files=("hyb_mgga_xc_wb97mv.c", "util.mpl"),
    )
    energy = density * module.call(
        graph, "f", rs, zeta, xt, xs_a, xs_b, 0, 0, ts_a, ts_b
    )
    return module, graph, _feature_roots(graph, energy, variables), variables


def _evaluate(
    graph: Graph,
    roots: tuple[Expr, ...],
    names: tuple[str, ...],
    point: np.ndarray,
) -> np.ndarray:
    values = evaluate_array_graph(
        graph,
        roots,
        dict(zip(names, np.asarray(point, dtype=float), strict=True)),
    )
    return np.asarray([float(np.asarray(value)) for value in values])


@pytest.mark.parametrize(
    ("spin", "point"),
    (
        ("unpolarized", np.array((0.5, 0.031, 0.0))),
        ("polarized", np.array((0.3, 0.2, 0.015, 0.003, 0.010, 0.0, 0.0))),
    ),
)
def test_imported_ityh_matches_audited_dag_through_feature_hessian(
    spin: str, point: np.ndarray
) -> None:
    _, graph, roots, _ = _imported_ityh(spin)
    spec = FunctionalSpec(
        "GGA_X_ITYH_MAPLE_REFERENCE",
        (("GGA_X_ITYH", Fraction(1)),),
        spin=spin,
        range_omega=Fraction(33, 100),
    )
    names = POLARIZED if spin == "polarized" else UNPOLARIZED
    actual = _evaluate(graph, roots, names, point)
    expected = build_program(spec, order=2).evaluate(point[:, None])[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=5e-13)


@pytest.mark.parametrize(
    ("spin", "point"),
    (
        ("unpolarized", np.array((0.5, 0.031, 0.14))),
        ("polarized", np.array((0.3, 0.2, 0.015, 0.003, 0.010, 0.08, 0.05))),
    ),
)
def test_imported_wb97mv_matches_audited_dag_through_feature_hessian(
    spin: str, point: np.ndarray
) -> None:
    _, graph, roots, _ = _imported_wb97mv(spin)
    spec = resolve_method("WB97M-V", spin=spin).primitives[0].functional
    names = POLARIZED if spin == "polarized" else UNPOLARIZED
    actual = _evaluate(graph, roots, names, point)
    expected = build_program(spec, order=2).evaluate(point[:, None])[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-12)


def test_wb97mv_import_recovers_retained_independent_unpolarized_oracle() -> None:
    _, graph, roots, _ = _imported_wb97mv("unpolarized")
    actual = _evaluate(graph, roots, UNPOLARIZED, np.array((0.5, 0.031, 0.14)))
    np.testing.assert_allclose(
        actual[:4],
        np.array(
            (
                -0.20794318755308802,
                -0.584943261595045,
                -0.0057485048406733,
                -0.04806556699716644,
            )
        ),
        rtol=4e-12,
        atol=3e-13,
    )
    # packed (rho,rho), (rho,sigma), (rho,tau), (sigma,sigma),
    # (sigma,tau), (tau,tau)
    np.testing.assert_allclose(
        actual[4:],
        np.array(
            (
                -0.4519292851770841,
                0.01463966616204928,
                -0.20181739805554566,
                -0.01737790146793691,
                0.00094734584691088,
                0.6669726284210433,
            )
        ),
        rtol=9e-11,
        atol=3e-12,
    )


def test_imports_pin_complete_source_and_parameter_identity() -> None:
    ityh, _, _, _ = _imported_ityh("unpolarized")
    wb97mv, _, _, _ = _imported_wb97mv("unpolarized")
    rsh_manifest = json.loads((LIBXC_ROOT / "rsh-manifest.json").read_text())
    wb_manifest = json.loads((LIBXC_ROOT / "wb97mv-manifest.json").read_text())

    ityh_hashes = dict(ityh.source_hashes)
    for name in (
        "gga_x_ityh.mpl",
        "gga_x_b88.mpl",
        "lda_x_erf.mpl",
        "attenuation.mpl",
        "util.mpl",
    ):
        assert ityh_hashes[name] == rsh_manifest["files"][name]["sha256"]

    wb_hashes = dict(wb97mv.source_hashes)
    for name in (
        "hyb_mgga_xc_wb97mv.mpl",
        "hyb_mgga_xc_wb97mv.c",
        "b97mv.mpl",
        "lda_c_pw.mpl",
        "lda_x_erf.mpl",
        "attenuation.mpl",
        "util.mpl",
    ):
        assert wb_hashes[name] == wb_manifest["files"][name]["sha256"]

    changed = dict(WB97MV_BINDINGS)
    changed["params_a_c_x"] = ("0.851", "1.007", "0.259")
    other = import_maple_file(
        LIBXC_ROOT,
        "hyb_mgga_xc_wb97mv.mpl",
        bindings=changed,
        support_files=("hyb_mgga_xc_wb97mv.c", "util.mpl"),
    )
    assert other.transitive_sha256 != wb97mv.transitive_sha256


@pytest.mark.parametrize("family", ("ityh", "wb97mv"))
def test_imported_full_hessians_emit_existing_c_and_cuda_backends(
    family: str,
) -> None:
    builder = _imported_ityh if family == "ityh" else _imported_wb97mv
    _, graph, roots, variables = builder("polarized")
    names = {
        str(graph.node(variable).payload): str(graph.node(variable).payload)
        for variable in variables
    }
    scalar = ScalarCEmitter(graph, names)
    cuda = CudaEmitter(graph, names)
    scalar.emit(roots)
    cuda.emit(roots)
    assert scalar.lines
    assert cuda.lines
    assert all(scalar.reference(root) for root in roots)

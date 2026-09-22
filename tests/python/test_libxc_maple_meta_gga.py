"""Qualified Maple imports for SCAN-family exchange and correlation."""

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
from vibeqc_compiler.xc.expressions import energy_expression
from vibeqc_compiler.xc.fixtures import load_fixture
from vibeqc_compiler.xc.libxc_maple import (
    MapleImportError,
    MapleModule,
    import_maple_file,
    import_maple_source,
)
from vibeqc_compiler.xc.spec import FunctionalSpec

from tools.vibeqc_validation.schema import block_error

ROOT = Path(__file__).resolve().parents[2]
LIBXC_ROOT = ROOT / "upstream/libxc/7.0.0"
LIBXC_MANIFEST_ROOT = ROOT / "manifests/libxc/7.0.0"
MANIFEST = json.loads((LIBXC_MANIFEST_ROOT / "manifest.json").read_text())
POLARIZED_FEATURES = (
    "rho_a",
    "rho_b",
    "sigma_aa",
    "sigma_ab",
    "sigma_bb",
    "tau_a",
    "tau_b",
)

_SCAN_BINDINGS = {
    "params_a_c1": "0.667",
    "params_a_c2": "0.8",
    "params_a_d": "1.24",
    "params_a_k1": "0.065",
}
_R2SCAN_BINDINGS = {
    **_SCAN_BINDINGS,
    "params_a_eta": "0.001",
    "params_a_dp2": "0.361",
}
_IMPORT_SPECS = {
    "MGGA_X_SCAN": ("mgga_x_scan.mpl", "mgga_x_scan.c", _SCAN_BINDINGS, False),
    "MGGA_X_R2SCAN": (
        "mgga_x_r2scan.mpl",
        "mgga_x_r2scan.c",
        _R2SCAN_BINDINGS,
        True,
    ),
}
_CORRELATION_SPECS = {
    "MGGA_C_SCAN": ("mgga_c_scan.mpl", "mgga_c_scan.c", "scan_f", {}, False),
    "MGGA_C_R2SCAN": (
        "mgga_c_r2scan.mpl",
        "mgga_c_r2scan.c",
        "r2scan_f",
        {"params_a_eta": "0.001"},
        True,
    ),
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


def _imported_exchange(
    name: str, spin: str
) -> tuple[MapleModule, Graph, tuple[Expr, ...], tuple[Expr, ...]]:
    graph = Graph()
    if spin == "polarized":
        variables = tuple(graph.variable(item) for item in POLARIZED_FEATURES)
        rho_a, rho_b, sigma_aa, _, sigma_bb, tau_a, tau_b = variables
        density = rho_a + rho_b
        channel_a, channel_b = rho_a, rho_b
        sigma_a, sigma_b = sigma_aa, sigma_bb
        tau_channel_a, tau_channel_b = tau_a, tau_b
    else:
        variables = tuple(graph.variable(item) for item in ("rho", "sigma", "tau"))
        density, sigma, tau = variables
        channel_a = channel_b = density / 2
        sigma_a = sigma_b = sigma / 4
        tau_channel_a = tau_channel_b = tau / 2

    xs_a = sigma_a.pow(0.5) * channel_a.pow(-4.0 / 3.0)
    xs_b = sigma_b.pow(0.5) * channel_b.pow(-4.0 / 3.0)
    ts_a = tau_channel_a * channel_a.pow(-5.0 / 3.0)
    ts_b = tau_channel_b * channel_b.pow(-5.0 / 3.0)

    entry, parameter_source, bindings, allow_duplicates = _IMPORT_SPECS[name]
    module = import_maple_file(
        LIBXC_ROOT,
        entry,
        bindings=bindings,
        support_files=("util.mpl", parameter_source),
        allow_duplicate_includes=allow_duplicates,
    )
    enhancement_name = "scan_f" if name == "MGGA_X_SCAN" else "r2scan_f"
    cx = 3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
    energy = -cx * channel_a.pow(4.0 / 3.0) * module.call(
        graph, enhancement_name, xs_a, 0, ts_a
    ) - cx * channel_b.pow(4.0 / 3.0) * module.call(
        graph, enhancement_name, xs_b, 0, ts_b
    )
    return module, graph, _feature_roots(graph, energy, variables), variables


def _manual_exchange(name: str, spin: str) -> tuple[Graph, tuple[Expr, ...]]:
    return _manual_component(name, spin)


def _imported_correlation(
    name: str, spin: str
) -> tuple[MapleModule, Graph, tuple[Expr, ...], tuple[Expr, ...]]:
    graph = Graph()
    if spin == "polarized":
        variables = tuple(graph.variable(item) for item in POLARIZED_FEATURES)
        rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b = variables
        density = rho_a + rho_b
        zeta = (rho_a - rho_b) / density
    else:
        variables = tuple(graph.variable(item) for item in ("rho", "sigma", "tau"))
        density, sigma, tau = variables
        rho_a = rho_b = density / 2
        sigma_aa = sigma_ab = sigma_bb = sigma / 4
        tau_a = tau_b = tau / 2
        zeta = graph.constant(0)

    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)
    ts_a = tau_a * rho_a.pow(-5.0 / 3.0)
    ts_b = tau_b * rho_b.pow(-5.0 / 3.0)

    entry, parameter_source, function_name, bindings, allow_duplicates = (
        _CORRELATION_SPECS[name]
    )
    module = import_maple_file(
        LIBXC_ROOT,
        entry,
        bindings=bindings,
        support_files=("util.mpl", parameter_source),
        allow_duplicate_includes=allow_duplicates,
    )
    epsilon = module.call(graph, function_name, rs, zeta, xt, 0, 0, ts_a, ts_b)
    energy = density * epsilon
    return module, graph, _feature_roots(graph, energy, variables), variables


def _manual_component(name: str, spin: str) -> tuple[Graph, tuple[Expr, ...]]:
    spec = FunctionalSpec(
        f"{name}_MPL_REFERENCE",
        ((name, Fraction(1)),),
        spin=spin,
    )
    graph, energy, variables = energy_expression(spec)
    return graph, _feature_roots(graph, energy, variables)


def test_scan_gx_zero_gradient_continuation_is_name_scoped() -> None:
    graph = Graph()
    x = graph.variable("x")
    module = import_maple_source(
        """
        scan_gx := x -> 1 - exp(-scan_a1/sqrt(X2S*x)):
        lookalike := x -> 1 - exp(-scan_a1/sqrt(X2S*x)):
        """,
        bindings={"scan_a1": "4.9479"},
    )

    scan = module.call(graph, "scan_gx", x)
    lookalike = module.call(graph, "lookalike", x)

    assert graph.node(scan).operation == "select_le"
    assert graph.node(lookalike).operation != "select_le"


def test_piecewise_comparisons_preserve_strictness_and_lazy_derivatives() -> None:
    graph = Graph()
    x = graph.variable("x")
    module = import_maple_source(
        """
        lt := x -> my_piecewise3(x < 0, -1, 1):
        le := x -> my_piecewise3(x <= 0, -1, 1):
        safe := x -> my_piecewise3(x <= 0, 0, 1/x):
        five := x -> my_piecewise5(x < 0, -1, x <= 1, 2, 3):
        """
    )

    lt = module.call(graph, "lt", x)
    le = module.call(graph, "le", x)
    safe = module.call(graph, "safe", x)
    five = module.call(graph, "five", x)
    safe_derivative = graph.differentiate(safe, x)

    assert graph.evaluate(lt, {"x": 0.0}) == 1.0
    assert graph.evaluate(le, {"x": 0.0}) == -1.0
    assert graph.evaluate(safe, {"x": 0.0}) == 0.0
    assert graph.evaluate(safe_derivative, {"x": 0.0}) == 0.0
    assert graph.evaluate(five, {"x": -1.0}) == -1.0
    assert graph.evaluate(five, {"x": 1.0}) == 2.0
    assert graph.evaluate(five, {"x": 2.0}) == 3.0


@pytest.mark.parametrize("name", ["MGGA_X_SCAN", "MGGA_X_R2SCAN"])
def test_scan_family_import_records_source_and_parameter_identity(name: str) -> None:
    module, _, _, _ = _imported_exchange(name, "polarized")
    entry, parameter_source, _, _ = _IMPORT_SPECS[name]
    expected = {
        file_name: MANIFEST["files"][file_name]["sha256"]
        for file_name in ("util.mpl", entry, parameter_source)
    }
    for file_name, digest in expected.items():
        assert dict(module.source_hashes)[file_name] == digest
    assert module.bindings

    changed = dict(_IMPORT_SPECS[name][2])
    changed["params_a_k1"] = "0.066"
    other = import_maple_file(
        LIBXC_ROOT,
        entry,
        bindings=changed,
        support_files=("util.mpl", parameter_source),
        allow_duplicate_includes=_IMPORT_SPECS[name][3],
    )
    assert other.transitive_sha256 != module.transitive_sha256


def test_r2scan_import_admits_only_explicit_duplicate_include_graph() -> None:
    module, _, _, _ = _imported_exchange("MGGA_X_R2SCAN", "polarized")
    assert module.include_edges == (
        ("mgga_x_r2scan.mpl", "mgga_x_rscan.mpl"),
        ("mgga_x_rscan.mpl", "mgga_x_scan.mpl"),
        ("mgga_x_r2scan.mpl", "mgga_x_scan.mpl"),
    )


@pytest.mark.parametrize("name", ["MGGA_X_SCAN", "MGGA_X_R2SCAN"])
@pytest.mark.parametrize(
    "features",
    [
        (0.8, 0.6, 0.12, 0.02, 0.07, 0.7, 0.5),
        (1.4, 0.3, 0.4, -0.05, 0.08, 1.0, 0.3),
    ],
)
def test_imported_exchange_matches_audited_dag_through_feature_hessian(
    name: str, features: tuple[float, ...]
) -> None:
    _, graph, roots, _ = _imported_exchange(name, "polarized")
    manual_graph, manual_roots = _manual_exchange(name, "polarized")
    inputs = dict(zip(POLARIZED_FEATURES, features, strict=True))

    imported = np.asarray(evaluate_array_graph(graph, roots, inputs), dtype=float)
    manual = np.asarray(
        evaluate_array_graph(manual_graph, manual_roots, inputs), dtype=float
    )
    np.testing.assert_allclose(imported, manual, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("name", ["MGGA_X_SCAN", "MGGA_X_R2SCAN"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
@pytest.mark.parametrize("domain", ["typical", "boundary"])
def test_imported_exchange_matches_pinned_independent_libxc_fixture(
    name: str, spin: str, domain: str
) -> None:
    metadata, features, expected, _ = load_fixture(name, spin=spin, domain=domain)
    _, graph, roots, _ = _imported_exchange(name, spin)
    feature_names = (
        POLARIZED_FEATURES if spin == "polarized" else ("rho", "sigma", "tau")
    )
    actual_values = evaluate_array_graph(
        graph,
        roots,
        dict(zip(feature_names, features, strict=True)),
    )
    actual = np.stack(
        [np.broadcast_to(value, expected.shape[1:]) for value in actual_values],
        axis=0,
    ).astype(float, copy=False)
    report = block_error(actual, expected, **metadata[f"{domain}_tolerance"])

    assert np.isfinite(actual).all()
    assert report["passed"], report


@pytest.mark.parametrize("name", ["MGGA_X_SCAN", "MGGA_X_R2SCAN"])
def test_imported_exchange_emits_scalar_c(name: str) -> None:
    _, graph, roots, _ = _imported_exchange(name, "polarized")
    feature_names = POLARIZED_FEATURES
    emitter = ScalarCEmitter(graph, {name: name for name in feature_names})
    emitter.emit(roots)
    references = tuple(emitter.reference(root) for root in roots)

    assert len(references) == 36
    assert all(references)
    assert any(
        graph.nodes[index].operation == "select_le"
        for index in graph.topological_order(roots)
    )


def test_qualified_eval_diff_uses_graph_ad_and_rejects_general_eval() -> None:
    graph = Graph()
    rs = graph.variable("rs")
    zeta = graph.variable("zeta")
    module = import_maple_source(
        """
        g := (x1, x2) -> x1^2*x2 + x2^2:
        f := (rs, z) -> eval(diff(g(x1, x2), x1), [x1=rs, x2=z]):
        """
    )
    derivative = module.call(graph, "f", rs, zeta)
    first_rs = graph.differentiate(derivative, rs)
    first_zeta = graph.differentiate(derivative, zeta)

    point = {"rs": 1.5, "zeta": -0.2}
    assert graph.evaluate(derivative, point) == pytest.approx(-0.6)
    assert graph.evaluate(first_rs, point) == pytest.approx(-0.4)
    assert graph.evaluate(first_zeta, point) == pytest.approx(3.0)

    with pytest.raises(MapleImportError, match="only two-argument eval"):
        import_maple_source("f := x -> eval(x):")


@pytest.mark.parametrize("name", ["MGGA_C_SCAN", "MGGA_C_R2SCAN"])
def test_correlation_import_records_source_and_parameter_identity(name: str) -> None:
    module, _, _, _ = _imported_correlation(name, "polarized")
    entry, parameter_source, _, bindings, allow_duplicates = _CORRELATION_SPECS[name]
    expected = {
        file_name: MANIFEST["files"][file_name]["sha256"]
        for file_name in ("util.mpl", entry, parameter_source)
    }
    for file_name, digest in expected.items():
        assert dict(module.source_hashes)[file_name] == digest

    if bindings:
        changed = dict(bindings)
        changed["params_a_eta"] = "0.002"
        other = import_maple_file(
            LIBXC_ROOT,
            entry,
            bindings=changed,
            support_files=("util.mpl", parameter_source),
            allow_duplicate_includes=allow_duplicates,
        )
        assert other.transitive_sha256 != module.transitive_sha256


@pytest.mark.parametrize("name", ["MGGA_C_SCAN", "MGGA_C_R2SCAN"])
@pytest.mark.parametrize(
    "features",
    [
        (0.8, 0.6, 0.12, 0.02, 0.07, 0.7, 0.5),
        (1.4, 0.3, 0.4, -0.05, 0.08, 1.0, 0.3),
    ],
)
def test_imported_correlation_matches_audited_dag_through_feature_hessian(
    name: str, features: tuple[float, ...]
) -> None:
    _, graph, roots, _ = _imported_correlation(name, "polarized")
    manual_graph, manual_roots = _manual_component(name, "polarized")
    inputs = dict(zip(POLARIZED_FEATURES, features, strict=True))

    imported = np.asarray(evaluate_array_graph(graph, roots, inputs), dtype=float)
    manual = np.asarray(
        evaluate_array_graph(manual_graph, manual_roots, inputs), dtype=float
    )
    np.testing.assert_allclose(imported, manual, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("name", ["MGGA_C_SCAN", "MGGA_C_R2SCAN"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
@pytest.mark.parametrize("domain", ["typical", "boundary"])
def test_imported_correlation_matches_pinned_independent_libxc_fixture(
    name: str, spin: str, domain: str
) -> None:
    metadata, features, expected, _ = load_fixture(name, spin=spin, domain=domain)
    _, graph, roots, _ = _imported_correlation(name, spin)
    feature_names = (
        POLARIZED_FEATURES if spin == "polarized" else ("rho", "sigma", "tau")
    )
    actual_values = evaluate_array_graph(
        graph,
        roots,
        dict(zip(feature_names, features, strict=True)),
    )
    actual = np.stack(
        [np.broadcast_to(value, expected.shape[1:]) for value in actual_values],
        axis=0,
    ).astype(float, copy=False)
    report = block_error(actual, expected, **metadata[f"{domain}_tolerance"])

    assert np.isfinite(actual).all()
    assert report["passed"], report


@pytest.mark.parametrize(
    "name",
    ["MGGA_X_SCAN", "MGGA_X_R2SCAN", "MGGA_C_SCAN", "MGGA_C_R2SCAN"],
)
def test_imported_scan_family_emits_scalar_c_and_cuda(name: str) -> None:
    if name.startswith("MGGA_X_"):
        _, graph, roots, _ = _imported_exchange(name, "polarized")
    else:
        _, graph, roots, _ = _imported_correlation(name, "polarized")
    variables = {name: name for name in POLARIZED_FEATURES}

    scalar = ScalarCEmitter(graph, variables)
    scalar.emit(roots)
    scalar_references = tuple(scalar.reference(root) for root in roots)
    cuda = CudaEmitter(graph, variables)
    cuda.emit(roots)

    assert len(scalar_references) == 36
    assert all(scalar_references)
    assert scalar.lines
    assert cuda.lines
    assert any(
        graph.nodes[index].operation == "select_le"
        for index in graph.topological_order(roots)
    )

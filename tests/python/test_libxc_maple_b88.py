"""Qualification tests for pinned Libxc B88 Maple import."""

import json
import math
import typing
from fractions import Fraction
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.cuda import CudaEmitter
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.xc.libxc_maple import import_maple_source
from vibeqc_compiler.xc.rsh_expressions import energy_expression
from vibeqc_compiler.xc.spec import FunctionalSpec

ROOT = Path(__file__).resolve().parents[2]
LIBXC_ROOT = ROOT / "upstream/libxc/7.0.0"
B88_SOURCE = LIBXC_ROOT / "gga_x_b88.mpl"
RSH_MANIFEST = ROOT / "manifests/libxc/7.0.0/rsh-manifest.json"
POLARIZED_FEATURES = (
    "rho_a",
    "rho_b",
    "sigma_aa",
    "sigma_ab",
    "sigma_bb",
    "tau_a",
    "tau_b",
)

# Independent 80-decimal closed-form values for canonical B88 exchange.
# Generated from the published B88 scalar definition directly, not through
# VibeQC Graph/Maple code. Output order is E, feature gradient, packed Hessian.
_INDEPENDENT = {
    "polarized": {
        "features": (0.8, 0.6, 0.12, 0.02, 0.07, 0.7, 0.5),
        "expected": (
            -1.163212553824013471718372851958301306209,
            -1.150651145209020151099383168403579724513,
            -1.045181386136603932129254692626369363089,
            -0.005596832519294079242684187153999632727654,
            0.0,
            -0.008192879517850232730958102266623901835802,
            0.0,
            0.0,
            -0.4830952383093518397228169690981251947704,
            0.0,
            0.009143152847316941911851622324917440557924,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.586182616061546594150848277920287591426,
            0.0,
            0.0,
            0.01776307638246417382548608505383625263965,
            0.0,
            0.0,
            0.0004622533787663087315550573293715349704119,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.001424965326723960782066885659982772485421,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    },
    "unpolarized": {
        "features": (1.1, 0.09, 0.2),
        "expected": (
            -0.8390577500513976766033831377237463243222,
            -1.016028439061841408502843543811147371859,
            -0.004634932363213969936317855997498569659805,
            0.0,
            -0.3091002090313406268560337827439462509156,
            0.005558681585531472391761026727604522448275,
            0.0,
            0.0002723336397250289617500497068046591109916,
            0.0,
            0.0,
        ),
    },
}


def _feature_roots(
    graph: Graph, energy: typing.Any, variables: tuple[typing.Any, ...]
) -> tuple[typing.Any, ...]:
    first = tuple(graph.differentiate(energy, variable) for variable in variables)
    second = tuple(
        graph.differentiate(first[i], variables[j])
        for i, j in combinations_with_replacement(range(len(variables)), 2)
    )
    return (energy, *first, *second)


def _imported_b88(
    spin: str,
) -> tuple[typing.Any, Graph, tuple[typing.Any, ...], tuple[str, ...]]:
    graph = Graph()
    if spin == "polarized":
        names = POLARIZED_FEATURES
        variables = tuple(graph.variable(name) for name in names)
        rho_a, rho_b, sigma_aa, _, sigma_bb, _, _ = variables
    elif spin == "unpolarized":
        names = ("rho", "sigma", "tau")
        variables = tuple(graph.variable(name) for name in names)
        rho, sigma, _ = variables
        rho_a = rho_b = rho / 2
        sigma_aa = sigma_bb = sigma / 4
    else:
        raise ValueError(f"unknown spin layout {spin!r}")

    module = import_maple_source(B88_SOURCE.read_text(), defines={"gga_x_b88_params"})
    cx = 3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
    energy = graph.sum(
        -cx
        * density.pow(4.0 / 3.0)
        * module.call(
            graph,
            "b88_f",
            same_spin_sigma.pow(0.5) * density.pow(-4.0 / 3.0),
        )
        for density, same_spin_sigma in (
            (rho_a, sigma_aa),
            (rho_b, sigma_bb),
        )
    )
    return module, graph, _feature_roots(graph, energy, variables), names


def _manual_b88(spin: str) -> tuple[Graph, tuple[typing.Any, ...], tuple[str, ...]]:
    spec = FunctionalSpec(
        "B88_MPL_REFERENCE",
        (("GGA_X_B88", Fraction(1)),),
        spin=spin,
    )
    graph, energy, variables = energy_expression(spec)
    names = POLARIZED_FEATURES if spin == "polarized" else ("rho", "sigma", "tau")
    return graph, _feature_roots(graph, energy, variables), names


def test_b88_maple_source_is_pinned_and_standard_branch_is_selected() -> None:
    module, _, _, _ = _imported_b88("polarized")
    manifest = json.loads(RSH_MANIFEST.read_text())

    assert module.source_sha256 == manifest["files"]["gga_x_b88.mpl"]["sha256"]
    assert module.assignment_names == ("params_a_beta", "params_a_gamma")
    assert module.function_names == ("b88_f", "f")


@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_imported_b88_matches_audited_dag_through_feature_hessian(spin: str) -> None:
    _, graph, roots, names = _imported_b88(spin)
    manual_graph, manual_roots, _ = _manual_b88(spin)
    features = _INDEPENDENT[spin]["features"]
    inputs = dict(zip(names, features, strict=True))

    imported = np.asarray(evaluate_array_graph(graph, roots, inputs), dtype=float)
    manual = np.asarray(
        evaluate_array_graph(manual_graph, manual_roots, inputs), dtype=float
    )
    np.testing.assert_allclose(imported, manual, rtol=5e-12, atol=5e-13)


@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_imported_b88_matches_independent_closed_form_oracle(spin: str) -> None:
    _, graph, roots, names = _imported_b88(spin)
    fixture = _INDEPENDENT[spin]
    actual = np.asarray(
        evaluate_array_graph(
            graph,
            roots,
            dict(zip(names, fixture["features"], strict=True)),
        ),
        dtype=float,
    )
    expected = np.asarray(fixture["expected"], dtype=float)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)


def test_b88_full_source_wrapper_matches_channel_adapter_energy() -> None:
    module, graph, roots, names = _imported_b88("polarized")
    rho_a = graph.variable("rho_a")
    rho_b = graph.variable("rho_b")
    sigma_aa = graph.variable("sigma_aa")
    sigma_bb = graph.variable("sigma_bb")
    density = rho_a + rho_b
    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    zeta = (rho_a - rho_b) / density
    xs_a = sigma_aa.pow(0.5) * rho_a.pow(-4.0 / 3.0)
    xs_b = sigma_bb.pow(0.5) * rho_b.pow(-4.0 / 3.0)
    wrapper = density * module.call(graph, "f", rs, zeta, 0, xs_a, xs_b)
    inputs = dict(zip(names, _INDEPENDENT["polarized"]["features"], strict=True))

    adapter_energy = evaluate_array_graph(graph, (roots[0],), inputs)[0]
    wrapper_energy = evaluate_array_graph(graph, (wrapper,), inputs)[0]
    np.testing.assert_allclose(wrapper_energy, adapter_energy, rtol=2e-14, atol=2e-15)


def test_imported_b88_emits_scalar_c_and_cuda() -> None:
    _, graph, roots, names = _imported_b88("polarized")
    variables = {name: name for name in names}

    scalar = ScalarCEmitter(graph, variables)
    scalar.emit(roots)
    cuda = CudaEmitter(graph, variables)
    cuda.emit(roots)

    assert scalar.lines
    assert cuda.lines
    assert all(scalar.reference(root) for root in roots)

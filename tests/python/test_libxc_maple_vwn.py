"""Qualification tests for pinned Libxc VWN/VWN-RPA Maple imports."""

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
from vibeqc_compiler.xc.libxc_maple import import_maple_file
from vibeqc_compiler.xc.rsh_expressions import energy_expression
from vibeqc_compiler.xc.spec import FunctionalSpec

ROOT = Path(__file__).resolve().parents[2]
LIBXC_ROOT = ROOT / "external/libxc-7.0.0"
RSH_MANIFEST = LIBXC_ROOT / "rsh-manifest.json"
POLARIZED_FEATURES = (
    "rho_a",
    "rho_b",
    "sigma_aa",
    "sigma_ab",
    "sigma_bb",
    "tau_a",
    "tau_b",
)
_ENTRIES = {
    "LDA_C_VWN": "lda_c_vwn.mpl",
    "LDA_C_VWN_RPA": "lda_c_vwn_rpa.mpl",
}

# Independent high-precision evaluations of the original VWN closed forms.
# The values were produced from the published rs/z equations directly, not
# from the VibeQC Graph or Maple importer. Output order is E, feature gradient,
# then the packed upper feature Hessian.
_INDEPENDENT = {
    ("LDA_C_VWN", "polarized"): (
        -0.10349413699222731123180863690140427,
        -0.076370951233090228665781838357225638,
        -0.090281456543944951542935477268342306,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.020429927551392210745296773773664182,
        -0.04076314631912753865745677479750929,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.038395662951080747984530986246458448,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    ("LDA_C_VWN", "unpolarized"): (
        -0.079628585408739244252200954638183447,
        -0.080768348817450595383491925076271639,
        0.0,
        0.0,
        -0.0079293128457275726018369790063791385,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    ("LDA_C_VWN_RPA", "polarized"): (
        -0.13190591495563997160699778010778432,
        -0.095912201715063854372362063808034304,
        -0.11285193306103894031596020931642746,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.026029951566141503030378448533999308,
        -0.048598474001791855323433663121163799,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.047471900606167168789889807555929552,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    ("LDA_C_VWN_RPA", "unpolarized"): (
        -0.10191857980864459849198976950774196,
        -0.10161450771177833093557012465339572,
        0.0,
        0.0,
        -0.0083989790892358882215373340230322653,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
}
_FEATURES = {
    "polarized": (0.8, 0.6, 0.12, 0.02, 0.07, 0.7, 0.5),
    "unpolarized": (1.1, 0.09, 0.2),
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


def _imported_vwn(
    name: str, spin: str
) -> tuple[typing.Any, Graph, tuple[typing.Any, ...], tuple[str, ...]]:
    graph = Graph()
    if spin == "polarized":
        names = POLARIZED_FEATURES
        variables = tuple(graph.variable(item) for item in names)
        rho_a, rho_b = variables[:2]
        density = rho_a + rho_b
        zeta = (rho_a - rho_b) / density
    elif spin == "unpolarized":
        names = ("rho", "sigma", "tau")
        variables = tuple(graph.variable(item) for item in names)
        density = variables[0]
        zeta = graph.constant(0)
    else:
        raise ValueError(f"unknown spin layout {spin!r}")

    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    module = import_maple_file(LIBXC_ROOT, _ENTRIES[name])
    energy = density * module.call(graph, "f", rs, zeta)
    return module, graph, _feature_roots(graph, energy, variables), names


def _manual_vwn(
    name: str, spin: str
) -> tuple[Graph, tuple[typing.Any, ...], tuple[str, ...]]:
    spec = FunctionalSpec(
        f"{name}_MPL_REFERENCE",
        ((name, Fraction(1)),),
        spin=spin,
    )
    graph, energy, variables = energy_expression(spec)
    names = POLARIZED_FEATURES if spin == "polarized" else ("rho", "sigma", "tau")
    return graph, _feature_roots(graph, energy, variables), names


@pytest.mark.parametrize("name", ["LDA_C_VWN", "LDA_C_VWN_RPA"])
def test_vwn_source_graph_is_pinned(name: str) -> None:
    module, _, _, _ = _imported_vwn(name, "polarized")
    manifest = json.loads(RSH_MANIFEST.read_text())
    expected = {
        file_name: manifest["files"][file_name]["sha256"]
        for file_name in (_ENTRIES[name], "vwn.mpl")
    }

    assert dict(module.source_hashes) == expected
    assert module.include_edges == ((_ENTRIES[name], "vwn.mpl"),)


@pytest.mark.parametrize("name", ["LDA_C_VWN", "LDA_C_VWN_RPA"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_imported_vwn_matches_audited_dag_through_feature_hessian(
    name: str, spin: str
) -> None:
    _, graph, roots, names = _imported_vwn(name, spin)
    manual_graph, manual_roots, _ = _manual_vwn(name, spin)
    inputs = dict(zip(names, _FEATURES[spin], strict=True))

    imported = np.asarray(evaluate_array_graph(graph, roots, inputs), dtype=float)
    manual = np.asarray(
        evaluate_array_graph(manual_graph, manual_roots, inputs), dtype=float
    )
    np.testing.assert_allclose(imported, manual, rtol=5e-11, atol=5e-12)


@pytest.mark.parametrize("name", ["LDA_C_VWN", "LDA_C_VWN_RPA"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_imported_vwn_matches_independent_closed_form_oracle(
    name: str, spin: str
) -> None:
    _, graph, roots, names = _imported_vwn(name, spin)
    actual = np.asarray(
        evaluate_array_graph(
            graph,
            roots,
            dict(zip(names, _FEATURES[spin], strict=True)),
        ),
        dtype=float,
    )
    expected = np.asarray(_INDEPENDENT[(name, spin)], dtype=float)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("name", ["LDA_C_VWN", "LDA_C_VWN_RPA"])
def test_imported_vwn_emits_scalar_c_and_cuda(name: str) -> None:
    _, graph, roots, names = _imported_vwn(name, "polarized")
    variables = {feature: feature for feature in names}

    scalar = ScalarCEmitter(graph, variables)
    scalar.emit(roots)
    cuda = CudaEmitter(graph, variables)
    cuda.emit(roots)

    assert scalar.lines
    assert cuda.lines
    assert len(tuple(scalar.reference(root) for root in roots)) == 36

"""Qualification tests for the fail-closed Libxc Maple importer PoC."""

import json
import math
import typing
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.xc.expressions import energy_expression
from vibeqc_compiler.xc.libxc_maple import MapleImportError, import_maple_source
from vibeqc_compiler.xc.spec import FunctionalSpec

ROOT = Path(__file__).resolve().parents[2]
PBE_X = ROOT / "external/libxc-7.0.0/gga_x_pbe.mpl"
MANIFEST = ROOT / "external/libxc-7.0.0/manifest.json"
FEATURES = (
    "rho_a",
    "rho_b",
    "sigma_aa",
    "sigma_ab",
    "sigma_bb",
    "tau_a",
    "tau_b",
)


def _imported_pbe_x() -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    graph = Graph()
    variables = tuple(graph.variable(name) for name in FEATURES)
    rho_a, rho_b, sigma_aa, _, sigma_bb, _, _ = variables
    density = rho_a + rho_b
    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    zeta = (rho_a - rho_b) / density
    xs_a = sigma_aa.pow(0.5) * rho_a.pow(-4.0 / 3.0)
    xs_b = sigma_bb.pow(0.5) * rho_b.pow(-4.0 / 3.0)

    module = import_maple_source(PBE_X.read_text(), defines={"gga_x_pbe_params"})
    epsilon = module.call(graph, "f", rs, zeta, 0, xs_a, xs_b)
    energy = density * epsilon
    roots = (
        energy,
        *(graph.differentiate(energy, variable) for variable in variables[:5]),
    )
    return module, graph, roots, variables


def _manual_pbe_x() -> tuple[typing.Any, typing.Any]:
    spec = FunctionalSpec(
        "PBE_X_MPL_REFERENCE",
        (("GGA_X_PBE", Fraction(1)),),
        spin="polarized",
    )
    graph, energy, variables = energy_expression(spec)
    roots = (
        energy,
        *(graph.differentiate(energy, variable) for variable in variables[:5]),
    )
    return graph, roots


def test_pbe_x_maple_source_is_pinned_and_standard_branch_is_selected() -> None:
    source = PBE_X.read_text()
    module = import_maple_source(source, defines={"gga_x_pbe_params"})
    manifest = json.loads(MANIFEST.read_text())

    assert module.source_sha256 == manifest["files"]["gga_x_pbe.mpl"]["sha256"]
    assert module.assignment_names == ("params_a_kappa", "params_a_mu")
    assert module.function_names == ("pbe_f0", "pbe_f", "f")


@pytest.mark.parametrize(
    "features",
    [
        (0.8, 0.6, 0.12, 0.02, 0.07, 0.0, 0.0),
        (1.4, 0.3, 0.4, -0.05, 0.08, 0.0, 0.0),
        (0.05, 0.11, 0.003, 0.001, 0.009, 0.0, 0.0),
    ],
)
def test_imported_pbe_x_matches_audited_dag_energy_and_first_partials(
    features: tuple[float, ...],
) -> None:
    _, imported_graph, imported_roots, _ = _imported_pbe_x()
    manual_graph, manual_roots = _manual_pbe_x()
    inputs = dict(zip(FEATURES, features, strict=True))

    imported = np.asarray(
        evaluate_array_graph(imported_graph, imported_roots, inputs), dtype=float
    )
    manual = np.asarray(
        evaluate_array_graph(manual_graph, manual_roots, inputs), dtype=float
    )

    np.testing.assert_allclose(imported, manual, rtol=2e-14, atol=2e-15)


def test_imported_pbe_x_uses_existing_scalar_c_emitter() -> None:
    _, graph, roots, _ = _imported_pbe_x()
    emitter = ScalarCEmitter(graph, {name: name for name in FEATURES})

    emitter.emit(roots)
    references = tuple(emitter.reference(root) for root in roots)

    assert emitter.lines
    assert len(references) == 6
    assert all(reference for reference in references)
    assert "params_a_" not in "\n".join(emitter.lines)


def test_importer_rejects_unqualified_maple_directives() -> None:
    with pytest.raises(MapleImportError, match="unsupported Maple directive"):
        import_maple_source('$include "lda_c_pw.mpl"\n')

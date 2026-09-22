"""Qualification tests for the fail-closed Libxc Maple importer PoC."""

import json
import math
import typing
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.common.evidence import block_error
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.xc.expressions import energy_expression
from vibeqc_compiler.xc.fixtures import load_fixture
from vibeqc_compiler.xc.libxc_maple import (
    MapleImportError,
    import_maple_file,
    import_maple_source,
)
from vibeqc_compiler.xc.spec import FunctionalSpec

ROOT = Path(__file__).resolve().parents[2]
LIBXC_ROOT = ROOT / "upstream/libxc/7.0.0"
LIBXC_MANIFEST_ROOT = ROOT / "manifests/libxc/7.0.0"
PBE_X = LIBXC_ROOT / "gga_x_pbe.mpl"
PBE_C = LIBXC_ROOT / "gga_c_pbe.mpl"
MANIFEST = LIBXC_MANIFEST_ROOT / "manifest.json"
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


def _feature_roots(
    graph: Graph,
    energy: typing.Any,
    variables: tuple[typing.Any, ...],
) -> tuple[typing.Any, ...]:
    first = tuple(graph.differentiate(energy, variable) for variable in variables)
    second = tuple(
        graph.differentiate(first[i], variables[j])
        for i in range(len(variables))
        for j in range(i, len(variables))
    )
    return (energy, *first, *second)


def _qualified_pbe_x(
    spin: str,
) -> tuple[Graph, tuple[typing.Any, ...], tuple[str, ...]]:
    """Build the imported PBE-X graph in canonical spin-channel coordinates."""

    graph = Graph()
    if spin == "polarized":
        names = FEATURES
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

    module = import_maple_source(PBE_X.read_text(), defines={"gga_x_pbe_params"})
    cx = 3.0 / 8.0 * (3.0 / math.pi) ** (1.0 / 3.0) * 4.0 ** (2.0 / 3.0)
    energy = graph.sum(
        -cx
        * density.pow(4.0 / 3.0)
        * module.call(
            graph,
            "pbe_f",
            sigma.pow(0.5) * density.pow(-4.0 / 3.0),
        )
        for density, sigma in ((rho_a, sigma_aa), (rho_b, sigma_bb))
    )
    return graph, _feature_roots(graph, energy, variables), names


def _imported_pbe_c() -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    graph = Graph()
    variables = tuple(graph.variable(name) for name in FEATURES)
    rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, _, _ = variables
    density = rho_a + rho_b
    rs = graph.approximate_constant(
        (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    ) * density.pow(-1.0 / 3.0)
    zeta = (rho_a - rho_b) / density
    total_sigma = sigma_aa + 2 * sigma_ab + sigma_bb
    xt = total_sigma.pow(0.5) * density.pow(-4.0 / 3.0)

    module = import_maple_file(
        LIBXC_ROOT,
        "gga_c_pbe.mpl",
        defines={"gga_c_pbe_params"},
        support_files=("util.mpl",),
    )
    epsilon = module.call(graph, "f", rs, zeta, xt, 0, 0)
    energy = density * epsilon
    return module, graph, _feature_roots(graph, energy, variables), variables


def _imported_pbe_c_unpolarized() -> tuple[typing.Any, typing.Any, typing.Any]:
    graph = Graph()
    variables = tuple(graph.variable(name) for name in ("rho", "sigma", "tau"))
    rho, sigma, _tau = variables
    rs = graph.approximate_constant((3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)) * rho.pow(
        -1.0 / 3.0
    )
    xt = sigma.pow(0.5) * rho.pow(-4.0 / 3.0)

    module = import_maple_file(
        LIBXC_ROOT,
        "gga_c_pbe.mpl",
        defines={"gga_c_pbe_params"},
        support_files=("util.mpl",),
    )
    energy = rho * module.call(graph, "f", rs, 0, xt, 0, 0)
    first = tuple(graph.differentiate(energy, variable) for variable in variables)
    roots = (
        energy,
        *first,
        *(
            graph.differentiate(first[i], variables[j])
            for i in range(len(variables))
            for j in range(i, len(variables))
        ),
    )
    return graph, roots, variables


def test_pbe_x_maple_source_is_pinned_and_standard_branch_is_selected() -> None:
    source = PBE_X.read_text()
    module = import_maple_source(source, defines={"gga_x_pbe_params"})
    manifest = json.loads(MANIFEST.read_text())

    assert module.source_sha256 == manifest["files"]["gga_x_pbe.mpl"]["sha256"]
    assert module.assignment_names == ("params_a_kappa", "params_a_mu")
    assert module.function_names == ("pbe_f0", "pbe_f", "f")


@pytest.mark.parametrize("domain", ["typical", "boundary"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_imported_pbe_x_matches_pinned_independent_oracle(
    spin: str, domain: str
) -> None:
    metadata, features, expected, _ = load_fixture(
        "GGA_X_PBE", spin=spin, domain=domain
    )
    graph, roots, names = _qualified_pbe_x(spin)
    evaluated = evaluate_array_graph(
        graph,
        roots,
        dict(zip(names, features, strict=True)),
    )
    actual = np.stack(
        [np.broadcast_to(value, expected.shape[1:]) for value in evaluated],
        axis=0,
    ).astype(float, copy=False)
    report = block_error(
        actual,
        expected,
        **metadata[f"{domain}_tolerance"],
    )

    assert np.isfinite(actual).all()
    assert report["passed"], report


def test_imported_pbe_x_feature_hessian_uses_existing_scalar_c_emitter() -> None:
    graph, roots, names = _qualified_pbe_x("polarized")
    emitter = ScalarCEmitter(graph, {name: name for name in names})

    emitter.emit(roots)
    references = tuple(emitter.reference(root) for root in roots)

    assert emitter.lines
    assert len(references) == 36
    assert all(references)


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


def test_pbe_c_include_graph_is_pinned_and_deterministic() -> None:
    manifest = json.loads(MANIFEST.read_text())
    first = import_maple_file(
        LIBXC_ROOT,
        "gga_c_pbe.mpl",
        defines={"gga_c_pbe_params"},
        support_files=("util.mpl",),
    )
    second = import_maple_file(
        LIBXC_ROOT,
        "gga_c_pbe.mpl",
        defines={"gga_c_pbe_params"},
        support_files=("util.mpl",),
    )

    expected = {
        name: manifest["files"][name]["sha256"]
        for name in ("gga_c_pbe.mpl", "lda_c_pw.mpl", "util.mpl")
    }
    assert dict(first.source_hashes) == expected
    assert first.source_sha256 == expected["gga_c_pbe.mpl"]
    assert first.include_edges == (("gga_c_pbe.mpl", "lda_c_pw.mpl"),)
    assert first.transitive_sha256 == second.transitive_sha256
    assert {
        "gga_c_pbe_params",
        "lda_c_pw_params",
        "lda_c_pw_modified_params",
    } <= set(first.defines)


@pytest.mark.parametrize("domain", ["typical", "boundary"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_imported_pbe_c_matches_pinned_independent_libxc_oracle(
    spin: str, domain: str
) -> None:
    metadata, features, expected, _ = load_fixture(
        "GGA_C_PBE", spin=spin, domain=domain
    )
    if spin == "polarized":
        _, graph, roots, _ = _imported_pbe_c()
        names = FEATURES
    else:
        graph, roots, _ = _imported_pbe_c_unpolarized()
        names = ("rho", "sigma", "tau")

    evaluated = evaluate_array_graph(
        graph,
        roots,
        dict(zip(names, features, strict=True)),
    )
    # The shared array evaluator deliberately keeps constant roots scalar;
    # broadcast those exact-zero derivative roots to the fixture lane shape
    # before stacking the complete E/vxc/fxc contract.
    actual = np.stack(
        [np.broadcast_to(value, expected.shape[1:]) for value in evaluated],
        axis=0,
    ).astype(float, copy=False)
    report = block_error(
        actual,
        expected,
        **metadata[f"{domain}_tolerance"],
    )

    assert np.isfinite(actual).all()
    assert report["passed"], report


def test_imported_pbe_c_preserves_stable_log_and_exponential_forms() -> None:
    _, graph, roots, _ = _imported_pbe_c()
    operations = {
        graph.nodes[index].operation for index in graph.topological_order(roots)
    }

    assert "log1p" in operations
    assert "expm1" in operations


def test_imported_pbe_c_uses_existing_scalar_c_emitter() -> None:
    _, graph, roots, _ = _imported_pbe_c()
    emitter = ScalarCEmitter(graph, {name: name for name in FEATURES})

    emitter.emit(roots)
    references = tuple(emitter.reference(root) for root in roots)

    assert emitter.lines
    assert len(references) == 36
    assert all(reference for reference in references)


def test_file_importer_rejects_include_escape(tmp_path: Path) -> None:
    source_root = tmp_path / "pinned"
    source_root.mkdir()
    (tmp_path / "outside.mpl").write_text("f := x -> x:")
    (source_root / "entry.mpl").write_text('$include "../outside.mpl"\nf := x -> x:')

    with pytest.raises(MapleImportError, match="escapes or is absent"):
        import_maple_file(source_root, "entry.mpl")


def test_file_importer_rejects_include_cycles(tmp_path: Path) -> None:
    (tmp_path / "a.mpl").write_text('$include "b.mpl"\nf := x -> x:')
    (tmp_path / "b.mpl").write_text('$include "a.mpl"\ng := x -> x:')

    with pytest.raises(MapleImportError, match="cyclic Maple include"):
        import_maple_file(tmp_path, "a.mpl")


def test_file_importer_rejects_duplicate_includes(tmp_path: Path) -> None:
    (tmp_path / "shared.mpl").write_text("g := x -> x:")
    (tmp_path / "entry.mpl").write_text(
        '$include "shared.mpl"\n$include "shared.mpl"\nf := x -> x:'
    )

    with pytest.raises(MapleImportError, match="duplicate Maple include"):
        import_maple_file(tmp_path, "entry.mpl")


def test_non_pbe_identity_does_not_claim_maple_source() -> None:
    spec = FunctionalSpec(
        "LDA_X_REFERENCE",
        (("LDA_X", Fraction(1)),),
        spin="polarized",
    )

    assert "expression_provenance" not in spec.to_payload()


@pytest.mark.parametrize("component", ["GGA_X_PBE", "GGA_C_PBE"])
def test_pbe_production_identity_records_maple_source(component: str) -> None:
    spec = FunctionalSpec(
        f"{component}_PRODUCTION",
        ((component, Fraction(1)),),
        spin="polarized",
    )
    provenance = spec.to_payload()["expression_provenance"]

    assert provenance["kind"] == "libxc-maple"
    assert provenance["importer_semantics"].startswith("libxc-maple-graph/")
    assert set(provenance["components"]) == {component}
    record = provenance["components"][component]
    assert len(record["source_sha256"]) == 64
    assert len(record["transitive_sha256"]) == 64


def test_pbe_production_builder_matches_qualified_imported_graph() -> None:
    features = (0.8, 0.6, 0.12, 0.02, 0.07, 0.0, 0.0)
    inputs = dict(zip(FEATURES, features, strict=True))

    for component, imported in (
        ("GGA_X_PBE", _qualified_pbe_x),
        ("GGA_C_PBE", None),
    ):
        spec = FunctionalSpec(
            f"{component}_PRODUCTION",
            ((component, Fraction(1)),),
            spin="polarized",
        )
        graph, energy, variables = energy_expression(spec)
        roots = _feature_roots(graph, energy, variables)
        if imported is not None:
            imported_graph, imported_roots, names = imported("polarized")
        else:
            _, imported_graph, imported_roots, _ = _imported_pbe_c()
            names = FEATURES

        actual = np.asarray(
            evaluate_array_graph(graph, roots, inputs),
            dtype=float,
        )
        expected = np.asarray(
            evaluate_array_graph(
                imported_graph,
                imported_roots,
                dict(zip(names, features, strict=True)),
            ),
            dtype=float,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)

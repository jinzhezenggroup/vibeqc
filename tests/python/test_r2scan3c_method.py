import hashlib
import json
import typing
from dataclasses import replace
from pathlib import Path

import pytest
from vibeqc import load_r2scan3c_basis
from vibeqc_compiler.method import (
    METHOD_CATALOG,
    BackendCapability,
    DispersionCorrectionPrimitive,
    GeometricCounterpoisePrimitive,
    MethodIR,
    MethodSpec,
    MethodTypeError,
    SemilocalXCPrimitive,
    UnsupportedMethod,
    resolve_method,
    validate_basis_snapshot,
    verify_method_ir,
)


def _root() -> typing.Any:
    return Path(__file__).resolve().parents[2]


def test_r2scan3c_resolves_one_inspectable_method_graph() -> None:
    graph = resolve_method("R2SCAN-3c")
    assert graph.identifier == "R2SCAN-3c"
    assert graph.basis.name == "def2-mTZVPP"
    assert [type(p) for p in graph.primitives] == [
        SemilocalXCPrimitive,
        DispersionCorrectionPrimitive,
        GeometricCounterpoisePrimitive,
    ]
    assert graph.requirements["operators"] == (
        "semilocal-xc",
        "geometry-d4-bj-eeq",
        "geometry-gcp",
    )
    assert graph.primitives[1].specification.profile == "r2scan3c"
    assert graph.primitives[2].specification.profile == "r2scan3c"


def test_plain_r2scan_cannot_be_reported_as_r2scan3c() -> None:
    plain = resolve_method("R2SCAN")
    composite = resolve_method("R2SCAN-3c")
    assert plain.basis is None
    assert len(plain.primitives) == 1
    assert plain.identity != composite.identity
    assert plain.manifest_identity != composite.manifest_identity


def test_canonical_label_rejects_component_override() -> None:
    canonical = METHOD_CATALOG["R2SCAN-3c"]
    wrong_gcp = replace(canonical.gcp, alpha=canonical.gcp.alpha + 0.01)
    with pytest.raises(UnsupportedMethod, match="different explicit identifier"):
        resolve_method(replace(canonical, gcp=wrong_gcp))
    custom = MethodSpec(
        "R2SCAN-3c-custom",
        canonical.semilocal_components,
        dispersion=canonical.dispersion,
        basis=canonical.basis,
        gcp=wrong_gcp,
    )
    assert resolve_method(custom).identity != resolve_method("R2SCAN-3c").identity


def test_canonical_basis_snapshot_is_exact_and_preflighted() -> None:
    basis = load_r2scan3c_basis()
    binding = METHOD_CATALOG["R2SCAN-3c"].basis
    assert (
        validate_basis_snapshot(binding, basis, atomic_numbers=(1, 6, 8, 18)) is basis
    )
    graph = resolve_method("R2SCAN-3c")
    assert graph.preflight_atomic_numbers((1, 6, 8, 18)) == (1, 6, 8, 18)
    with pytest.raises(UnsupportedMethod, match="does not support atomic numbers"):
        graph.preflight_atomic_numbers((19,))
    wrong = replace(basis, name="not-def2-mTZVPP")
    with pytest.raises(ValueError):
        validate_basis_snapshot(binding, wrong, atomic_numbers=(1,))


def test_r2scan3c_correction_parameters_are_method_defining() -> None:
    spec = METHOD_CATALOG["R2SCAN-3c"]
    d4 = spec.dispersion
    assert (d4.s6, d4.s8, d4.s9, d4.a1, d4.a2, d4.ga, d4.gc) == (
        1.0,
        0.0,
        2.0,
        0.42,
        5.65,
        2.0,
        1.0,
    )
    gcp = spec.gcp
    assert (gcp.sigma, gcp.eta, gcp.eta_spec, gcp.alpha, gcp.beta) == (
        1.0,
        1.315,
        1.15,
        0.9410,
        1.4636,
    )
    assert (gcp.damping_scale, gcp.damping_exponent) == (4.0, 6.0)


def test_method_ir_rejects_double_gcp_application() -> None:
    graph = resolve_method("R2SCAN-3c")
    with pytest.raises(UnsupportedMethod, match="unique by operator family"):
        MethodIR(
            graph.identifier,
            graph.spin,
            graph.primitives + (graph.primitives[-1],),
            basis=graph.basis,
        )


def test_r2scan3c_backend_must_acknowledge_both_corrections() -> None:
    graph = resolve_method("R2SCAN-3c")
    capability = BackendCapability(
        backend="qualified-r2scan3c",
        dtypes=("float64",),
        spins=("unpolarized",),
        derivative_orders=(0, 1),
        ingredients=("rho", "sigma", "tau"),
        operators=("semilocal-xc", "geometry-d4-bj-eeq", "geometry-gcp"),
    )
    assert (
        verify_method_ir(graph, capability=capability, derivative_order=1).method
        is graph
    )
    with pytest.raises(MethodTypeError, match="geometry-gcp"):
        verify_method_ir(
            graph,
            capability=replace(
                capability, operators=("semilocal-xc", "geometry-d4-bj-eeq")
            ),
            derivative_order=1,
        )


def test_r2scan3c_audit_manifest_hashes_match_catalog() -> None:
    root = _root()
    manifest = json.loads((root / "external/r2scan3c/manifest.json").read_text())
    spec = METHOD_CATALOG["R2SCAN-3c"]
    assert manifest["basis"]["basis_identity"] == spec.basis.basis_identity
    assert manifest["basis"]["source_export_sha256"] == spec.basis.source_sha256
    gcp_data = root / "external/r2scan3c/gcp-r2scan3c-h-ar.json"
    assert hashlib.sha256(gcp_data.read_bytes()).hexdigest() == spec.gcp.data_sha256
    assert manifest["gcp"]["data_sha256"] == spec.gcp.data_sha256
    generated = root / "src/dft/dispersion/gcp_r2scan3c_data.hpp"
    assert (
        hashlib.sha256(generated.read_bytes()).hexdigest()
        == manifest["gcp"]["generated_header_sha256"]
    )


def test_custom_r2scan3c_keeps_nonlocal_dispersion_and_gcp_distinct() -> None:
    from vibeqc_compiler.method import VV10, original_nonlocal_correlation

    spec = replace(
        METHOD_CATALOG["R2SCAN-3c"],
        identifier="custom-r2scan3c-vv10",
        nonlocal_correlation=original_nonlocal_correlation(VV10),
    )
    graph = resolve_method(spec)
    assert graph.requirements["operators"] == (
        "semilocal-xc",
        "nonlocal-correlation",
        "geometry-d4-bj-eeq",
        "geometry-gcp",
    )
    assert graph.identity != resolve_method("R2SCAN-3c").identity


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_direct_method_ir_cannot_relabel_plain_r2scan_as_canonical_3c(
    spin: typing.Any,
) -> None:
    plain = resolve_method("R2SCAN", spin=spin)
    with pytest.raises(UnsupportedMethod, match="canonical manifest"):
        replace(plain, identifier="R2SCAN-3c")


@pytest.mark.parametrize("component", ["basis", "gcp", "dispersion"])
def test_direct_canonical_method_ir_rejects_changed_components(
    component: typing.Any,
) -> None:
    graph = resolve_method("R2SCAN-3c")
    if component == "basis":
        changes = {"basis": replace(graph.basis, representation="cartesian")}
    elif component == "gcp":
        node = graph.primitives[-1]
        changed = replace(node, specification=replace(node.specification, alpha=1.0))
        changes = {"primitives": (*graph.primitives[:-1], changed)}
    else:
        changes = {"primitives": (graph.primitives[0], graph.primitives[-1])}
    with pytest.raises(UnsupportedMethod, match="canonical manifest"):
        replace(graph, **changes)
    custom = replace(graph, identifier="explicit-custom", **changes)
    assert custom.identity != graph.identity

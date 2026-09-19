"""Canonical DFT MethodSpec -> MethodIR composition gates for #396."""

import json
from fractions import Fraction

import pytest
from vibeqc_compiler.method import (
    METHOD_CATALOG,
    ExactExchangePrimitive,
    MethodIR,
    MethodSpec,
    NonlocalCorrelationPrimitive,
    NonlocalCorrelationSpec,
    SemilocalXCPrimitive,
    UnsupportedMethod,
    original_nonlocal_correlation,
    resolve_method,
)
from vibeqc_compiler.xc.spec import FunctionalSpec, functional


def test_pbe_and_pbe0_resolve_to_typed_primitive_graphs():
    pbe = resolve_method("PBE", spin="unpolarized")
    assert len(pbe.primitives) == 1
    assert isinstance(pbe.primitives[0], SemilocalXCPrimitive)
    assert pbe.primitives[0].functional.components == (
        ("GGA_C_PBE", Fraction(1)),
        ("GGA_X_PBE", Fraction(1)),
    )
    assert pbe.requirements == {
        "spin": "unpolarized",
        "reference": "restricted",
        "ingredients": ("rho", "sigma"),
        "operators": ("semilocal-xc",),
    }

    pbe0 = resolve_method("PBE0", spin="polarized")
    assert len(pbe0.primitives) == 2
    semilocal, exchange = pbe0.primitives
    assert isinstance(semilocal, SemilocalXCPrimitive)
    assert semilocal.functional.components == (
        ("GGA_C_PBE", Fraction(1)),
        ("GGA_X_PBE", Fraction(3, 4)),
    )
    assert isinstance(exchange, ExactExchangePrimitive)
    assert exchange.coefficient == Fraction(1, 4)
    assert exchange.operator == "full-range"
    assert pbe0.requirements["reference"] == "unrestricted"
    assert pbe0.requirements["operators"] == (
        "semilocal-xc",
        "full-range-exchange",
    )


@pytest.mark.parametrize(
    "spin,reference",
    [("unpolarized", "restricted"), ("polarized", "unrestricted")],
)
def test_r2scan_is_one_tau_semilocal_primitive_without_exchange(spin, reference):
    r2scan = resolve_method("R2SCAN", spin=spin)
    assert len(r2scan.primitives) == 1
    primitive = r2scan.primitives[0]
    assert isinstance(primitive, SemilocalXCPrimitive)
    assert not isinstance(primitive, ExactExchangePrimitive)
    assert primitive.functional.components == (
        ("MGGA_C_R2SCAN", Fraction(1)),
        ("MGGA_X_R2SCAN", Fraction(1)),
    )
    assert primitive.functional.ingredients == ("rho", "sigma", "tau")
    assert r2scan.requirements == {
        "spin": spin,
        "reference": reference,
        "ingredients": ("rho", "sigma", "tau"),
        "operators": ("semilocal-xc",),
    }


def test_r2scan_catalog_extension_needs_no_new_primitive_family():
    r2scan = resolve_method("R2SCAN")
    pbe = resolve_method("PBE")
    assert type(r2scan.primitives[0]) is type(pbe.primitives[0])
    assert functional("R2SCAN").exact_exchange == 0


def test_method_identity_is_semantic_while_manifest_identity_retains_name():
    a = MethodSpec(
        "alias-a",
        (
            ("GGA_X_PBE", Fraction(1, 2)),
            ("GGA_C_PBE", Fraction(1)),
            ("GGA_X_PBE", Fraction(1, 4)),
        ),
        exact_exchange=Fraction(1, 4),
    )
    b = MethodSpec(
        "alias-b",
        (
            ("GGA_C_PBE", Fraction(1)),
            ("GGA_X_PBE", Fraction(3, 4)),
        ),
        exact_exchange=Fraction(1, 4),
    )
    ir_a = resolve_method(a, spin="unpolarized")
    ir_b = resolve_method(b, spin="unpolarized")
    assert ir_a.identity == ir_b.identity
    assert ir_a.manifest_identity != ir_b.manifest_identity
    assert (
        ir_a.primitives[0].functional.components
        == ir_b.primitives[0].functional.components
    )


def test_audited_method_catalog_is_read_only():
    with pytest.raises(TypeError):
        METHOD_CATALOG["PBE"] = MethodSpec("mutated", (("LDA_X", Fraction(1)),))


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_direct_method_ir_canonicalizes_existing_xc_specs(spin):
    """Public primitive construction shares the resolver's semantic identity."""
    catalog = functional("PBE", spin=spin)
    # FunctionalSpec preserves component order and permits inactive components.
    # Neither may fragment MethodIR identity for the same mathematical graph.
    variants = (
        catalog,
        FunctionalSpec("reordered", tuple(reversed(catalog.components)), spin=spin),
        FunctionalSpec(
            "with-inactive-lda",
            (*catalog.components, ("LDA_X", Fraction(0))),
            spin=spin,
        ),
    )
    expected = resolve_method("PBE", spin=spin)
    for spec in variants:
        primitive = SemilocalXCPrimitive(spec)
        direct = MethodIR("PBE", spin, (primitive,))
        assert direct.identity == expected.identity
        assert (
            primitive.functional.components
            == expected.primitives[0].functional.components
        )
        assert primitive.functional.identifier == spec.identifier
    assert catalog.components[0][0] == "GGA_X_PBE"


def test_same_family_extension_is_data_only_and_json_serializable():
    custom = MethodSpec(
        "PBE-X-half",
        (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
    )
    resolved = resolve_method(custom, spin="polarized")
    assert resolved.primitives[0].functional.components == (
        ("GGA_C_PBE", Fraction(1)),
        ("GGA_X_PBE", Fraction(1, 2)),
    )
    payload = resolved.to_payload()
    assert json.loads(json.dumps(payload, sort_keys=True))["identifier"] == "PBE-X-half"


def test_representation_does_not_hide_exchange_inside_semilocal_functional():
    pbe0 = resolve_method("PBE0")
    semilocal = pbe0.primitives[0].functional
    assert semilocal.exact_exchange == 0
    assert semilocal.range_omega == 0
    assert semilocal.long_range_exchange == 0
    assert isinstance(pbe0.primitives[1], ExactExchangePrimitive)


@pytest.mark.parametrize("spin", ["closed", "rks", ""])
def test_bad_spin_fails_before_graph_construction(spin):
    with pytest.raises(UnsupportedMethod, match="spin mode"):
        resolve_method("PBE", spin=spin)


def test_unknown_or_ambiguous_compositions_fail_closed():
    with pytest.raises(UnsupportedMethod, match="unknown DFT method"):
        resolve_method("pbe")
    with pytest.raises(UnsupportedMethod, match="unsupported semilocal component"):
        MethodSpec("bad", (("NOT_A_COMPONENT", Fraction(1)),))
    with pytest.raises(UnsupportedMethod, match="exact Fraction"):
        MethodSpec("float-weight", (("GGA_X_PBE", 0.75),))
    with pytest.raises(UnsupportedMethod, match="zero-valued"):
        MethodSpec("zero", (("GGA_X_PBE", Fraction(0)),))


def test_exact_cancellation_is_canonical_and_cannot_make_an_empty_method():
    mixed = MethodSpec(
        "exchange-only-after-cancel",
        (("LDA_X", Fraction(1)), ("LDA_X", Fraction(-1))),
        exact_exchange=Fraction(1, 3),
    )
    resolved = resolve_method(mixed)
    assert len(resolved.primitives) == 1
    assert isinstance(resolved.primitives[0], ExactExchangePrimitive)

    empty = MethodSpec(
        "empty-after-cancel",
        (("LDA_X", Fraction(1)), ("LDA_X", Fraction(-1))),
    )
    with pytest.raises(UnsupportedMethod, match="empty graph"):
        resolve_method(empty)


def test_original_nonlocal_variants_are_versioned_and_semantically_distinct():
    vv10 = original_nonlocal_correlation("vv10")
    rvv10 = original_nonlocal_correlation("rvv10")
    assert vv10.b == Fraction("5.9")
    assert rvv10.b == Fraction("6.3")
    assert vv10.c == rvv10.c == Fraction("0.0093")
    assert vv10.identity != rvv10.identity
    payload = vv10.to_payload()
    assert payload["kernel_convention"] == "finite-system-real-space-total-density-v1"
    assert payload["quadrature"] == "real-space-weighted-point-pairs-v1"
    assert payload["regularization"] == "none-positive-density-domain-v1"
    assert payload["pair_integration"] == "full-double-integral-with-one-half-v1"


def test_nonlocal_primitive_participates_in_method_identity_and_requirements():
    vv10 = original_nonlocal_correlation("vv10")
    spec = MethodSpec(
        "PBE+VV10-test",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
        nonlocal_correlation=vv10,
    )
    resolved = resolve_method(spec, spin="polarized")
    assert isinstance(resolved.primitives[-1], NonlocalCorrelationPrimitive)
    assert resolved.primitives[-1].derivative_capabilities == (
        "energy",
        "ks-potential",
    )
    assert resolved.requirements["ingredients"] == ("rho", "sigma")
    assert resolved.requirements["operators"] == (
        "semilocal-xc",
        "nonlocal-correlation",
    )


def test_nonlocal_variant_and_parameters_change_method_identity():
    original = original_nonlocal_correlation("vv10")
    custom = NonlocalCorrelationSpec("vv10", Fraction("6.0"), Fraction("0.0093"))
    a = resolve_method(MethodSpec("a", (), nonlocal_correlation=original))
    b = resolve_method(MethodSpec("b", (), nonlocal_correlation=custom))
    assert a.identity != b.identity
    assert a.manifest_identity != b.manifest_identity


def test_nonlocal_method_spec_rejects_untyped_or_ambiguous_parameters():
    with pytest.raises(TypeError, match="NonlocalCorrelationSpec"):
        MethodSpec("bad-nlc", (), nonlocal_correlation="vv10")
    with pytest.raises(ValueError, match="exact Fraction"):
        NonlocalCorrelationSpec("vv10", 5.9, Fraction("0.0093"))
    with pytest.raises(ValueError, match="unsupported nonlocal-correlation variant"):
        NonlocalCorrelationSpec("not-vv10", Fraction("5.9"), Fraction("0.0093"))

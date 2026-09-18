"""Canonical DFT MethodSpec -> MethodIR composition gates for #396."""

import json
from fractions import Fraction

import pytest
from vibeqc_compiler.method import (
    METHOD_CATALOG,
    ExactExchangePrimitive,
    MethodIR,
    MethodSpec,
    SemilocalXCPrimitive,
    UnsupportedMethod,
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

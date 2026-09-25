"""Executable point bindings cannot retain mutable or non-string identities."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from vibeqc_compiler.xc.bulk_aot import SourceVariant
from vibeqc_compiler.xc.bulk_point_program import SemilocalPointBinding


def variant() -> SourceVariant:
    return SourceVariant(
        name="synthetic-lda",
        family="LDA",
        spin="polarized",
        import_identity="source-identity",
        domain="synthetic-contract",
        features=("rho_a", "rho_b"),
        derivative_order=1,
        backend="cpu",
        source="void bulk_xc_point(const double* x, double* y) { y[0]=x[0]; y[1]=1; y[2]=0; }\n",
        energy_nodes=1,
        ssa={},
    )


@pytest.mark.parametrize("field", ["capability_identity", "point_expression_identity"])
@pytest.mark.parametrize("value", [True, 7, ["identity"], {"identity": "value"}, " "])
def test_binding_identities_are_nonempty_strings(field: str, value: object) -> None:
    arguments = {
        "capability_identity": "capability",
        "point_expression_identity": "expression",
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        SemilocalPointBinding(variant(), domain_version=1, **arguments)


@pytest.mark.parametrize("field", ["name", "import_identity", "domain", "source"])
@pytest.mark.parametrize("value", [True, ["identity"], {"identity": "value"}, " "])
def test_binding_rejects_mutable_variant_identity(field: str, value: object) -> None:
    source = replace(variant(), **{field: value})
    with pytest.raises(ValueError, match=field):
        SemilocalPointBinding(source, "capability", "expression", 1)


def test_binding_requires_an_actual_source_variant() -> None:
    source = variant()
    duck = SimpleNamespace(**vars(source))
    with pytest.raises(TypeError, match="SourceVariant"):
        SemilocalPointBinding(duck, "capability", "expression", 1)


@pytest.mark.parametrize(
    "features,mask",
    [
        (("rho_a", "rho_b"), 1),
        (("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"), 7),
        (("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb", "tau_a", "tau_b"), 15),
    ],
)
def test_valid_binding_payload_is_detached_and_emission_is_stable(
    features: tuple[str, ...], mask: int
) -> None:
    binding = SemilocalPointBinding(
        replace(variant(), features=features), "capability", "expression", 1
    )
    before = binding.identity, binding.emit_source()
    payload = binding.to_payload()
    payload["capability_identity"] = "other"
    payload["features"].append("not-an-input")
    assert binding.ingredient_mask == mask
    assert (binding.identity, binding.emit_source()) == before
    assert binding.variant.emission_identity in before[1]

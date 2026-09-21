"""Compiler-only gates for generic MethodIR-derived DFT HVP planning (#180)."""

import typing
from dataclasses import replace
from fractions import Fraction

import pytest
from vibeqc_compiler.method import (
    MethodSpec,
    SemilocalXCPrimitive,
    StationaryHVPPlan,
    StationaryMeanField,
    UnsupportedMethod,
    resolve_method,
)
from vibeqc_compiler.method.stationary_gradient import SCF_POINT_MODEL


def plan(
    method: typing.Any = "PBE", spin: typing.Any = "unpolarized"
) -> StationaryHVPPlan:
    return StationaryHVPPlan(
        resolve_method(method, spin=spin),
        StationaryMeanField(SCF_POINT_MODEL),
    )


def test_lda_gga_and_same_family_extension_share_one_hvp_source_topology() -> None:
    lda = plan("LDA_XC_PW")
    pbe = plan("PBE")
    custom = plan(
        MethodSpec(
            "custom-gga-hvp",
            (
                ("GGA_X_PBE", Fraction(1)),
                ("GGA_C_PBE", Fraction(1, 2)),
            ),
        )
    )

    assert lda.active_ingredients == ("rho",)
    assert pbe.active_ingredients == ("rho", "sigma")
    assert custom.active_ingredients == pbe.active_ingredients
    assert lda.source_names == pbe.source_names == custom.source_names
    assert lda.source_names == (
        "one_electron",
        "coulomb",
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "overlap_pulay",
        "nuclear",
    )
    assert pbe.identity != lda.identity
    assert custom.identity != pbe.identity


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_plan_binds_shared_cpks_and_second_order_primitive_contracts(
    spin: typing.Any,
) -> None:
    p = plan("PBE", spin)
    assert p.response_contract == {
        "solver": "shared-native-cpks-issue-179",
        "xc_action": "feature-hessian-times-direction",
        "integral_second_order": "issue-178-weighted-hvp",
        "assembly": "matrix-free-directional",
    }
    source = {item.name: item for item in p.sources}
    assert "feature-hessian" in source["xc_ao"].requirements
    assert "feature-hessian" in source["xc_grid"].requirements
    assert "feature-hessian" in source["xc_weight"].requirements
    assert "second-integral-hvp" in source["one_electron"].requirements
    assert "second-integral-hvp" in source["coulomb"].requirements
    assert "second-integral-hvp" in source["overlap_pulay"].requirements


def test_plan_identity_depends_on_semantics_not_method_name() -> None:
    pbe = plan("PBE")
    alias = StationaryHVPPlan(
        replace(pbe.method, identifier="same-pbe-mathematics"),
        pbe.mean_field,
    )
    assert alias.identity == pbe.identity
    assert alias.to_payload() == pbe.to_payload()


@pytest.mark.parametrize("method", ["PBE0", "CAM-B3LYP"])
def test_unqualified_exchange_primitives_fail_closed(method: typing.Any) -> None:
    with pytest.raises(UnsupportedMethod, match="second-order rule"):
        plan(method)


def test_tau_meta_gga_fails_closed_until_tau_response_is_registered() -> None:
    with pytest.raises(UnsupportedMethod, match="ingredients"):
        plan("R2SCAN")


def test_ecp_and_non_scf_point_models_do_not_inherit_hessian_support() -> None:
    method = resolve_method("PBE")
    with pytest.raises(UnsupportedMethod, match="all-electron"):
        StationaryHVPPlan(
            method,
            StationaryMeanField(
                SCF_POINT_MODEL,
                hamiltonian="scalar-semilocal-ecp",
            ),
        )
    with pytest.raises(UnsupportedMethod, match="SCF point model"):
        StationaryHVPPlan(method, StationaryMeanField("interior-v1"))


def test_missing_feature_hessian_rule_rejects_every_semilocal_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SemilocalXCPrimitive,
        "derivative_capabilities",
        property(lambda self: ("energy-density", "feature-gradient")),
    )
    with pytest.raises(UnsupportedMethod, match="second-order feature derivative"):
        plan("PBE")


def test_compiler_plan_does_not_publish_native_hvp_capability() -> None:
    p = plan("PBE")
    for backend in ("cpu", "cuda"):
        with pytest.raises(NotImplementedError, match="qualification"):
            p.require_native_endpoint(backend)
    with pytest.raises(ValueError, match="backend"):
        p.require_native_endpoint("fallback")


def test_payload_is_explicit_about_capability_and_active_ingredients() -> None:
    p = plan("PBE", "polarized")
    payload = p.to_payload()
    assert payload["schema"] == "stationary-hvp-plan-v1"
    assert payload["active_ingredients"] == ["rho", "sigma"]
    assert payload["method"]["spin"] == "polarized"
    assert payload["mean_field"]["coulomb"] == "direct-full-range"
    assert "compiler plan only" in payload["capability"]
    assert [source["name"] for source in payload["sources"]] == list(p.source_names)


def test_wrong_input_types_reject_before_capability_inference() -> None:
    with pytest.raises(TypeError, match="resolved MethodIR"):
        StationaryHVPPlan("PBE", StationaryMeanField(SCF_POINT_MODEL))
    with pytest.raises(TypeError, match="mean-field"):
        StationaryHVPPlan(resolve_method("PBE"), "direct")

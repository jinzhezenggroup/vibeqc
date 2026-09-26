"""Compiler-only gates for generic MethodIR-derived DFT HVP planning (#180)."""

import typing
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pytest
import vibeqc_compiler.method.stationary_hvp as stationary_hvp_module
from vibeqc_compiler.method import (
    ExactExchangePrimitive,
    MethodSpec,
    SemilocalXCPrimitive,
    StationaryHVPPlan,
    StationaryMeanField,
    UnsupportedMethod,
    resolve_method,
)
from vibeqc_compiler.method.stationary_gradient import SCF_POINT_MODEL
from vibeqc_compiler.tensor import Program, execute


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


def test_primitive_rule_registration_extends_plan_without_method_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange_source = stationary_hvp_module.HVPSource(
        "exact_exchange",
        "exact_exchange",
        ("density-response", "first-integral-direction", "second-integral-hvp"),
        ("density_left", "density_right"),
    )
    exchange_rule = stationary_hvp_module.HVPPrimitiveRule(
        "test-full-range-exchange-v1",
        ExactExchangePrimitive,
        "exact_exchange",
        ("energy", "fock", "eri-first-derivative"),
        (),
        (exchange_source,),
    )
    monkeypatch.setattr(
        stationary_hvp_module,
        "_PRIMITIVE_HVP_RULES",
        MappingProxyType(
            {
                **dict(stationary_hvp_module._PRIMITIVE_HVP_RULES),
                ExactExchangePrimitive: exchange_rule,
            }
        ),
    )

    hybrid = plan("PBE0")
    assert "exact_exchange" in hybrid.source_names
    assert [rule.identifier for rule in hybrid.primitive_rules] == [
        "semilocal-rho-sigma-v1",
        "test-full-range-exchange-v1",
    ]
    block = hybrid.integral_block("exact_exchange", terms=2)
    assert block.response_inputs == ("density_left", "density_right")


def test_primitive_rule_cannot_collide_with_envelope_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    colliding_rule = stationary_hvp_module.HVPPrimitiveRule(
        "test-colliding-exchange-v1",
        ExactExchangePrimitive,
        "exact_exchange",
        ("energy", "fock", "eri-first-derivative"),
        (),
        (
            stationary_hvp_module.HVPSource(
                "nuclear",
                "exact_exchange",
                ("second-integral-hvp",),
                ("density_left", "density_right"),
            ),
        ),
    )
    monkeypatch.setattr(
        stationary_hvp_module,
        "_PRIMITIVE_HVP_RULES",
        MappingProxyType(
            {
                **dict(stationary_hvp_module._PRIMITIVE_HVP_RULES),
                ExactExchangePrimitive: colliding_rule,
            }
        ),
    )
    with pytest.raises(UnsupportedMethod, match="duplicate source names"):
        plan("PBE0")


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
    with pytest.raises(UnsupportedMethod, match="derivative capabilities"):
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
    assert payload["schema"] == "stationary-hvp-plan-v2"
    assert [rule["identifier"] for rule in payload["primitive_rules"]] == [
        "semilocal-rho-sigma-v1"
    ]
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


def _integral_hvp_fixture(
    source: str, spins: int, terms: int, coordinates: int
) -> typing.Any:
    rng = np.random.default_rng(180)
    density = rng.normal(size=(spins, terms))
    density_direction = rng.normal(size=(spins, terms))
    if source == "one_electron":
        feeds = {
            "density_left": density,
            "d_density_left": density_direction,
        }
        weights = density.sum(axis=0)
        response_weights = density_direction.sum(axis=0)
    elif source == "coulomb":
        left = rng.normal(size=(spins, terms))
        right = rng.normal(size=(spins, terms))
        d_left = rng.normal(size=(spins, terms))
        d_right = rng.normal(size=(spins, terms))
        feeds = {
            "density_left": left,
            "density_right": right,
            "d_density_left": d_left,
            "d_density_right": d_right,
        }
        weights = 0.5 * left.sum(axis=0) * right.sum(axis=0)
        response_weights = 0.5 * (
            d_left.sum(axis=0) * right.sum(axis=0)
            + left.sum(axis=0) * d_right.sum(axis=0)
        )
    else:
        weighted = rng.normal(size=(spins, terms))
        weighted_direction = rng.normal(size=(spins, terms))
        feeds = {
            "weighted_density": weighted,
            "d_weighted_density": weighted_direction,
        }
        weights = -weighted.sum(axis=0)
        response_weights = -weighted_direction.sum(axis=0)
    first = rng.normal(size=(terms, coordinates))
    second = rng.normal(size=coordinates)
    feeds.update({"integral_derivatives": first, "weighted_second_hvp": second})
    return feeds, weights, response_weights


@pytest.mark.parametrize("source", ["one_electron", "coulomb", "overlap_pulay"])
@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_integral_hvp_block_executes_response_and_weighted_second_terms(
    source: str, spin: str
) -> None:
    p = plan("PBE", spin)
    spins = 2 if spin == "polarized" else 1
    feeds, expected_weights, expected_response = _integral_hvp_fixture(
        source, spins, terms=5, coordinates=6
    )
    block = p.integral_block(source, terms=5, coordinates=6)

    weights = execute(block.weights, feeds).outputs["weights"]
    response_weights = execute(block.response_weights, feeds).outputs[
        "response_weights"
    ]
    np.testing.assert_allclose(weights, expected_weights, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(
        response_weights, expected_response, atol=2e-14, rtol=2e-14
    )

    result = execute(
        block.contraction,
        {
            "response_weights": response_weights,
            "integral_derivatives": feeds["integral_derivatives"],
            "weighted_second_hvp": feeds["weighted_second_hvp"],
        },
    ).outputs
    expected_response_term = expected_response @ feeds["integral_derivatives"]
    expected_second_term = feeds["weighted_second_hvp"]
    np.testing.assert_allclose(
        result["response"], expected_response_term, atol=2e-13, rtol=2e-13
    )
    np.testing.assert_allclose(
        result["second"], expected_second_term, atol=2e-13, rtol=2e-13
    )
    np.testing.assert_allclose(
        result["hvp"], expected_response_term + expected_second_term, atol=2e-13
    )

    replay = Program.loads(block.contraction.dumps())
    np.testing.assert_array_equal(
        execute(
            replay,
            {
                "response_weights": response_weights,
                "integral_derivatives": feeds["integral_derivatives"],
                "weighted_second_hvp": feeds["weighted_second_hvp"],
            },
        ).outputs["hvp"],
        result["hvp"],
    )


def test_integral_hvp_block_stays_fail_closed_for_geometric_xc_sources() -> None:
    p = plan("PBE")
    for source in ("xc_ao", "xc_grid", "xc_weight", "nuclear"):
        with pytest.raises(ValueError, match="integral primitive"):
            p.integral_block(source, terms=2)
    with pytest.raises(ValueError, match="element budget"):
        p.integral_block("coulomb", terms=64, coordinates=12, max_elements=1)


def test_integral_hvp_contraction_rejects_missing_response_or_second_term() -> None:
    block = plan("PBE").integral_block("one_electron", terms=2, coordinates=3)
    with pytest.raises(ValueError, match="response_weights"):
        execute(
            block.contraction,
            {
                "integral_derivatives": np.ones((2, 3)),
                "weighted_second_hvp": np.ones(3),
            },
        )
    with pytest.raises(ValueError, match="weighted_second_hvp"):
        execute(
            block.contraction,
            {
                "response_weights": np.ones(2),
                "integral_derivatives": np.ones((2, 3)),
            },
        )


@pytest.mark.parametrize("source", ["one_electron", "coulomb", "overlap_pulay"])
@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_integral_hvp_matches_multistep_displaced_gradient_and_detects_omissions(
    source: str, spin: str
) -> None:
    """Differentiate scalar-loop gradients independently of generated AD rules."""
    feeds, _, _ = _integral_hvp_fixture(
        source, 2 if spin == "polarized" else 1, terms=5, coordinates=6
    )
    rng = np.random.default_rng(181)
    second = rng.normal(size=(5, 6))

    def displaced_gradient(step: float) -> np.ndarray:
        result = np.zeros(6)
        for term in range(5):
            name = "weighted_density" if source == "overlap_pulay" else "density_left"
            left = sum(
                float(value + step * direction)
                for value, direction in zip(
                    feeds[name][:, term], feeds["d_" + name][:, term], strict=True
                )
            )
            if source == "coulomb":
                right = sum(
                    float(value + step * direction)
                    for value, direction in zip(
                        feeds["density_right"][:, term],
                        feeds["d_density_right"][:, term],
                        strict=True,
                    )
                )
                weight = 0.5 * left * right
            else:
                weight = -left if source == "overlap_pulay" else left
            for coordinate in range(6):
                result[coordinate] += weight * (
                    feeds["integral_derivatives"][term, coordinate]
                    + step * second[term, coordinate]
                )
        return result

    block = plan("PBE", spin).integral_block(source, terms=5, coordinates=6)
    weights = execute(block.weights, feeds).outputs["weights"]
    response = execute(block.response_weights, feeds).outputs["response_weights"]
    actual = execute(
        block.contraction,
        {
            "response_weights": response,
            "integral_derivatives": feeds["integral_derivatives"],
            "weighted_second_hvp": weights @ second,
        },
    ).outputs
    errors = []
    for step in (1e-3, 2e-4, 4e-5):
        finite = (displaced_gradient(step) - displaced_gradient(-step)) / (2 * step)
        errors.append(np.max(np.abs(actual["hvp"] - finite)))
    assert errors[-1] < 1e-7
    if source == "coulomb":
        assert errors[1] < 0.06 * errors[0]
    # A successful comparison must detect either omitted chain-rule term;
    # merely checking that both named output arrays exist would not do that.
    assert np.max(np.abs(actual["second"] - finite)) > 1e-3
    assert np.max(np.abs(actual["response"] - finite)) > 1e-3

"""Issue #673 Slice D: MethodIR production pruning acceptance gates."""

import pytest
from vibeqc.mean_field import compile_fixed_density_method
from vibeqc_compiler.method import (
    UnsupportedMethod,
    infer_feature_types,
    resolve_method,
)
from vibeqc_compiler.xc.contractions import ContractionProgram


def _feature_names(method: object) -> tuple[str, ...]:
    return tuple(
        feature.ingredient for feature in infer_feature_types(method, dtype="float64")
    )


def _production_contract(name: str) -> tuple[object, object, object]:
    method = resolve_method(name)
    plan = compile_fixed_density_method(method)
    contract = ContractionProgram(plan.functional).contract
    return method, plan, contract


def test_pure_gga_prunes_meta_gga_and_exchange_requirements_end_to_end() -> None:
    method, plan, contract = _production_contract("PBE")

    assert method.requirements["ingredients"] == ("rho", "sigma")
    assert method.requirements["operators"] == ("semilocal-xc",)
    assert _feature_names(method) == ("rho", "sigma")
    assert contract.ingredients.family == "gga"
    assert contract.scalar_outputs == ((), (0,), (1,))
    assert not plan.fock_spec.exchange.present
    assert plan.functional.exact_exchange == 0
    assert plan.functional.long_range_exchange == 0
    assert plan.functional.range_omega == 0
    assert "tau" not in plan.functional.ingredients


def test_global_hybrid_keeps_only_live_full_range_exchange() -> None:
    method, plan, contract = _production_contract("PBE0")

    assert method.requirements["ingredients"] == ("rho", "sigma")
    assert method.requirements["operators"] == (
        "semilocal-xc",
        "full-range-exchange",
    )
    assert _feature_names(method) == ("rho", "sigma")
    assert contract.ingredients.family == "gga"
    assert contract.scalar_outputs == ((), (0,), (1,))
    assert plan.fock_spec.exchange.present
    assert plan.fock_spec.exchange.operator == "full_range"
    assert plan.fock_spec.exchange.coefficient == pytest.approx(-0.125)
    assert plan.functional.exact_exchange == 0
    assert plan.functional.long_range_exchange == 0


def test_meta_gga_retains_tau_as_a_positive_liveness_case() -> None:
    method, plan, contract = _production_contract("R2SCAN")

    assert method.requirements["ingredients"] == ("rho", "sigma", "tau")
    assert method.requirements["operators"] == ("semilocal-xc",)
    assert _feature_names(method) == ("rho", "sigma", "tau")
    assert contract.ingredients.family == "mgga"
    assert contract.scalar_outputs == ((), (0,), (1,), (2,))
    assert "tau" in plan.functional.ingredients
    assert not plan.fock_spec.exchange.present


def test_range_separated_exchange_is_retained_or_fails_closed() -> None:
    method = resolve_method("CAM-B3LYP")

    assert method.requirements["ingredients"] == ("rho", "sigma")
    assert method.requirements["operators"] == (
        "semilocal-xc",
        "short-range-exchange",
        "long-range-exchange",
    )
    assert _feature_names(method) == ("rho", "sigma")
    with pytest.raises(
        UnsupportedMethod,
        match="range_separated_exchange",
    ):
        compile_fixed_density_method(method)

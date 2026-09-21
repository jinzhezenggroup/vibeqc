"""MethodIR-driven self-consistent KS execution-plan gates for #396."""

from fractions import Fraction

import pytest
from vibeqc.ks import ks_coefficients
from vibeqc_compiler.method import (
    MethodIR,
    RangeSeparatedExchangePrimitive,
    compile_ks_execution_plan,
    resolve_method,
)


@pytest.mark.parametrize(
    "method, expected_lowerers",
    (
        ("PBE", ("semilocal-xc",)),
        ("PBE0", ("semilocal-xc", "full-range-exchange")),
        (
            "CAM-B3LYP",
            ("semilocal-xc", "short-range-exchange", "long-range-exchange"),
        ),
        (
            "WB97M-V",
            (
                "semilocal-xc",
                "short-range-exchange",
                "long-range-exchange",
                "nonlocal-correlation",
            ),
        ),
    ),
)
def test_methodir_compiles_one_common_ks_contribution_plan(
    method: str, expected_lowerers: tuple[str, ...]
) -> None:
    plan = compile_ks_execution_plan(resolve_method(method))
    assert plan.self_consistent
    assert plan.required_lowerers == expected_lowerers
    assert plan.method.identity == resolve_method(method).identity
    assert plan.to_payload()["method_identifier"] == method


def test_range_separated_and_nonlocal_parameters_are_derived_from_methodir() -> None:
    cam = compile_ks_execution_plan(resolve_method("CAM-B3LYP"))
    assert tuple(
        (term.operator, term.coefficient, term.omega, term.fock_coefficient)
        for term in cam.exchange
    ) == (
        ("short-range", Fraction(19, 100), Fraction(33, 100), Fraction(-19, 200)),
        ("long-range", Fraction(65, 100), Fraction(33, 100), Fraction(-13, 40)),
    )
    assert cam.nonlocal_correlation is None

    wb97mv = compile_ks_execution_plan(resolve_method("WB97M-V"))
    assert tuple(
        (term.operator, term.coefficient, term.omega, term.fock_coefficient)
        for term in wb97mv.exchange
    ) == (
        ("short-range", Fraction(3, 20), Fraction(3, 10), Fraction(-3, 40)),
        ("long-range", Fraction(1), Fraction(3, 10), Fraction(-1, 2)),
    )
    assert wb97mv.nonlocal_correlation is not None
    assert wb97mv.nonlocal_correlation.spec.variant == "vv10"
    assert wb97mv.nonlocal_correlation.spec.b == Fraction(6)
    assert wb97mv.nonlocal_correlation.spec.c == Fraction(1, 100)


@pytest.mark.parametrize(
    "canonical, alias",
    (("PBE0", "PBE1PBE"), ("CAM-B3LYP", "CAMB3LYP")),
)
def test_execution_identity_uses_semantics_not_method_name(
    canonical: str, alias: str
) -> None:
    first = compile_ks_execution_plan(resolve_method(canonical))
    second = compile_ks_execution_plan(resolve_method(alias))
    assert first.identity == second.identity
    assert (
        first.to_payload()["method_identifier"]
        != second.to_payload()["method_identifier"]
    )


def test_execution_plan_rejects_cross_primitive_omega_drift() -> None:
    wb97mv = resolve_method("WB97M-V")
    semilocal = wb97mv.primitives[0]
    wrong_short = RangeSeparatedExchangePrimitive(
        Fraction(3, 20), Fraction(1, 2), "short-range"
    )
    wrong = MethodIR("wrong-omega", "unpolarized", (semilocal, wrong_short))
    with pytest.raises(ValueError, match="disagree on omega"):
        compile_ks_execution_plan(wrong)


def test_native_v2_projection_fails_by_missing_lowerer_not_named_method() -> None:
    cam = resolve_method("CAM-B3LYP")
    with pytest.raises(NotImplementedError, match="short-range-exchange"):
        ks_coefficients(cam)

    wb97mv = resolve_method("WB97M-V")
    with pytest.raises(NotImplementedError, match="nonlocal-correlation"):
        ks_coefficients(wb97mv)


def test_existing_native_full_range_hybrid_uses_compiled_plan_coefficients() -> None:
    pbe0_rks = resolve_method("PBE0", spin="unpolarized")
    pbe0_uks = resolve_method("PBE0", spin="polarized")
    assert ks_coefficients(pbe0_rks) == (0.75, 1.0, -0.125)
    assert ks_coefficients(pbe0_uks) == (0.75, 1.0, -0.25)

    plan = compile_ks_execution_plan(pbe0_rks)
    assert plan.exchange[0].fock_coefficient == Fraction(-1, 8)
    assert plan.required_lowerers == ("semilocal-xc", "full-range-exchange")

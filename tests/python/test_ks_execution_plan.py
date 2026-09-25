"""MethodIR-driven self-consistent KS execution-plan gates for #396."""

from fractions import Fraction

import pytest
from vibeqc.ks import ks_coefficients, ks_range_exchange_parameters
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


def test_native_projection_carries_wb97mv_range_exchange_from_methodir() -> None:
    cam = resolve_method("CAM-B3LYP")
    with pytest.raises(NotImplementedError, match="semilocal composition"):
        ks_coefficients(cam)

    wb97mv = resolve_method("WB97M-V")
    assert ks_coefficients(wb97mv) == (1.0, 1.0, -0.075)
    assert ks_range_exchange_parameters(wb97mv) == pytest.approx((0.15, 1.0, 0.3))


def test_existing_native_full_range_hybrid_uses_compiled_plan_coefficients() -> None:
    pbe0_rks = resolve_method("PBE0", spin="unpolarized")
    pbe0_uks = resolve_method("PBE0", spin="polarized")
    assert ks_coefficients(pbe0_rks) == (0.75, 1.0, -0.125)
    assert ks_coefficients(pbe0_uks) == (0.75, 1.0, -0.25)

    plan = compile_ks_execution_plan(pbe0_rks)
    assert plan.exchange[0].fock_coefficient == Fraction(-1, 8)
    assert plan.required_lowerers == ("semilocal-xc", "full-range-exchange")


@pytest.mark.parametrize(
    "spin,selector", (("unpolarized", "r2scan-rks"), ("polarized", "r2scan-uks"))
)
def test_wb97mv_internal_projection_preserves_all_primitives_and_domain(
    spin: str, selector: str
) -> None:
    from dataclasses import replace

    from vibeqc.ks import (
        WB97MV_SCF_DOMAIN,
        KsOptions,
        _native_semilocal_family,
        native_ks_options,
        resolve_ks_method,
        resolve_ks_options,
    )
    from vibeqc_compiler.dft.grid import GridSpec

    graph = resolve_method("WB97M-V", spin=spin)
    renamed = replace(graph, identifier="not-a-method-dispatch-key")
    assert _native_semilocal_family(renamed) == 4
    options = resolve_ks_options(
        selector, KsOptions(composition=renamed, grid=GridSpec())
    )
    assert options.scf_domain == WB97MV_SCF_DOMAIN
    assert options.to_payload()["nonlocal_density_policy"] == {
        "version": "vv10-molecular-rho-ge-1e-8-v1",
        "threshold": "1/100000000",
        "active_comparison": ">=",
    }
    from vibeqc import _native

    native = native_ks_options(options)
    assert native.scf_domain.decode() == WB97MV_SCF_DOMAIN
    assert native.spin_channels == (1 if spin == "unpolarized" else 2)
    assert {
        native.semilocal_components[i].component_id.decode()
        for i in range(native.semilocal_component_count)
    } == {"MGGA_X_WB97M_V", "MGGA_C_WB97M_V"}
    assert native.semilocal_range_omega == pytest.approx(0.3)
    terms = {
        native.exchange_terms[i].operator_kind: native.exchange_terms[i]
        for i in range(native.exchange_term_count)
    }
    assert terms[_native.KS_EXCHANGE_SHORT_RANGE].coefficient == pytest.approx(0.15)
    assert terms[_native.KS_EXCHANGE_LONG_RANGE].coefficient == pytest.approx(1.0)
    assert terms[_native.KS_EXCHANGE_SHORT_RANGE].omega == pytest.approx(0.3)
    assert native.has_nonlocal_correlation == 1
    assert native.nonlocal_variant == 1
    assert native.nonlocal_b == pytest.approx(6.0)
    assert native.nonlocal_c == pytest.approx(0.01)
    # The public selector must preserve the spin of this parameterized case.
    public_selector = "wb97m-v" if spin == "unpolarized" else "wb97m-v-uks"
    public_method, public_functional = resolve_ks_method(public_selector)
    assert public_method.semantic_payload() == graph.semantic_payload()
    assert public_functional.spin == spin


def test_wb97mv_internal_projection_rejects_missing_or_changed_contributions() -> None:
    from dataclasses import replace

    from vibeqc.ks import _native_semilocal_family
    from vibeqc_compiler.method import (
        NonlocalCorrelationPrimitive,
        RangeSeparatedExchangePrimitive,
        SemilocalXCPrimitive,
    )

    graph = resolve_method("WB97M-V")
    for drop in range(1, 4):
        missing = replace(
            graph, primitives=graph.primitives[:drop] + graph.primitives[drop + 1 :]
        )
        with pytest.raises(NotImplementedError):
            _native_semilocal_family(missing)
    changed = []
    for primitive in graph.primitives:
        if isinstance(primitive, SemilocalXCPrimitive):
            changed.append(
                replace(
                    primitive,
                    functional=replace(
                        primitive.functional, range_omega=Fraction(2, 5)
                    ),
                )
            )
        elif isinstance(primitive, RangeSeparatedExchangePrimitive):
            changed.append(replace(primitive, omega=Fraction(2, 5)))
        else:
            changed.append(primitive)
    with pytest.raises(NotImplementedError, match="canonical"):
        _native_semilocal_family(replace(graph, primitives=tuple(changed)))
    nonlocal_term = graph.primitives[-1]
    assert isinstance(nonlocal_term, NonlocalCorrelationPrimitive)
    wrong = replace(nonlocal_term, spec=replace(nonlocal_term.spec, b=Fraction(59, 10)))
    with pytest.raises(NotImplementedError, match="canonical"):
        _native_semilocal_family(
            replace(graph, primitives=(*graph.primitives[:-1], wrong))
        )

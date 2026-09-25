"""Source-derived split global-hybrid Graph construction."""

from fractions import Fraction

import pytest
from vibeqc_compiler.xc.libxc_maple import MapleImportError

from tools.libxc_split_hybrid import (
    available_split_global_hybrids,
    build_split_global_hybrid,
)


@pytest.mark.parametrize(
    ("name", "exact", "exchange", "correlation"),
    (
        ("M06-2X", Fraction(27, 50), "HYB_MGGA_X_M06_2X", "MGGA_C_M06_2X"),
        ("MN15", Fraction(11, 25), "HYB_MGGA_X_MN15", "MGGA_C_MN15"),
    ),
)
def test_split_global_hybrid_builds_pinned_semilocal_graphs(
    name: str,
    exact: Fraction,
    exchange: str,
    correlation: str,
) -> None:
    assert name in available_split_global_hybrids()
    program = build_split_global_hybrid(name)
    assert program.exact_exchange == exact
    assert program.exchange_registration == exchange
    assert program.correlation_registration == correlation
    assert program.exchange.family == program.correlation.family == "mgga"
    assert program.exchange.features == program.correlation.features
    assert program.exchange.features == (
        "rho_a",
        "rho_b",
        "sigma_aa",
        "sigma_ab",
        "sigma_bb",
        "lapl_a",
        "lapl_b",
        "tau_a",
        "tau_b",
    )
    assert len(program.exchange.graph.topological_order(program.exchange.roots(1))) > 10
    assert len(program.correlation.graph.topological_order(program.correlation.roots(1))) > 10


def test_split_hybrid_work_policies_follow_component_specific_libxc_thresholds() -> None:
    m062x = build_split_global_hybrid("M06-2X")
    assert m062x.exchange_policy.density_threshold == pytest.approx(1.0e-15)
    assert m062x.correlation_policy.density_threshold == pytest.approx(1.0e-12)
    assert m062x.exchange_policy.needs_tau
    assert m062x.correlation_policy.needs_tau
    assert m062x.exchange_policy.tau_threshold == pytest.approx(1.0e-20)
    assert m062x.correlation_policy.tau_threshold == pytest.approx(1.0e-20)
    assert not m062x.exchange_policy.enforce_fhc
    assert not m062x.correlation_policy.enforce_fhc

    mn15 = build_split_global_hybrid("MN15")
    assert mn15.exchange_policy.density_threshold == pytest.approx(1.0e-15)
    assert mn15.correlation_policy.density_threshold == pytest.approx(1.0e-15)
    assert mn15.exchange_policy.needs_tau
    assert mn15.correlation_policy.needs_tau


def test_split_hybrid_builder_remains_fail_closed_for_unknown_method() -> None:
    with pytest.raises(MapleImportError, match="unknown or unrepresentable"):
        build_split_global_hybrid("NOT-A-METHOD")

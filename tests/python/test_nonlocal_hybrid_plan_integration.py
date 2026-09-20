"""Integrate independent hybrid/nonlocal plans without confusing their nodes."""

from fractions import Fraction

import pytest
from vibeqc_compiler.method import (
    MethodSpec,
    UnsupportedMethod,
    original_nonlocal_correlation,
    resolve_method,
)
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_nonlocal_node_is_not_positional_exact_exchange(spin: str) -> None:
    semilocal = (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1)))
    spec = MethodSpec(
        "PBE+VV10",
        semilocal,
        nonlocal_correlation=original_nonlocal_correlation("vv10"),
    )
    plan = StationaryGradientPlan(
        resolve_method(spec, spin=spin), StationaryMeanField(SCF_POINT_MODEL)
    )
    assert plan.exchange is None
    assert "exact_exchange" not in plan.source_names
    assert {"nonlocal_ao", "nonlocal_grid", "nonlocal_weight"} <= set(plan.source_names)
    hybrid = StationaryGradientPlan(
        resolve_method("PBE0", spin=spin), StationaryMeanField(SCF_POINT_MODEL)
    )
    assert hybrid.exchange.coefficient == Fraction(1, 4)
    assert "exact_exchange" in hybrid.source_names
    assert not any(name.startswith("nonlocal_") for name in hybrid.source_names)
    combined = MethodSpec(
        "PBE0+VV10",
        semilocal,
        exact_exchange=Fraction(1, 4),
        nonlocal_correlation=spec.nonlocal_correlation,
    )
    with pytest.raises(UnsupportedMethod, match="combined hybrid/nonlocal"):
        StationaryGradientPlan(
            resolve_method(combined, spin=spin), StationaryMeanField(SCF_POINT_MODEL)
        )

"""omegaB97M-V stationary CUDA geometry lowering stays on the canonical XC graph."""

from dataclasses import replace
from fractions import Fraction

import pytest
from vibeqc_compiler.method import MethodSpec, resolve_method
from vibeqc_compiler.method.spec import SemilocalXCPrimitive
from vibeqc_compiler.method.stationary_cuda import emit_stationary_cuda
from vibeqc_compiler.method.stationary_gradient import (
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.xc.geometry_cuda import emit_geometry_cuda

WB97MV_SCF_DOMAIN = "libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16"


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_wb97mv_stationary_cuda_uses_methodir_semilocal_graph(spin: str) -> None:
    method = resolve_method("WB97M-V", spin=spin)
    plan = StationaryGradientPlan(
        method,
        StationaryMeanField(WB97MV_SCF_DOMAIN),
    )
    source = emit_stationary_cuda("", functional=4, plan=plan)
    assert "stationary_wb97mv_raw" in source
    assert "kStationaryWb97mvExpressionIdentity" in source
    assert "work_sigma[1] = fmax(-sigma_average" in source
    assert "out.kinetic[0] = 0.5 * raw.feature_derivative[5]" in source
    assert "stationary_functional == 2 || stationary_functional == 4" in source
    assert "r2scan_device" not in source


def test_wb97mv_geometry_lowering_refuses_numeric_selector_without_semantics() -> None:
    with pytest.raises(ValueError, match="FunctionalSpec"):
        emit_geometry_cuda(functional=4)

    method = resolve_method("WB97M-V", spin="polarized")
    semilocal = next(
        primitive.functional
        for primitive in method.primitives
        if type(primitive) is SemilocalXCPrimitive
    )
    first = emit_geometry_cuda(functional=4, semilocal=semilocal)
    second = emit_geometry_cuda(functional=4, semilocal=semilocal)
    assert first == second


def test_rks_spin_projection_preserves_method_owned_parameters() -> None:
    """A named WB97M-V reconstruction would silently erase these parameters."""
    custom = MethodSpec(
        "custom-b97m",
        (("MGGA_X_WB97M_V", Fraction(2, 3)), ("MGGA_C_WB97M_V", Fraction(4, 5))),
        range_omega=Fraction(2, 5),
        short_range_exchange=Fraction(3, 20),
        long_range_exchange=Fraction(1),
    )
    method = resolve_method(custom, spin="unpolarized")
    spec = next(
        item.functional
        for item in method.primitives
        if type(item) is SemilocalXCPrimitive
    )
    plan = StationaryGradientPlan(method, StationaryMeanField(WB97MV_SCF_DOMAIN))
    source = emit_stationary_cuda("", functional=4, plan=plan)
    expected = emit_geometry_cuda(
        functional=4, semilocal=replace(spec, spin="polarized")
    )
    assert expected in source
    assert spec.spin == "unpolarized"

"""Explicit hybrid grids must survive the Calculator ABI compatibility probe."""

import pytest
from vibeqc import Calculator, GridSpec, KsOptions


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks"))
def test_named_hybrid_calculator_accepts_explicit_grid(method: str) -> None:
    grid = GridSpec(radial_points=10, angular_polar=6, angular_azimuth=12)
    calculator = Calculator(method=method, ks_options=KsOptions(grid=grid))
    assert calculator.ks_options.grid == grid
    assert calculator.ks_options.requires_composition_v2


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks"))
def test_named_hybrid_still_requires_explicit_grid(method: str) -> None:
    with pytest.raises(NotImplementedError, match="explicit GridSpec"):
        Calculator(method=method)

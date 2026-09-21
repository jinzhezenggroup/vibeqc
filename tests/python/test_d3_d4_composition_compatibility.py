"""D3 composition must preserve the existing explicit/resolved PBE-D4 seam."""

import pytest
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc_compiler.method import resolve_method


@pytest.mark.parametrize("mode", ("explicit", "resolved"))
def test_existing_d4_options_are_not_intercepted_as_d3(mode: str) -> None:
    graph = resolve_method("PBE-D4(BJ-EEQ-ATM)", spin="unpolarized")
    grid = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
    options = (
        KsOptions(composition=graph, grid=grid)
        if mode == "explicit"
        else Calculator(method="pbe-d4-rks", ks_options=KsOptions(grid=grid)).ks_options
    )
    calculator = Calculator(method="pbe-d4-rks", ks_options=options)
    assert calculator.method_ir.identity == graph.identity
    assert calculator._dispersion_method_ir is None
    assert calculator.ks_options.method_ir.identity == graph.identity


def test_d4_remains_rejected_by_an_uncorrected_pbe_selector() -> None:
    graph = resolve_method("PBE-D4(BJ-EEQ-ATM)", spin="unpolarized")
    with pytest.raises(NotImplementedError):
        Calculator(method="pbe-rks", ks_options=KsOptions(composition=graph))

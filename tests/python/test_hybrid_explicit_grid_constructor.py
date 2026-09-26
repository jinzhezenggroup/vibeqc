"""Explicit hybrid grids require the current semantic KS ABI."""

from types import SimpleNamespace

import pytest
from vibeqc import Calculator, GridSpec, KsOptions, ResourceBudget, _native


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks"))
def test_named_hybrid_calculator_accepts_explicit_grid(method: str) -> None:
    grid = GridSpec(radial_points=10, angular_polar=6, angular_azimuth=12)
    calculator = Calculator(method=method, ks_options=KsOptions(grid=grid))
    assert calculator.ks_options.grid == grid
    assert calculator.ks_options.has_nondefault_composition


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks"))
def test_named_hybrid_still_requires_explicit_grid(method: str) -> None:
    with pytest.raises(NotImplementedError, match="explicit GridSpec"):
        Calculator(method=method)


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks"))
@pytest.mark.parametrize("version", (0, 6))
def test_noncurrent_schema_rejects_explicit_hybrid_without_resolving_default(
    method: str, version: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = SimpleNamespace(vibeqc_ks_options_version=lambda: version)
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)
    with pytest.raises(NotImplementedError, match="semantic KS execution-plan ABI"):
        Calculator(method=method, ks_options=KsOptions(grid=GridSpec()))


@pytest.mark.parametrize(
    "method,charge,multiplicity", (("pbe0-rks", 0, 1), ("pbe0-uks", 1, 2))
)
def test_explicit_hybrid_budgeted_scf_matches_unbudgeted(
    method: str, charge: int, multiplicity: int
) -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    options = KsOptions(
        grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
    )
    ordinary = Calculator(method=method, ks_options=options).singlepoint(
        atoms, charge=charge, multiplicity=multiplicity
    )
    budgeted = Calculator(
        method=method, ks_options=options, resource_budget=ResourceBudget()
    ).singlepoint(atoms, charge=charge, multiplicity=multiplicity)
    assert ordinary.converged and budgeted.converged
    assert budgeted.energy == pytest.approx(ordinary.energy, abs=2e-12)

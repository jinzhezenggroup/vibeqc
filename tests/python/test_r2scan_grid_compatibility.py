"""New LDA/PBE grid defaults must preserve existing r2SCAN admission."""

import pytest
from vibeqc import KsOptions
from vibeqc.ks import resolve_ks_options
from vibeqc_compiler.dft.grid import GridPolicy, GridSpec


@pytest.mark.parametrize("method", ["r2scan-rks", "r2scan-uks"])
def test_r2scan_default_retains_qualified_legacy_grid(method: str) -> None:
    options = resolve_ks_options(method)
    assert options.grid == GridSpec()
    assert options.grid.version == 1


@pytest.mark.parametrize("method", ["r2scan-rks", "r2scan-uks"])
def test_r2scan_does_not_silently_ignore_tight_grid_request(method: str) -> None:
    with pytest.raises(NotImplementedError, match="explicit GridSpec"):
        resolve_ks_options(method, KsOptions(grid_accuracy="tight"))


@pytest.mark.parametrize("method", ["r2scan-rks", "r2scan-uks"])
def test_r2scan_explicit_grid_remains_authoritative(method: str) -> None:
    grid = GridSpec(radial_points=32, angular_polar=12, angular_azimuth=24)
    options = resolve_ks_options(method, KsOptions(grid=grid, grid_accuracy="tight"))
    assert options.grid == grid


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_qualified_families_keep_new_production_defaults(method: str) -> None:
    assert resolve_ks_options(method).grid == GridPolicy().resolve(method)

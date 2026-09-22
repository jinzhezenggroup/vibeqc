from pathlib import Path

"""Full native WB97M-V endpoint, live-state and independent force qualification."""

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
from vibeqc_compiler.dft import NativeAO

ATOMS = [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]
GRID = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)


@pytest.mark.parametrize("method", ["wb97m-v", "wb97m-v-rks", "wb97m-v-uks"])
def test_public_wb97mv_live_state_and_complete_gradient(
    method: str, tmp_path: Path
) -> None:
    calc = Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=180,
    )
    with calc.prepare_batch([ATOMS]) as batch, NativeAO(ATOMS) as basis:
        energy = batch.execute(strict=True).items[0].energy
        state = StationaryKsState.from_native(batch, basis)
        try:
            assert state._source.metadata[6:8] == (4, 3)
            grad = complete_rks_gradient_diagnostic(
                state, basis, cache=tmp_path, execution="native"
            )
            assert np.isfinite(energy)
            assert np.isfinite(grad.gradient).all()
            assert set(grad.components) >= {
                "exchange_short_range",
                "exchange_long_range",
                "nonlocal_ao",
                "nonlocal_grid",
                "nonlocal_weight",
            }
            np.testing.assert_allclose(grad.gradient.sum(axis=0), 0, atol=2e-8)
        finally:
            state._source.close()

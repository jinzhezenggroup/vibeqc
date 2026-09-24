"""Full native WB97M-V endpoint, live-state and independent force qualification."""

import json
from pathlib import Path

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


@pytest.mark.parametrize(
    "method,multiplicity,atoms",
    [
        (
            "wb97m-v-rks",
            1,
            [
                ("O", (0.02, -0.03, 0.04)),
                ("H", (0.10, 1.43, 1.10)),
                ("H", (-0.15, -1.45, 1.12)),
            ],
        ),
        (
            "wb97m-v-uks",
            2,
            [("O", (0.02, -0.03, 0.04)), ("H", (0.21, 0.31, 1.78))],
        ),
    ],
    ids=("water-rks", "oh-doublet-uks"),
)
@pytest.mark.parametrize("grid_shape", [(12, 4, 8), (16, 6, 12)])
def test_public_wb97mv_force_matches_reconverged_energy_differences(
    method: str,
    multiplicity: int,
    atoms: list,
    grid_shape: tuple[int, int, int],
    tmp_path: Path,
) -> None:
    """Differentiate complete public energies, not the component force algebra."""
    radial, polar, azimuth = grid_shape
    calc = Calculator(
        method=method,
        basis="sto-3g",
        device="cpu",
        ks_options=KsOptions(
            grid=GridSpec(
                radial_points=radial, angular_polar=polar, angular_azimuth=azimuth
            )
        ),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    center = calc.singlepoint(
        atoms, multiplicity=multiplicity, properties=("energy", "forces")
    )
    assert center.converged and center.forces is not None
    forces = np.asarray(center.forces)
    assert np.isfinite(center.energy) and np.isfinite(forces).all()
    np.testing.assert_allclose(forces.sum(axis=0), 0, atol=2e-8, rtol=0)

    direction = np.random.default_rng(9351071).normal(size=(len(atoms), 3))
    direction -= direction.mean(axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    analytic = -float(np.vdot(forces, direction))
    samples = []
    for step in (1e-3, 3e-4, 1e-4):
        energies = []
        for sign in (-1.0, 1.0):
            displaced = [
                (symbol, (np.asarray(xyz) + sign * step * delta).tolist())
                for (symbol, xyz), delta in zip(atoms, direction, strict=True)
            ]
            # Each singlepoint prepares and reconverges the complete displaced
            # Hamiltonian. No fixed density or component-gradient oracle is used.
            result = calc.singlepoint(
                displaced, multiplicity=multiplicity, properties=("energy",)
            )
            assert result.converged and np.isfinite(result.energy)
            energies.append(float(result.energy))
        finite = (energies[1] - energies[0]) / (2 * step)
        samples.append(
            {
                "step": step,
                "energies": energies,
                "derivative": finite,
                "absolute_error": abs(finite - analytic),
            }
        )
    report = {
        "method": method,
        "multiplicity": multiplicity,
        "atoms_bohr": atoms,
        "grid_shape": grid_shape,
        "direction": direction.tolist(),
        "analytic_directional_derivative": analytic,
        "samples": samples,
    }
    (tmp_path / "wb97mv-force-finite-difference.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Keep the entire step sequence; accepting only its best point could hide
    # cancellation or a discontinuous density-domain response.
    assert samples[0]["absolute_error"] < 1e-5, report
    assert max(sample["absolute_error"] for sample in samples[1:]) < 2e-6, report

"""Composite CPU energy/replay and correction gradients retain their scopes."""

import numpy as np
import pytest
from vibeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    evaluate_r2scan3c_correction,
    load_r2scan3c_basis,
)
from vibeqc_compiler.method import resolve_method


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_composite_cpu_energy_and_replay_preserve_both_spin_layouts(spin: str) -> None:
    xyz = np.array([[0.11, -0.07, -0.71], [0.04, 0.08, 0.76]])
    atoms = [("H", tuple(row)) for row in xyz]
    graph = resolve_method("R2SCAN-3c", spin=spin)
    controls = {
        "device": "cpu",
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "max_iterations": 200,
    }
    plain = Calculator(
        method="r2scan-rks" if spin == "unpolarized" else "r2scan-uks",
        basis=load_r2scan3c_basis(),
        **controls,
    )
    composite = Calculator(method=graph, **controls)
    reference = plain.singlepoint(atoms, properties=("energy",))
    total = composite.singlepoint(atoms, properties=("energy",))
    correction = evaluate_r2scan3c_correction(graph, [1, 1], xyz, gradients=False)
    assert reference.converged and total.converged
    assert total.energy == pytest.approx(
        reference.energy + correction.energy, abs=2e-11
    )
    assert total.forces is None
    # A composite must not silently widen the underlying CPU force capability.
    for calculator in (plain, composite):
        with pytest.raises(ValueError, match="does not support properties: forces"):
            calculator.singlepoint(atoms, properties=("energy", "forces"))
    moved = xyz.copy()
    moved[1] += [0.02, -0.03, 0.06]
    with composite.prepare_batch([atoms, atoms]) as batch:
        batch.execute(strict=True, properties=("energy",))
        replay = batch.execute(
            coordinates=[moved, None], strict=True, properties=("energy",)
        )
        fresh = composite.singlepoint(
            [("H", tuple(row)) for row in moved], properties=("energy",)
        )
        assert replay.items[0].energy == pytest.approx(fresh.energy, abs=2e-10)
        failed = batch.execute(coordinates=[[0.0], None], properties=("energy",))
        assert not failed.items[0].succeeded and failed.items[0].forces is None
        assert failed.items[1].succeeded
        restored = batch.execute(
            coordinates=[xyz, None], strict=True, properties=("energy",)
        )
        assert restored.items[0].energy == pytest.approx(total.energy, abs=2e-10)


@pytest.mark.parametrize("charge", [0.0, 1.0])
def test_complete_correction_gradient_matches_two_step_energy_difference(
    charge: float,
) -> None:
    xyz = np.array([[0.11, -0.07, 0.0], [1.42, 0.07, 1.18], [-1.32, -0.11, 1.21]])
    direction = np.array([[0.2, -0.13, 0.07], [-0.11, 0.08, 0.19], [0.04, 0.09, -0.17]])
    graph = resolve_method("R2SCAN-3c")
    result = evaluate_r2scan3c_correction(graph, [8, 1, 1], xyz, total_charge=charge)
    assert result.ok and result.d4.ok and result.gcp.ok
    assert abs(result.d4.atm_energy) > 0
    np.testing.assert_allclose(
        result.gradient, result.d4.gradient + result.gcp.gradient, atol=2e-14, rtol=0
    )
    np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=2e-10, rtol=0)
    for step in (3e-4, 1e-4):
        energies = [
            evaluate_r2scan3c_correction(
                graph,
                [8, 1, 1],
                xyz + sign * step * direction,
                total_charge=charge,
                gradients=False,
            ).energy
            for sign in (1, -1)
        ]
        derivative = (energies[0] - energies[1]) / (2 * step)
        assert abs(derivative - np.sum(result.gradient * direction)) < 2e-8

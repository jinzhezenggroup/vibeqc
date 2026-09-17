"""Physical direct-UHF force convergence against retained independent CPU data."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


def test_localized_uhf_force_error_survives_small_density_rms():
    """Cold OH can satisfy density RMS while its transverse force is still bad.

    The CPU fixture fixes the predeclared tolerances and both geometries. Frozen
    replays test the force determinant's returned seed; property changes must
    continue to bypass force-only finalization for energy-only requests.
    """
    assert os.environ.get("SLURM_JOB_ID")
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "data/direct_force_oh.json").read_text()
    )
    calculator = Calculator(
        method="uhf",
        basis=fixture["basis"],
        basis_representation=fixture["representation"],
        device="cuda",
        density_fitting="none",
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )

    def compare(result, references, forces=True):
        assert all(x.converged and x.executed_backend == "cuda" for x in result.items)
        np.testing.assert_allclose(
            result.energies,
            [x["energy"] for x in references],
            atol=fixture["energy_gate"],
            rtol=0,
        )
        if forces:
            np.testing.assert_allclose(
                [x.forces for x in result.items],
                [x["forces"] for x in references],
                atol=fixture["force_gate"],
                rtol=0,
            )
        else:
            assert all(x.forces is None for x in result.items)

    size = len(fixture["atoms"])
    with calculator.prepare_batch(
        fixture["atoms"],
        charges=[fixture["charge"]] * size,
        multiplicities=[fixture["multiplicity"]] * size,
        warm_start=True,
    ) as batch:
        compare(batch.execute(strict=True), fixture["oracles"])
        batch.set_warm_start_updates(False)
        for _ in range(3):
            compare(batch.execute(strict=True), fixture["oracles"])
        coordinates = [[r for _, r in atoms] for atoms in fixture["changed_atoms"]]
        compare(batch.execute(coordinates, strict=True), fixture["changed_oracles"])
        compare(
            batch.execute(coordinates, strict=True, properties=("energy",)),
            fixture["changed_oracles"],
            forces=False,
        )

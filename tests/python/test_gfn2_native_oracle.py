"""Complete native endpoints against independent molecular tblite fixtures.

The fixtures include Cl/Si d shells; they qualify more than the scalar H0
lowering alone. Production never imports tblite to execute these tests.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator

FIXTURE = Path(__file__).resolve().parents[1] / "data/gfn2_native_tblite.json"
CASES = json.loads(FIXTURE.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_gfn2_native_energy_force_against_tblite(case: dict, device: str) -> None:
    if device == "cuda":
        if os.environ.get("VIBEQC_TEST_GFN2_CUDA") != "1":
            pytest.skip("explicit GFN2 CUDA qualification is disabled")
        if not os.environ.get("SLURM_JOB_ID"):
            pytest.fail("GFN2 CUDA qualification requires a Slurm allocation")
    calculator = Calculator(
        method="gfn2-xtb",
        device=device,
        max_iterations=300,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    atoms = list(zip(case["symbols"], case["positions"], strict=True))
    result = calculator.singlepoint(atoms)
    assert result.converged
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert result.energy == pytest.approx(case["energy"], rel=0, abs=5e-7)
    np.testing.assert_allclose(result.forces, case["forces"], rtol=0, atol=5e-7)
    np.testing.assert_allclose(np.sum(result.forces, axis=0), 0, rtol=0, atol=1e-10)

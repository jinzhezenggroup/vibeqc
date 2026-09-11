"""Real CUDA target solves after explicit native-CPU overlap/projection setup."""

import os

import numpy as np
import pytest
from vibeqc import Calculator, projected_singlepoint

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_PROJECTION_CUDA_TEST") != "1",
    reason="explicit Slurm projection GPU tier",
)


@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
@pytest.mark.parametrize("fitted", [False, True])
def test_cuda_target_converges_after_cpu_metric_projection(
    method, charge, multiplicity, fitted
):
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]
    settings = {
        "method": method,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-14,
    }
    source = Calculator(device="cuda", basis="sto-3g", **settings)
    target = Calculator(
        device="cuda",
        basis="def2-svp",
        density_fitting="cuda" if fitted else "none",
        **settings,
    )
    result = projected_singlepoint(
        target, source, atoms, charge=charge, multiplicity=multiplicity
    )
    reference = Calculator(
        basis="def2-svp", density_fitting="cpu" if fitted else "none", **settings
    ).singlepoint(atoms, charge=charge, multiplicity=multiplicity)
    assert result.source.executed_backend == result.target.executed_backend == "cuda"
    assert result.target.restart_origin == "basis_projection"
    assert result.target.fock_builds is None
    np.testing.assert_allclose(
        result.target.energy, reference.energy, atol=3e-10, rtol=0
    )
    np.testing.assert_allclose(
        result.target.forces, reference.forces, atol=3e-9, rtol=0
    )


def test_cuda_batch_projection_and_moved_geometry_are_isolated():
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [
        [("H", (0, 0, -0.7)), ("H", (0.1 * i, 0, 0.7 + 0.02 * i))] for i in range(3)
    ]
    source_calc = Calculator(device="cuda")
    target_calc = Calculator(
        device="cuda", basis="def2-svp", energy_tolerance=1e-12, density_tolerance=1e-10
    )
    with (
        source_calc.prepare_batch(atoms) as source,
        target_calc.prepare_batch(atoms) as target,
    ):
        source.execute(strict=True)
        report = target.initialize_from(source)
        assert all(r["accepted"] for r in report["items"])
        target.execute(strict=True)
        moved = [np.array([r for _, r in item]) for item in atoms]
        moved[1][1, 0] += 0.03
        actual = target.execute(moved, strict=True)
        for i, positions in enumerate(moved):
            expected = target_calc.singlepoint([("H", tuple(r)) for r in positions])
            np.testing.assert_allclose(
                actual.items[i].energy, expected.energy, atol=3e-10, rtol=0
            )
            np.testing.assert_allclose(
                actual.items[i].forces, expected.forces, atol=3e-9, rtol=0
            )

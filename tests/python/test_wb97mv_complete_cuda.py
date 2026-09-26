"""Complete public CUDA energy/force gates against independent GPU4PySCF.

Run VIBEQC_TEST_WB97MV_CUDA=1 through Slurm main with one 5090 and a finite
time limit. Grid motion, partition response, SR/LR exchange and self-consistent
VV10 are all included in both engines; no component-only success promotes API.
"""

import os
import typing

import numpy as np
import pytest


@pytest.mark.parametrize(
    "method,spin,atoms,basis",
    [
        ("wb97m-v", 0, [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))], "sto-3g"),
        (
            "wb97m-v-uks",
            1,
            [
                ("H", (0.0, 0.0, 0.0)),
                ("H", (0.15, 0.13, 1.5)),
                ("H", (1.8, -0.1, -0.3)),
            ],
            "sto-3g",
        ),
        (
            "wb97m-v",
            0,
            [
                ("O", (0.02, -0.03, 0.04)),
                ("H", (0.1, 1.43, 1.1)),
                ("H", (-0.15, -1.45, 1.12)),
            ],
            "def2-svp",
        ),
    ],
)
def test_complete_cuda_force_matches_independent_engine(
    method: str, spin: int, atoms: typing.Any, basis: str
) -> None:
    if os.environ.get("VIBEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_WB97MV_CUDA=1 inside Slurm")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    import cupy as cp
    from vibeqc import Calculator, GridSpec, KsOptions

    from benchmarks.readme_wb97mv import reference_engine, reference_sample

    grid = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
    calc = Calculator(
        method=method,
        basis=basis,
        basis_representation="spherical",
        device="cuda",
        ks_options=KsOptions(grid=grid),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=200,
    )
    with calc.prepare_batch(
        [atoms], multiplicities=[spin + 1], warm_start=True
    ) as batch:
        cold = batch.execute(strict=True).items[0]
        warm = batch.execute(strict=True).items[0]
        assert warm.iterations == 1
        work = batch._stationary_cuda_execution.last_work
        assert work["prepared_execution_reused"]
        assert work["nonlocal_pair_evaluations"] > 0
        assert len(work["source_names"]) == 12
        energy_only = batch.execute(strict=True, properties=("energy",)).items[0]
        assert cold.executed_backend == warm.executed_backend == "cuda"
        assert energy_only.forces is None
        np.testing.assert_allclose(warm.forces, cold.forces, atol=2e-8, rtol=0)
        assert abs(energy_only.energy - cold.energy) < 1e-9
    oracle = reference_sample(reference_engine(atoms, basis, grid, spin=spin), cp)
    assert abs(cold.energy - oracle["energies_hartree"][0]) < 1e-8
    np.testing.assert_allclose(
        cold.forces, oracle["forces_hartree_per_bohr"][0], atol=1e-7, rtol=1e-7
    )
    np.testing.assert_allclose(cold.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
    direction = np.random.default_rng(13421389).normal(size=(len(atoms), 3))
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction)
    analytic = -np.vdot(cold.forces, direction)
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        energies = []
        for sign in (-1, 1):
            displaced = [
                (symbol, np.asarray(xyz) + sign * step * d)
                for (symbol, xyz), d in zip(atoms, direction, strict=True)
            ]
            result = calc.singlepoint(
                displaced, multiplicity=spin + 1, properties=("energy",)
            )
            assert result.converged
            energies.append(result.energy)
        errors.append(abs((energies[1] - energies[0]) / (2 * step) - analytic))
    assert errors[0] < 1e-5 and max(errors[1:]) < 2e-6, errors


def test_cuda_force_rebuild_failure_isolation_and_stale_snapshot() -> None:
    """Rebuild real owners, reject old SCF generations, then recover cleanly."""
    if os.environ.get("VIBEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_WB97MV_CUDA=1 inside Slurm")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    from vibeqc import Calculator, GridSpec, KsOptions
    from vibeqc._dft_gradient import StationaryKsState
    from vibeqc_compiler.dft import NativeAO

    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]
    calc = Calculator(
        method="wb97m-v",
        basis="sto-3g",
        device="cuda",
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=200,
    )
    with calc.prepare_batch([atoms, atoms], warm_start=True) as batch:
        cold = batch.execute(strict=True)
        xyz = np.asarray([p for _, p in atoms])
        displaced = xyz.copy()
        displaced[1, 0] += 0.02
        changed = batch.execute(coordinates=[displaced, xyz], strict=True)
        fresh = calc.singlepoint([("H", p) for p in displaced])
        np.testing.assert_allclose(
            changed.items[0].forces, fresh.forces, atol=2e-8, rtol=0
        )
        np.testing.assert_allclose(
            changed.items[1].forces, cold.items[1].forces, atol=2e-8, rtol=0
        )
        failed = batch.execute(coordinates=[[0.0], xyz], strict=False)
        assert not failed.items[0].succeeded and failed.items[1].succeeded
        assert np.isfinite(failed.items[1].forces).all()
        batch.execute(coordinates=[xyz, xyz], strict=True)
        with NativeAO(atoms, basis="sto-3g", representation="cartesian") as basis:
            state = StationaryKsState.from_native(batch, basis, index=0)
            try:
                batch.execute(strict=True, properties=("energy",))
                # An old snapshot must fail before returning any derivative;
                # the next public call must rebuild the discarded scratch.
                with pytest.raises(
                    (RuntimeError, ValueError), match="stale|current|generation"
                ):
                    batch._stationary_cuda_execution.execute(
                        state,
                        basis,
                        compiler=batch._stationary_cuda_compiler(),
                        cache=".cache/stationary-cuda",
                        library=calc._library._name,
                    )
            finally:
                state._source.close()
        recovered = batch.execute(strict=True)
        np.testing.assert_allclose(
            recovered.items[0].forces, cold.items[0].forces, atol=2e-8, rtol=0
        )

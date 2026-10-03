"""Independent complete PBE0 force/FD gates for the optional AO product screen.

Use the retained full def2-SVP H/O snapshot, including p/d functions, rather
than an s-only fixture that never reaches the refined high-order force path.
These are correctness controls, not endpoint performance measurements.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_FORCE_AO_DENSITY_CUDA_TEST") != "1",
    reason="explicit finite Slurm force-screening qualification",
)


@pytest.mark.parametrize("enabled", ("0", "1"))
@pytest.mark.parametrize("spin", ("rks", "uks"))
def test_high_order_force_screening_independent_endpoint_and_fd(
    enabled: str, spin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check both spin contractions, independent gradients and two FD steps."""
    from generativeqc import Calculator, GridSpec, KsOptions, load_basis
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import independent_global_hybrid_gradient
    from test_dft_complete_cuda import no_cpu_derivatives

    assert os.environ.get("SLURM_JOB_ID"), "real GPU qualification requires Slurm"
    monkeypatch.setenv("GENERATIVEQC_BOUNDED_FORCE_AO_DENSITY", enabled)
    atoms = [
        ("O", (0.13, -0.21, 0.0)),
        ("H", (-1.4, 0.2, 1.1)),
        ("H", (1.55, -0.12, 1.15)),
    ]
    multiplicity = 1 if spin == "rks" else 2
    charge = 0 if spin == "rks" else 1
    basis = load_basis(
        ROOT / "benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json"
    )
    method = f"pbe0-{spin}"
    calculator = Calculator(
        method=method,
        basis=basis,
        basis_representation="spherical",
        device="cuda",
        precision="fp64",
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        screening_tolerance=1e-12,
        max_iterations=200,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=48, angular_polar=16, angular_azimuth=32)
        ),
    )
    with calculator.prepare_batch(
        [atoms], charges=[charge], multiplicities=[multiplicity], warm_start=True
    ) as batch:
        with no_cpu_derivatives():
            point = batch.execute(properties=("energy", "forces"), strict=True).items[0]
        with NativeAO(
            atoms,
            basis=basis,
            representation="spherical",
            charge=charge,
            multiplicity=multiplicity,
        ) as ao:
            state = StationaryKsState.from_native(batch, ao)
            try:
                reference_energy, reference_gradient = (
                    independent_global_hybrid_gradient(ao, state, method, cart=False)
                )
            finally:
                state._source.close()
    assert point.converged and point.executed_backend == "cuda"
    energy_error = abs(point.energy - reference_energy)
    force_error = float(np.max(np.abs(point.forces + reference_gradient)))
    assert energy_error <= 1e-8
    assert force_error <= 1e-7
    np.testing.assert_allclose(point.forces.sum(axis=0), 0, atol=1e-9, rtol=0)

    coordinates = np.asarray([position for _, position in atoms])
    direction = np.array(
        [[0.13, -0.07, 0.11], [-0.05, 0.17, 0.03], [0.09, 0.02, -0.14]]
    )[: len(atoms)]
    estimates = []
    for step in (3e-4, 1e-4):
        energies = []
        for sign in (1, -1):
            moved = [
                (atom[0], position)
                for atom, position in zip(
                    atoms, coordinates + sign * step * direction, strict=True
                )
            ]
            result = calculator.singlepoint(
                moved, charge=charge, multiplicity=multiplicity, properties=("energy",)
            )
            assert result.converged and np.isfinite(result.energy)
            energies.append(result.energy)
        estimates.append((energies[0] - energies[1]) / (2 * step))
    analytic = -float(np.sum(point.forces * direction))
    assert abs(estimates[-1] - estimates[-2]) <= 1e-6
    assert abs(estimates[-1] - analytic) <= 1e-6
    evidence = os.environ.get("GENERATIVEQC_FORCE_AO_DENSITY_EVIDENCE")
    if evidence:
        directory = Path(evidence)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{spin}-{enabled}.json").write_text(
            json.dumps(
                {
                    "job": os.environ["SLURM_JOB_ID"],
                    "method": method,
                    "case": "water" if spin == "rks" else "water-cation",
                    "atoms_bohr": atoms,
                    "charge": charge,
                    "representation": "spherical",
                    "enabled": enabled,
                    "energy_error": energy_error,
                    "force_error": force_error,
                    "finite_difference": estimates,
                    "analytic_directional_derivative": analytic,
                },
                indent=2,
            )
            + "\n"
        )

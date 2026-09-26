"""Real CUDA hybrid exports must preserve the composition used by the SCF owner."""

import os

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import StationaryKsState
from vibeqc_compiler.dft import NativeAO

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks", "b3lyp-rks", "b3lyp-uks"))
def test_cuda_hybrid_snapshot_matches_cpu_composition_and_state(method: str) -> None:
    """Exercise the native writer and Python reader after a converged solve."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    charge, multiplicity = (1, 2) if method.endswith("-uks") else (0, 1)
    options = KsOptions(
        grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
    )
    exported = {}
    for device in ("cpu", "cuda"):
        calculator = Calculator(
            method=method,
            device=device,
            ks_options=options,
            max_iterations=200,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
        )
        with (
            calculator.prepare_batch(
                [atoms], charges=[charge], multiplicities=[multiplicity]
            ) as batch,
            NativeAO(atoms, charge=charge, multiplicity=multiplicity) as basis,
        ):
            item = batch.execute(strict=True).items[0]
            state = StationaryKsState.from_native(batch, basis)
            try:
                snapshot = state._source
                assert snapshot.backend == device
                assert snapshot.metadata[0] == (6 if device == "cpu" else 8)
                assert snapshot.metadata[7] == (2 if method.startswith("b3lyp") else 1)
                assert item.ks_diagnostic.scf_domain == (
                    "b3lyp-vwn-rpa-tail-v1/density-vacuum-1e-18"
                    if method.startswith("b3lyp")
                    else "semilocal-scaled-v1/pbe-spin-c2-1e-18"
                )
                assert snapshot.coefficients == calculator.ks_options.coefficients
                assert state.identity.method == method
                exported[device] = (
                    item.energy,
                    np.array(state.density),
                    np.array(state.fock),
                )
            finally:
                state._source.close()
    assert exported["cuda"][0] == pytest.approx(exported["cpu"][0], abs=2e-9)
    for cpu, cuda in zip(exported["cpu"][1:], exported["cuda"][1:], strict=True):
        np.testing.assert_allclose(cuda, cpu, atol=2e-8, rtol=0)

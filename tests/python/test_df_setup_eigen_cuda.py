"""Setup-provider substitution preserves seeds, caches and full molecular endpoints."""

import os

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks.df_component_ledger import aggregate_host, read_host_trace
from benchmarks.df_host_workloads import validate_final_eigen_counts

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("batch_size", (1, 4))
# Cover provider substitution with the current per-item solver/DIIS floors.
# Batch-four rejection of the former 8-MiB request has its own resource test.
@pytest.mark.parametrize("budget", (0, 32 << 20))
def test_setup_provider_matches_reference_across_replans(
    method, representation, batch_size, budget, monkeypatch, tmp_path
):
    """Two prepared owners compare only the setup eigensolver, including forces.

    Oxygen's d shell distinguishes Cartesian and spherical layouts. Full
    gradients exercise Pulay, auxiliary and metric response after the changed
    setup frame. A bad neighbor must not overwrite an earlier good item.
    """
    assert os.environ.get("SLURM_JOB_ID")
    # Provider substitution must hold final work fixed after candidate reuse.
    monkeypatch.setenv("VIBEQC_DF_FORCE_FINAL_REBUILD", "1")
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8))]
    if method == "rhf":
        atoms.append(("H", (1.7, 0, -0.6)))
    options = {
        "method": method,
        "basis": "def2-svp",
        "basis_representation": representation,
        "device": "cuda",
        "density_fitting": "cuda",
        "density_fitting_memory_budget_bytes": budget,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    preparation = {"multiplicities": [2 if method == "uhf" else 1] * batch_size}
    systems = [atoms] * batch_size
    coordinates = np.array([atom[1] for atom in atoms])
    changed = coordinates.copy()
    changed[-1, 0] += 0.02
    with (
        Calculator(**options).prepare_batch(systems, **preparation) as device,
        Calculator(**options).prepare_batch(systems, **preparation) as oracle,
    ):
        for step, (positions, properties) in enumerate(
            (
                (None, ("energy", "forces")),
                (None, ("energy",)),
                ([None] * (batch_size - 1) + [changed], ("energy", "forces")),
                (None, ("energy",)),
            )
        ):
            outputs = []
            for reference, owner in ((True, oracle), (False, device)):
                path = tmp_path / f"{step}-{reference}.jsonl"
                monkeypatch.setenv(
                    "VIBEQC_DF_REFERENCE_SETUP_EIGEN", "1" if reference else "0"
                )
                monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
                try:
                    result = owner.execute(
                        positions, strict=True, properties=properties
                    )
                finally:
                    monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
                    monkeypatch.delenv("VIBEQC_DF_REFERENCE_SETUP_EIGEN")
                components = aggregate_host(read_host_trace(path))
                validate_final_eigen_counts(
                    components,
                    batch_size=batch_size,
                    method=method,
                    reference=False,
                    strict_final_state=True,
                )
                expected = {
                    "overlap": batch_size if step == 0 else int(step >= 2),
                    "core_guess": batch_size if step == 0 else 0,
                }
                for reason, count in expected.items():
                    for key, selected in (
                        ("eigensolves_by_reason", reference),
                        ("device_eigensolves_by_reason", not reference),
                    ):
                        assert components[key].get(reason, {}).get("calls", 0) == (
                            count if selected else 0
                        )
                assert all(item.executed_backend == "cuda" for item in result.items)
                outputs.append(result)
            np.testing.assert_allclose(
                outputs[0].energies, outputs[1].energies, atol=1e-9, rtol=0
            )
            for a, b in zip(outputs[0].items, outputs[1].items, strict=True):
                if "forces" in properties:
                    np.testing.assert_allclose(a.forces, b.forces, atol=1e-8, rtol=0)
                else:
                    assert a.forces is None and b.forces is None
            if step == 0:
                device.set_warm_start_updates(False)
                oracle.set_warm_start_updates(False)
            if step == 2:
                # A subsequent positions=None evaluates the prepared original
                # geometry. Recovery below explicitly submits `changed` again.
                changed_result = outputs[1]
        if batch_size > 1:
            broken = changed.copy()
            broken[-1, 0] = np.nan
            failed = device.execute([None] * (batch_size - 1) + [broken], strict=False)
            assert failed.failure_indices == (batch_size - 1,)
            recovered = device.execute(
                [None] * (batch_size - 1) + [changed], strict=True
            )
            np.testing.assert_allclose(
                recovered.energies, changed_result.energies, atol=1e-9, rtol=0
            )
            for actual, expected in zip(
                recovered.items, changed_result.items, strict=True
            ):
                np.testing.assert_allclose(
                    actual.forces, expected.forces, atol=1e-8, rtol=0
                )


def test_setup_empty_beta_channel(monkeypatch, tmp_path):
    """The empty UHF spin stays empty while its shared cold frame is replaced."""
    assert os.environ.get("SLURM_JOB_ID")
    # Provider substitution must hold final work fixed after candidate reuse.
    monkeypatch.setenv("VIBEQC_DF_FORCE_FINAL_REBUILD", "1")
    atoms = [("H", (0, 0, 0))]
    calc = Calculator(
        method="uhf", basis="def2-svp", device="cuda", density_fitting="cuda"
    )
    results = []
    for reference in (True, False):
        path = tmp_path / f"empty-beta-{reference}.jsonl"
        monkeypatch.setenv("VIBEQC_DF_REFERENCE_SETUP_EIGEN", "1" if reference else "0")
        monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
        result = calc.singlepoint(
            atoms, multiplicity=2, properties=("energy", "forces")
        )
        monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
        monkeypatch.delenv("VIBEQC_DF_REFERENCE_SETUP_EIGEN")
        components = aggregate_host(read_host_trace(path))
        for reason in ("overlap", "core_guess"):
            assert components["eigensolves_by_reason"].get(reason, {}).get(
                "calls", 0
            ) == int(reference)
            assert components["device_eigensolves_by_reason"].get(reason, {}).get(
                "calls", 0
            ) == int(not reference)
        assert result.converged
        results.append(result)
    assert abs(results[0].energy - results[1].energy) < 1e-9
    np.testing.assert_allclose(results[0].forces, results[1].forces, atol=1e-8, rtol=0)

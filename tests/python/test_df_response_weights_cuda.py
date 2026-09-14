"""Retained values use device force weights without changing the physical result."""

import os

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks.df_component_ledger import read_trace
from benchmarks.df_progress_ledger import read_progress, summarize_progress

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("budget", (0, 64 << 20))
@pytest.mark.parametrize("metric_dot", ("blas", "serial"))
def test_response_route_and_host_ablation(
    method, representation, budget, metric_dot, monkeypatch, tmp_path
):
    """Switch only force-weight placement, including reuse of a warm value plan.

    The source route must remain on device even when the diagnostic host switch
    is set. Independent RHF/UHF gradients include the auxiliary-basis response.
    """
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv(
        "VIBEQC_DF_SERIAL_RESPONSE_DOT", "1" if metric_dot == "serial" else "0"
    )
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8)), ("H", (1.7, 0, -0.6))]
    if method == "uhf":
        atoms.pop()
    mol = gto.M(
        atom=atoms,
        basis="def2-svp",
        unit="Bohr",
        cart=representation == "cartesian",
        spin=int(method == "uhf"),
        verbose=0,
    )
    mf = (scf.UHF(mol) if method == "uhf" else scf.RHF(mol)).density_fit(
        auxbasis="def2-svp"
    )
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 100
    mf.kernel()
    assert mf.converged
    gradient = mf.nuc_grad_method()
    gradient.auxbasis_response = True
    expected_forces = -gradient.kernel()
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=budget,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calc.prepare_batch(
        [atoms], multiplicities=[2 if method == "uhf" else 1]
    ) as owner:
        for step, host in enumerate((False, True, False)):
            monkeypatch.setenv("VIBEQC_DF_HOST_RESPONSE_WEIGHTS", "1" if host else "0")
            trace_path = tmp_path / f"cuda-{step}.jsonl"
            progress_path = tmp_path / f"progress-{step}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace_path))
            monkeypatch.setenv("VIBEQC_DF_PROGRESS_TRACE", str(progress_path))
            item = owner.execute(properties=("energy", "forces"), strict=True).items[0]
            assert abs(item.energy - mf.e_tot) < 1e-9
            np.testing.assert_allclose(item.forces, expected_forces, atol=1e-8, rtol=0)
            records = [
                r for r in read_trace(trace_path) if r["operation"] == "force_response"
            ]
            assert len(records) == 1
            record = records[0]
            names = {r["name"] for r in record["regions"]}
            counters = record["counters"]
            on_device = bool(budget) or not host
            assert ("response_weights" in names) == on_device
            assert ("host_response_weights" in names) != on_device
            uploaded = counters.get("raw_value_upload_bytes", 0)
            assert bool(uploaded) == (on_device and not budget)
            assert uploaded == counters["tensor_host_to_device_bytes"]
            assert bool(counters["response_host_to_device_bytes"]) != on_device
            for dot in ("blas", "serial"):
                assert bool(counters.get(f"response_metric_{dot}_dots", 0)) == (
                    on_device and metric_dot == dot
                )
            if step and not budget:
                progress = summarize_progress(read_progress(progress_path))
                assert "host:df_plan_setup" not in progress["phases"]

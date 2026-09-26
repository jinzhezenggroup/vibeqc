"""Retained values use device force weights without changing the physical result."""

import os
import typing

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
def test_raw_upload_attribution_preserves_complete_response(
    method: typing.Any,
    representation: typing.Any,
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
) -> None:
    """Diagnostic transfers and sink layouts preserve complete weighted response.

    Returning to default on the same prepared plan also exercises the pinned
    buffer lifetime: its async read must finish before response cleanup.
    Independent physical parity is covered by the response-route test below.
    """
    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8))]
    if method == "rhf":
        atoms.append(("H", (1.7, 0, -0.6)))
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calc.prepare_batch(
        [atoms], multiplicities=[2 if method == "uhf" else 1]
    ) as owner:
        owner.execute(properties=("energy", "forces"), strict=True)
        expected = None
        source_backed = False
        probes = (("", ""), ("drain", ""), ("packed", ""), ("", "sharded"), ("", ""))
        for step, (probe, sink) in enumerate(probes):
            monkeypatch.setenv("VIBEQC_DF_RESPONSE_UPLOAD_PROBE", probe)
            monkeypatch.setenv("VIBEQC_DF_RESPONSE_SCATTER_PROBE", sink)
            path = tmp_path / f"probe-{step}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(path))
            if probe and source_backed:
                # Automatic budgets also use device sources: a host-only
                # upload diagnostic must reject that owner, not create a copy.
                # The public batch API reports this unsupported native route
                # as an item numerical failure; returning to default below
                # must still preserve the prepared owner and physical result.
                with pytest.raises(RuntimeError, match="numerical failure"):
                    owner.execute(properties=("energy", "forces"), strict=True)
                continue
            item = owner.execute(properties=("energy", "forces"), strict=True).items[0]
            (record,) = [
                r for r in read_trace(path) if r["operation"] == "force_response"
            ]
            counters = record["counters"]
            if expected is None:
                expected = item
                source_backed = record["source_backed"]
                raw_bytes = counters.get("raw_value_upload_bytes", 0)
                blocks = counters["response_auxiliary_blocks"]
                scratch = counters["response_scratch_bytes"]
            assert abs(item.energy - expected.energy) < 1e-10
            np.testing.assert_allclose(item.forces, expected.forces, rtol=0, atol=1e-9)
            assert counters.get("raw_value_upload_bytes", 0) == raw_bytes
            assert counters["response_auxiliary_blocks"] == blocks
            assert counters["response_scratch_bytes"] == scratch
            assert counters.get("derivative_probe_gradient_copies", 0) == (
                128 if sink else 0
            )
            names = {r["name"] for r in record["regions"]}
            assert ("gradient_probe_shard_reduction" in names) == bool(sink)
            assert counters["response_probe_total_device_bytes"] == (
                scratch + counters.get("derivative_probe_scratch_bytes", 0)
            )
            assert counters.get("raw_probe_prior_stream_drains", 0) == (
                raw_bytes // (8 * record["nbf"] ** 2) if probe else 0
            )
            assert counters.get("raw_probe_gather_elements", 0) == (
                raw_bytes // 8 if probe == "packed" else 0
            )
            assert counters.get("raw_probe_pinned_host_bytes", 0) == (
                8 * record["nbf"] ** 2 if probe == "packed" and raw_bytes else 0
            )


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("budget", (0, 64 << 20))
@pytest.mark.parametrize("metric_dot", ("blas", "serial"))
def test_response_route_and_host_ablation(
    method: typing.Any,
    representation: typing.Any,
    budget: typing.Any,
    metric_dot: typing.Any,
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
) -> None:
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
            # Public zero now resolves to a positive automatic source budget.
            # Follow the actual source owner, not the old zero-means-host rule.
            source_backed = record["source_backed"]
            on_device = source_backed or bool(budget) or not host
            assert ("response_weights" in names) == on_device
            assert ("host_response_weights" in names) != on_device
            uploaded = counters.get("raw_value_upload_bytes", 0)
            assert bool(uploaded) == (on_device and not source_backed)
            assert uploaded == counters["tensor_host_to_device_bytes"]
            assert bool(counters["response_host_to_device_bytes"]) != on_device
            for dot in ("blas", "serial"):
                assert bool(counters.get(f"response_metric_{dot}_dots", 0)) == (
                    on_device and metric_dot == dot
                )
            if step and not budget:
                progress = summarize_progress(read_progress(progress_path))
                assert "host:df_plan_setup" not in progress["phases"]

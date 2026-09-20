"""Host DIIS retries retain a qualified device eigen provider and failure limits."""

import os
import typing

import pytest
from vibeqc import Calculator

from benchmarks.df_component_ledger import aggregate_host, read_host_trace
from benchmarks.df_progress_ledger import read_progress, summarize_progress

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("water_count", (1, 2))
@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("route", ("single", "batch-one", "batch-four"))
def test_diis_retry_provider_and_iteration_limit(
    method: typing.Any,
    representation: typing.Any,
    route: typing.Any,
    water_count: typing.Any,
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
) -> None:
    """One iteration forces the existing compact-to-DIIS transition.

    Compare the explicit reference diagnostic with ordinary device execution
    through both public entry points. Neither provider may turn the exhausted
    iteration limit into success; actual retry leaves must identify the chosen
    provider. This intentionally failed solve is not molecular convergence
    evidence, and the reference operation remains an independent oracle.
    """
    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8)), ("H", (1.7, 0, -0.6))]
    atoms = [
        (symbol, (x, y, z + 6.0 * i))
        for i in range(water_count)
        for symbol, (x, y, z) in atoms
    ]
    # Occupied mode executes a dense seed before capture; this gives the
    # generic provider an ordinary stream boundary even on capture-capable GPUs.
    monkeypatch.setenv(
        "VIBEQC_DF_EXCHANGE", "dense" if water_count == 1 else "occupied"
    )
    spin = int(method == "uhf")
    count = 4 if route == "batch-four" else 1
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=32 << 20,
        max_iterations=1,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    for reference in (True, False):
        path = tmp_path / f"retry-{reference}.jsonl"
        progress_path = tmp_path / f"progress-{reference}.jsonl"
        monkeypatch.setenv("VIBEQC_DF_PROGRESS_TRACE", str(progress_path))
        monkeypatch.setenv("VIBEQC_DF_REFERENCE_ITERATION_EIGEN", str(int(reference)))
        monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
        try:
            if route == "single":
                with pytest.raises(RuntimeError, match="converg"):
                    calc.singlepoint(
                        atoms,
                        charge=spin,
                        multiplicity=1 + spin,
                        properties=("energy",),
                    )
            else:
                with calc.prepare_batch(
                    [atoms] * count,
                    charges=[spin] * count,
                    multiplicities=[1 + spin] * count,
                ) as batch:
                    result = batch.execute(strict=False, properties=("energy",))
                    assert result.failure_indices == tuple(range(count))
                    assert all(item.iterations == 1 for item in result.items)
        finally:
            monkeypatch.delenv("VIBEQC_DF_PROGRESS_TRACE")
            monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
            monkeypatch.delenv("VIBEQC_DF_REFERENCE_ITERATION_EIGEN")
        journal = read_progress(progress_path)
        assert journal["complete"]
        progress = summarize_progress(journal)
        assert progress["phases"]["host:diis_retry_iteration"]["calls"] == 1
        readbacks = [
            r["value"]
            for r in progress["observations"]
            if r["key"] == "device_iterations"
        ]
        assert readbacks == [1] * count
        # A batch solve submits one provider call per spin. Each call solves
        # count matrices; graph construction is retained separately.
        providers = [
            r
            for r in progress["observations"]
            if r["name"] == "compact_eigensolve" and r["key"] == "eigen_provider"
        ]
        executed = [r for r in providers if r["execution"] == "stream"]
        replayed = any(
            r["key"] == "host_graph_replay" for r in progress["observations"]
        )
        assert len(executed) == (0 if replayed else 1 + spin)
        assert (
            len([r for r in providers if r["execution"] == "graph_capture"]) == 1 + spin
        )
        expected_provider = (
            "cusolverDnDsyevjBatched" if water_count == 1 else "cusolverDnXsyevBatched"
        )
        assert providers and all(r["value"] == expected_provider for r in providers)
        if water_count == 2:
            assert executed  # The dense seed ran before graph construction.
        counts = [
            r["value"]
            for r in progress["observations"]
            if r["name"] == "compact_eigensolve"
            and r["key"] == "eigensystems"
            and r["execution"] == "stream"
        ]
        assert counts == ([] if replayed else [count] * (1 + spin))
        assert any(
            r["key"] == "seed_generation" and r["value"] == "original_caller_density"
            for r in progress["observations"]
        )
        components = aggregate_host(read_host_trace(path))
        expected = count * (1 + spin)
        for key, active in (
            ("eigensolves_by_reason", reference),
            ("device_eigensolves_by_reason", not reference),
        ):
            assert components[key].get("fallback", {}).get("calls", 0) == (
                expected if active else 0
            )
        if not reference:
            assert not components["eigensolves_by_reason"]

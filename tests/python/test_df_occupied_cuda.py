"""Actual SCF factor reuse must preserve force/replay semantics and provenance."""

import json
import os

import numpy as np
import pytest
from vibeqc import Calculator

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated CUDA test job",
)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("budget", [0, 512 << 10, 8 << 20])
def test_occupied_scf_matches_dense_across_warm_replays(
    method, budget, monkeypatch, tmp_path
):
    """Policy changes rebuild captured shapes; imported warm D seeds dense K."""
    assert os.environ.get("SLURM_JOB_ID")
    # The smaller positive allowance still covers the complete three-item
    # source/SCF plan. Deliberately partial AO/auxiliary tiles are exercised by
    # the native fixed-density tests; an infeasible plan cannot test K parity.
    systems = [[(1, (0.0, 0.0, -r)), (1, (0.0, 0.0, r))] for r in (0.7, 1.1, 1.5)]
    options = {
        "method": method,
        "basis": "def2-svp",
        "device": "cuda",
        "density_fitting": "cuda",
        "density_fitting_memory_budget_bytes": budget,
        "max_iterations": 100,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    # UHF includes rank-zero beta and two neighbors in an independent bucket.
    preparation = {"multiplicities": [3, 1, 1]} if method == "uhf" else {}
    with (
        Calculator(**options).prepare_batch(systems, **preparation) as batch,
        Calculator(**options).prepare_batch(systems, **preparation) as reference,
    ):
        for replay, policy in enumerate(("occupied", "dense", "occupied")):
            monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "dense")
            expected = reference.execute(strict=True)
            monkeypatch.setenv("VIBEQC_DF_EXCHANGE", policy)
            trace = tmp_path / f"replay-{replay}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            actual = batch.execute(strict=True)
            monkeypatch.delenv("VIBEQC_DF_TRACE")
            np.testing.assert_allclose(
                actual.energies, expected.energies, atol=1e-9, rtol=0
            )
            for a, b in zip(actual.items, expected.items, strict=True):
                np.testing.assert_allclose(a.forces, b.forces, atol=1e-8, rtol=0)
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            provenance = [
                r for r in records if r["operation"] == "occupied_scf_provenance"
            ]
            if policy == "occupied":
                # A numerical-recovery path could match energies while hiding
                # a broken device optimization; require executed validation.
                assert provenance
                for record in provenance:
                    assert record["execution"] == "stream"
                    assert record["counters"]["validated_density_generations"] > 0
                    assert record["counters"]["dense_seed_iterations"] == 1
                    assert record["counters"]["occupied_iterations"] > 0
            else:
                assert not provenance

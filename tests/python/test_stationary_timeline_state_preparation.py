"""Timeline fixtures must provide a valid state before derivative measurement."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from tools import benchmark_stationary_cuda_timeline as benchmark


def test_benchmark_pins_snapshot_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "Calculator", lambda **kw: SimpleNamespace(**kw))
    calculator = benchmark._calculator("lda-rks")
    assert calculator.density_tolerance == 1e-12
    assert calculator.energy_tolerance == 1e-12
    assert calculator.max_iterations == 200


@pytest.mark.skipif(
    os.environ.get("VIBEQC_TIMELINE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated CUDA timeline qualification run",
)
def test_changed_geometry_keeps_strict_snapshot_eligible() -> None:
    from vibeqc_compiler.dft import NativeAO

    assert os.environ.get("SLURM_JOB_ID"), "CUDA qualification requires Slurm"
    atoms = benchmark.SYSTEMS["water"]
    calculator = benchmark._calculator("lda-rks")
    with calculator.prepare_batch([atoms]) as batch:
        for scenario in ("cold", "warm", "changed_geometry"):
            xyz = np.asarray([a[1] for a in atoms], dtype=float)
            if scenario == "changed_geometry":
                xyz[-1] += (0.0010, -0.0005, 0.0003)
            coordinates = [xyz] if scenario == "changed_geometry" else None
            item = batch.execute(
                coordinates=coordinates, strict=True, properties=("energy",)
            ).items[0]
            assert item.converged and item.executed_backend == "cuda"
            current = [(a[0], tuple(r)) for a, r in zip(atoms, xyz, strict=True)]
            with NativeAO(current) as basis:
                state, seconds = benchmark._export_state(batch, basis)
                assert state is not None and seconds >= 0

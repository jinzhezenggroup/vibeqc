"""A later failure must preserve earlier measurements without becoming success."""

import json
import sys
from pathlib import Path

import pytest

from tools import benchmark_stationary_cuda_timeline as benchmark


def test_main_persists_completed_rows_when_later_case_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evidence.json"
    monkeypatch.setenv("SLURM_JOB_ID", "host-control-flow-test")
    monkeypatch.setenv("CUDACXX", "/unused/nvcc")
    monkeypatch.delenv("VIBEQC_LIBRARY", raising=False)
    monkeypatch.setattr(benchmark, "CudaCompilerAdapter", lambda *a, **k: None)
    monkeypatch.setattr(benchmark, "_git", lambda command: None)
    monkeypatch.setattr(benchmark, "_gpu_identity", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "timeline",
            "--output",
            str(output),
            "--methods",
            "pbe-rks",
            "--systems",
            "h2,water",
        ],
    )
    completed = {"status": "ok", "system": "h2", "scenario": "cold"}

    def case(**kwargs: object) -> list[dict]:
        if kwargs["system"] == "water":
            raise RuntimeError("injected state-export failure")
        records = kwargs.get("records", [])
        records.append(completed)
        return records

    monkeypatch.setattr(benchmark, "benchmark_case", case)
    with pytest.raises(RuntimeError, match="injected state-export failure"):
        benchmark.main()
    assert output.is_file(), "later failure discarded all completed measurements"
    payload = json.loads(output.read_text())
    assert completed in payload["records"]
    assert payload["provenance"]["scf_preparation"] == dict(benchmark.SCF_PREPARATION)
    assert payload["completed"] is False
    assert payload["failure"]["error_type"] == "RuntimeError"


def test_partial_case_retains_completed_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    batch = SimpleNamespace(
        execute=lambda **kwargs: SimpleNamespace(items=[SimpleNamespace(energy=-1.0)])
    )
    calculator = SimpleNamespace(prepare_batch=lambda *a, **k: nullcontext(batch))
    monkeypatch.setattr(benchmark, "_calculator", lambda method: calculator)
    monkeypatch.setattr(benchmark, "NativeAO", lambda *a, **k: nullcontext(None))
    exports = iter([(object(), 0.01), (object(), 0.02)])

    def export(*args: object) -> tuple[object, float]:
        try:
            return next(exports)
        except StopIteration as error:
            raise RuntimeError("changed-geometry export failed") from error

    monkeypatch.setattr(benchmark, "_export_state", export)
    monkeypatch.setattr(benchmark, "_diagnostic", lambda *a: (None, 0.03))
    monkeypatch.setattr(
        benchmark,
        "_successful_record",
        lambda **k: {"status": "ok", "scenario": k["scenario"]},
    )
    records: list[dict] = []
    with pytest.raises(RuntimeError, match="changed-geometry export failed"):
        benchmark.benchmark_case(
            system="h2",
            atoms=benchmark.SYSTEMS["h2"],
            method="pbe-rks",
            compiler=None,
            cache=tmp_path,
            same_state_repeats=1,
            records=records,
        )
    assert [r["scenario"] for r in records] == [
        "cold",
        "artifact_warm",
        "same_state_warm",
    ]

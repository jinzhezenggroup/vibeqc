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
    library = tmp_path / "libvibeqc.so"
    library.write_bytes(b"host-only provenance fixture; never loaded")
    monkeypatch.setenv("SLURM_JOB_ID", "host-control-flow-test")
    monkeypatch.delenv("CUDACXX", raising=False)
    monkeypatch.setenv("VIBEQC_LIBRARY", str(library))
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
        assert kwargs["compiler"] is None
        assert kwargs["library"] == library.resolve()
        assert kwargs["target"] is not None
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
    assert payload["provenance"]["execution_route"] == "prepared-aot"
    assert payload["provenance"]["runtime_compilation"] is False
    assert payload["completed"] is False
    assert payload["failure"]["error_type"] == "RuntimeError"


@pytest.mark.parametrize("prepared_aot", [False, True])
def test_partial_case_retains_completed_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared_aot: bool
) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    batch = SimpleNamespace(
        execute=lambda **kwargs: SimpleNamespace(items=[SimpleNamespace(energy=-1.0)])
    )
    calculator = SimpleNamespace(prepare_batch=lambda *a, **k: nullcontext(batch))
    monkeypatch.setattr(benchmark, "_calculator", lambda method: calculator)
    monkeypatch.setattr(benchmark, "NativeAO", lambda *a, **k: nullcontext(None))
    owner = object()
    target = object()
    lifecycle: list[str] = []

    @contextmanager
    def prepared_owner() -> Iterator[object]:
        lifecycle.append("enter")
        try:
            yield owner
        finally:
            lifecycle.append("close")

    monkeypatch.setattr(benchmark, "PreparedStationaryCudaExecution", prepared_owner)
    exports = iter([(object(), 0.01), (object(), 0.02)])

    def export(*args: object) -> tuple[object, float]:
        try:
            return next(exports)
        except StopIteration as error:
            raise RuntimeError("changed-geometry export failed") from error

    def diagnostic(*args: object, **kwargs: object) -> tuple[None, float]:
        assert args[2] is None
        if prepared_aot:
            assert kwargs["prepared"] is owner
            assert kwargs["target"] is target
            assert kwargs["library"] == tmp_path / "libvibeqc.so"
        else:
            assert not kwargs
        return None, 0.03

    monkeypatch.setattr(benchmark, "_export_state", export)
    monkeypatch.setattr(benchmark, "_diagnostic", diagnostic)
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
            target=target if prepared_aot else None,
            library=tmp_path / "libvibeqc.so" if prepared_aot else None,
        )
    assert [r["scenario"] for r in records] == [
        "cold",
        "artifact_warm",
        "same_state_warm",
    ]
    assert lifecycle == (["enter", "close"] if prepared_aot else [])

"""Benchmark setup must neither delete caller files nor run unmeasured forces."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_stationary_cuda_timeline as benchmark


@pytest.mark.parametrize("check", ("cache", "properties"))
def test_cold_case_preserves_cache_and_prepares_energy_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check: str
) -> None:
    calls, caches = [], []

    def execute(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(items=[SimpleNamespace(energy=-1.0)])

    batch = SimpleNamespace(execute=execute)
    calc = SimpleNamespace(prepare_batch=lambda *args, **kwargs: nullcontext(batch))
    monkeypatch.setattr(benchmark, "_calculator", lambda method: calc)
    monkeypatch.setattr(
        benchmark, "NativeAO", lambda *args, **kwargs: nullcontext(object())
    )
    monkeypatch.setattr(benchmark, "_export_state", lambda *args: (object(), 0.01))
    monkeypatch.setattr(
        benchmark,
        "_diagnostic",
        lambda state, basis, compiler, cache: (caches.append(cache), 0.02),
    )
    monkeypatch.setattr(
        benchmark,
        "_successful_record",
        lambda **kwargs: {"scenario": kwargs["scenario"]},
    )
    sentinel = tmp_path / "existing-evidence.txt"
    sentinel.write_text("preserve")
    records = benchmark.benchmark_case(
        system="h2",
        atoms=benchmark.SYSTEMS["h2"],
        method="pbe-rks",
        compiler=None,
        cache=tmp_path,
        same_state_repeats=1,
    )
    if check == "properties":
        assert all(call.get("properties") == ("energy",) for call in calls)
        return
    assert sentinel.is_file(), "cold preparation deleted the supplied cache contents"
    assert len(records) == 4 and len(set(caches)) == 1 and caches[0] != tmp_path
    assert all(call.get("properties") == ("energy",) for call in calls)

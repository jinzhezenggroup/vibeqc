"""Issue #662 timeline evidence must exercise the packaged prepared CUDA route."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_stationary_cuda_timeline as benchmark


def test_diagnostic_binds_packaged_aot_and_declared_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    expected = object()

    def execute(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(benchmark, "complete_rks_cuda_gradient_diagnostic", execute)
    library = tmp_path / "libvibeqc.so"
    library.write_bytes(b"fixture")
    prepared = object()
    target = object()
    cache = tmp_path / "cache"

    result, observed = benchmark._diagnostic(
        object(),
        object(),
        None,
        cache,
        prepared=prepared,
        target=target,
        library=library,
    )

    assert result is expected and observed >= 0.0
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["compiler"] is None
    assert kwargs["cache"] == cache
    assert kwargs["aot_directory"] == library.parent
    assert kwargs["native_grid_library"] == library
    assert kwargs["target"] is target
    assert kwargs["prepared"] is prepared
    assert kwargs["profile_device"] is True
    assert {key: kwargs[key] for key in benchmark.PRODUCTION_EXECUTION} == dict(
        benchmark.PRODUCTION_EXECUTION
    )
    assert kwargs["primitive_tile"] == 4096


def test_case_reuses_one_prepared_owner_across_geometry_rebind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    prepared = object()
    target = object()
    library = tmp_path / "libvibeqc.so"
    library.write_bytes(b"fixture")

    def execute(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(items=[SimpleNamespace(energy=-1.0)])

    batch = SimpleNamespace(execute=execute)
    calculator = SimpleNamespace(
        prepare_batch=lambda *args, **kwargs: nullcontext(batch)
    )
    monkeypatch.setattr(benchmark, "_calculator", lambda method: calculator)
    monkeypatch.setattr(
        benchmark, "NativeAO", lambda *args, **kwargs: nullcontext(object())
    )
    monkeypatch.setattr(
        benchmark, "PreparedStationaryCudaExecution", lambda: nullcontext(prepared)
    )
    monkeypatch.setattr(benchmark, "_export_state", lambda *args: (object(), 0.01))

    def diagnostic(
        state: object,
        basis: object,
        compiler: object,
        cache: Path,
        **kwargs: object,
    ) -> tuple[object, float]:
        calls.append({"compiler": compiler, "cache": cache, **kwargs})
        return object(), 0.02

    monkeypatch.setattr(benchmark, "_diagnostic", diagnostic)
    monkeypatch.setattr(
        benchmark,
        "_successful_record",
        lambda **kwargs: {"scenario": kwargs["scenario"]},
    )

    records = benchmark.benchmark_case(
        system="h2",
        atoms=benchmark.SYSTEMS["h2"],
        method="pbe-rks",
        compiler=None,
        target=target,
        library=library,
        cache=tmp_path / "cache-root",
        same_state_repeats=1,
    )

    assert [record["scenario"] for record in records] == [
        "cold",
        "artifact_warm",
        "same_state_warm",
        "changed_geometry",
    ]
    assert len(calls) == 4
    assert all(call["compiler"] is None for call in calls)
    assert all(call["prepared"] is prepared for call in calls)
    assert all(call["target"] is target for call in calls)
    assert all(call["library"] == library for call in calls)
    assert len({call["cache"] for call in calls}) == 1

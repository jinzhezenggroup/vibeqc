"""Host-only contracts for the opt-in stationary CUDA timeline."""

from pathlib import Path
from typing import Any

import pytest
from vibeqc._stationary_cuda import _CudaSources, complete_rks_cuda_gradient_diagnostic


class _FakeLibrary:
    def stationary_probe(self, *args: object) -> int:
        return 0


def test_native_call_timeline_is_opt_in_and_counts_calls() -> None:
    owner = _CudaSources.__new__(_CudaSources)
    owner.library = _FakeLibrary()
    owner._timeline = {}

    owner._call("stationary_probe")
    owner._call("stationary_probe")

    assert owner._timeline["native_call_counts"] == {"stationary_probe": 2}
    assert owner._timeline["native_call_seconds"]["stationary_probe"] >= 0.0


def test_native_call_without_timeline_keeps_old_path() -> None:
    owner = _CudaSources.__new__(_CudaSources)
    owner.library = _FakeLibrary()
    owner._timeline = None
    owner._call("stationary_probe")


def test_measure_timeline_requires_bool_before_state_access() -> None:
    with pytest.raises(TypeError, match="measure_timeline must be bool"):
        complete_rks_cuda_gradient_diagnostic(
            None,
            None,
            compiler=None,
            cache=".",
            measure_timeline=1,
        )


def test_cold_timeline_does_not_delete_caller_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks import stationary_cuda_timeline as timeline

    root = tmp_path / "caller-cache"
    root.mkdir()
    sentinel = root / "unrelated-artifact"
    sentinel.write_text("preserve")
    caches = []

    def sample(**kwargs: Any) -> tuple[dict[str, Any], None]:
        caches.append(kwargs["cache"])
        assert sentinel.read_text() == "preserve"
        assert kwargs["cache"] != root
        assert kwargs["cache"].parent == root
        record = {"mode": kwargs["mode"]}
        if kwargs["mode"] == "artifact-warm":
            record["same_state_replay"] = {"mode": "same-state-warm"}
        return record, None

    monkeypatch.setattr(timeline, "_sample", sample)
    case = timeline._run_case("h2", "lda-rks", None, root, True)
    assert case.get("status", "ok") == "ok"
    assert len(caches) == 3 and len(set(caches)) == 1
    assert not caches[0].exists()
    assert sentinel.read_text() == "preserve"

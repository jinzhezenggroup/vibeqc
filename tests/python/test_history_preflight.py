"""History orchestration must preflight all retained sources and freeze residuals.

A lightweight transport isolates this boundary from orbital-rotation mathematics;
the full classified exact/projected integration lives in test_state_transport_history.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.vibeqc_cc import history_transport as history
from tools.vibeqc_cc.gpu_state import AmplitudeSnapshot


class _Transport:
    source = SimpleNamespace(reference_id="source", nocc=1, nvir=1)
    target = SimpleNamespace(
        reference_id="target", identity="target-state", nocc=1, nvir=1
    )
    identity = "transport"
    compatibility = history.TransportCompatibility.exact_orbital_rotation

    def rotate_amplitudes(
        self, singles: np.ndarray, doubles: np.ndarray
    ) -> AmplitudeSnapshot:
        return AmplitudeSnapshot("target", singles, doubles)


def _snapshot(reference: str = "source", nvir: int = 1) -> AmplitudeSnapshot:
    return AmplitudeSnapshot(reference, np.ones((1, nvir)), np.ones((1, 1, nvir, nvir)))


@pytest.mark.parametrize("invalid", ["reference", "shape", "type"])
def test_later_invalid_source_prevents_all_target_operator_work(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    monkeypatch.setattr(history, "StateTransport", _Transport)
    bad = {
        "reference": lambda: _snapshot("stale"),
        "shape": lambda: _snapshot(nvir=2),
        "type": object,
    }[invalid]()
    calls = []

    def residual(value: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        calls.append(value.reference_id)
        return value.t1.copy(), value.t2.copy()

    with pytest.raises(
        (TypeError, ValueError), match="transport source|AmplitudeSnapshot"
    ):
        history.recycle_diis_history(
            _Transport(),
            [_snapshot(), bad],
            history.TargetResidualEvaluator("target-state", residual),
        )
    assert calls == []


@pytest.mark.parametrize("field", ["residual_singles", "residual_doubles"])
def test_recycled_residual_cannot_reenable_write_access(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setattr(history, "StateTransport", _Transport)

    def residual(value: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        return value.t1.copy(), value.t2.copy()

    result = history.recycle_diis_history(
        _Transport(),
        [_snapshot()],
        history.TargetResidualEvaluator("target-state", residual),
    )
    with pytest.raises(ValueError):
        getattr(result.entries[0], field).setflags(write=True)


def test_discarded_prefix_does_not_enter_retained_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(history, "StateTransport", _Transport)
    calls = []

    def residual(value: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        calls.append(value.reference_id)
        return value.t1.copy(), value.t2.copy()

    result = history.recycle_diis_history(
        _Transport(),
        [_snapshot("stale"), _snapshot()],
        history.TargetResidualEvaluator("target-state", residual),
        policy=history.HistoryRecyclePolicy(maximum_vectors=1),
    )
    assert result.dropped_prefix_count == 1
    assert result.entries[0].source_index == 1
    assert calls == ["target"]

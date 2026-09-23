"""Checkpoint recovery must never bypass an expired endpoint's termination."""

from __future__ import annotations

import json
import signal
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from benchmarks import _endpoint_progress as module

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("payload", [b"\xff\xfe", b"{broken", b"[]"])
def test_unreadable_checkpoint_preserves_fallback_and_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    output = tmp_path / "result.json"
    output.write_bytes(payload)
    original = {"active_endpoint": "reference/cold", "prior_energy": -75.0}
    connection = Mock()
    connection.poll.return_value = False
    kill = Mock()
    monkeypatch.setattr(module.os, "kill", kill)

    module._watch(connection, 12345, 0.01, output, original)

    kill.assert_called_once_with(12345, signal.SIGKILL)
    connection.close.assert_called_once()
    record = json.loads(output.read_text())
    assert record["prior_energy"] == -75.0
    assert record["status"] == "stopped"
    assert "status" not in original


def test_unexpected_checkpoint_exception_still_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = Mock()
    connection.poll.return_value = False
    kill = Mock()
    monkeypatch.setattr(module.os, "kill", kill)
    monkeypatch.setattr(
        module, "_latest_record", Mock(side_effect=RuntimeError("read failed"))
    )
    with pytest.raises(RuntimeError, match="read failed"):
        module._watch(
            connection,
            12345,
            0.01,
            tmp_path / "result.json",
            {"active_endpoint": "reference/cold"},
        )
    kill.assert_called_once_with(12345, signal.SIGKILL)
    connection.close.assert_called_once()

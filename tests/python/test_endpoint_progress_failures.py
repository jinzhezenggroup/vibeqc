"""Deadline cancellation must survive diagnostic I/O and startup failures."""

from __future__ import annotations

import signal
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from benchmarks import _endpoint_progress as module

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("failing", ("_append", "save_record"))
def test_timeout_kills_owner_even_if_evidence_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    connection = Mock()
    connection.poll.return_value = False
    kill = Mock()
    append, save = Mock(), Mock()
    monkeypatch.setattr(module.os, "kill", kill)
    monkeypatch.setattr(module, "_append", append)
    monkeypatch.setattr(module, "save_record", save)
    getattr(module, failing).side_effect = OSError("disk unavailable")
    with pytest.raises(OSError, match="disk unavailable"):
        module._watch(
            connection,
            12345,
            0.01,
            tmp_path / "result.json",
            {"active_endpoint": "reference/cold"},
        )
    kill.assert_called_once_with(12345, signal.SIGKILL)
    connection.close.assert_called_once()
    append.assert_called_once()
    save.assert_called_once()


@pytest.mark.parametrize("failure", ("begin", "start"))
def test_failed_arming_closes_pipes_and_disarms_started_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    progress = module.EndpointProgress(tmp_path / "result.json", {}, 1, False)
    receiver, sender, watchdog = Mock(), Mock(), Mock()
    watchdog.is_alive.return_value = False
    context = SimpleNamespace(
        Pipe=Mock(return_value=(receiver, sender)),
        Process=Mock(return_value=watchdog),
    )
    monkeypatch.setattr(module.mp, "get_context", lambda _: context)
    if failure == "start":
        watchdog.start.side_effect = OSError("start failed")
    else:
        monkeypatch.setattr(
            progress, "emit", Mock(side_effect=OSError("begin failed"))
        )
    with pytest.raises(OSError, match=f"{failure} failed"), progress.measure("cold"):
        pytest.fail("must not start the solve")
    receiver.close.assert_called_once()
    sender.close.assert_called_once()
    if failure == "begin":
        watchdog.join.assert_called_once_with(timeout=2)
    else:
        watchdog.join.assert_not_called()

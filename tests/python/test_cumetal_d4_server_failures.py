"""Exercise the benchmark protocol and process cleanup without a GPU."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import typing
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def module() -> typing.Any:
    spec = importlib.util.spec_from_file_location(
        "review_d4_server", ROOT / "benchmarks/test_cumetal_d4_codspeed.py"
    )
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("reply", ["inf", "1e999", "nan", "0", "-1"])
def test_nonfinite_or_nonpositive_duration_is_rejected(
    module: typing.Any, monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    process = Mock()
    process.stdin = io.StringIO()
    process.stdout = io.StringIO(f"READY device=fixture\nOK d4 {reply}\n")
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=process))
    server = module._D4Server(Path("fixture"))
    try:
        with pytest.raises(RuntimeError, match="invalid CuMetal D4 device duration"):
            server.run_once()
    finally:
        server.close()


@pytest.mark.parametrize("exited", [False, True])
def test_close_reaps_child_and_closes_pipes(
    module: typing.Any, monkeypatch: pytest.MonkeyPatch, exited: bool
) -> None:
    process = Mock()
    process.stdin = io.StringIO()
    process.stdout = io.StringIO("READY device=fixture\n")
    process.poll.return_value = 0 if exited else None
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=process))
    server = module._D4Server(Path("fixture"))
    server.close()
    assert process.stdin.closed and process.stdout.closed
    if not exited:
        process.wait.assert_called_once_with(timeout=10)
    process.poll.return_value = 0
    server.close()


def test_startup_decode_failure_cleans_up(
    module: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = Mock()
    process.stdin = io.StringIO()
    process.stdout = Mock()
    process.stdout.readline.side_effect = UnicodeDecodeError("utf-8", b"x", 0, 1, "bad")
    process.poll.return_value = None
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(UnicodeDecodeError):
        module._D4Server(Path("fixture"))
    process.wait.assert_called_once_with(timeout=10)
    assert process.stdin.closed
    process.stdout.close.assert_called_once()


def test_close_after_broken_quit_still_reaps_child(
    module: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = Mock()
    process.stdin = Mock()
    process.stdin.closed = False
    process.stdin.write.side_effect = OSError("quit write failed")
    process.stdout = io.StringIO("READY device=fixture\n")
    process.poll.return_value = None
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=process))
    server = module._D4Server(Path("fixture"))
    server.close()
    process.wait.assert_called_once_with(timeout=10)
    process.stdin.close.assert_called_once()
    assert process.stdout.closed


def test_real_child_roundtrip_and_closed_streams(
    module: typing.Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "server.py"
    script.write_text(
        "import sys\nprint('READY device=fixture', flush=True)\n"
        "for line in sys.stdin:\n"
        " if line.strip() == 'quit': break\n"
        " print('OK d4 0.25', flush=True)\n"
    )
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda args, **kwargs: real_popen([sys.executable, str(script)], **kwargs),
    )
    server = module._D4Server(script)
    try:
        assert server.run_once() == 0.25
        assert server.run_once() == 0.25
    finally:
        server.close()
    assert server._process.poll() == 0
    assert server._process.stdin.closed
    assert server._process.stdout.closed


def test_unresponsive_child_is_killed_and_pipes_closed(
    module: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = Mock()
    process.stdin = io.StringIO()
    process.stdout = io.StringIO("READY device=fixture\n")
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 10), 0]
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=process))
    server = module._D4Server(Path("fixture"))
    server.close()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2
    assert process.stdin.closed and process.stdout.closed

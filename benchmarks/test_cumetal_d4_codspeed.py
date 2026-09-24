"""CodSpeed walltime harness for one real CuMetal VibeQC production kernel."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")


class _BenchmarkFixture(Protocol):
    def __call__(self, target: Callable[[], _T]) -> _T: ...


class _D4Server:
    def __init__(self, executable: Path) -> None:
        self._process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            ready = self._process.stdout.readline().strip()
            if not ready.startswith("READY device="):
                raise RuntimeError(
                    f"CuMetal D4 benchmark did not become ready: {ready!r}"
                )
            print(ready, flush=True)
        except BaseException:
            self.close()
            raise

    def run_once(self) -> float:
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write("run d4\n")
        self._process.stdin.flush()
        reply = self._process.stdout.readline().strip()
        prefix = "OK d4 "
        if not reply.startswith(prefix):
            raise RuntimeError(f"invalid CuMetal D4 benchmark reply: {reply!r}")
        device_ms = float(reply[len(prefix) :])
        if not math.isfinite(device_ms) or device_ms <= 0.0:
            raise RuntimeError(f"invalid CuMetal D4 device duration: {device_ms}")
        return device_ms

    def close(self) -> None:
        process = self._process
        try:
            if process.poll() is None:
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.write("quit\n")
                        process.stdin.flush()
                    except (OSError, ValueError):
                        pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        finally:
            # Reap the child before releasing its pipes, including startup
            # failures and a child that has already exited on a numerical error.
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass


@pytest.fixture(scope="module")
def cumetal_d4_server() -> _D4Server:
    path = os.environ.get("VIBEQC_CUMETAL_D4_BENCH")
    if not path:
        pytest.skip("VIBEQC_CUMETAL_D4_BENCH is not configured")
    executable = Path(path)
    if not executable.is_file():
        pytest.fail(f"CuMetal D4 benchmark executable is missing: {executable}")

    server = _D4Server(executable)
    try:
        yield server
    finally:
        server.close()


def test_cumetal_d4_production_walltime(
    benchmark: _BenchmarkFixture,
    cumetal_d4_server: _D4Server,
) -> None:
    """Measure the real cooperative D4 CUDA schedule after strict validation."""
    device_ms = benchmark(cumetal_d4_server.run_once)
    assert device_ms > 0.0

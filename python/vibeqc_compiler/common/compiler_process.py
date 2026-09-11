"""Finite compiler process-tree execution shared by CPU and CUDA adapters."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompileResult:
    """Compiler outcome including deterministic timeout diagnostics."""

    returncode: int
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str


def run_compiler(command: list[str], timeout: float, *, label: str) -> CompileResult:
    """Capture diagnostics and terminate all compiler children on timeout."""
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
    duration = time.monotonic() - started
    if timed_out:
        stderr += f"{label} compilation timed out after {timeout:g} seconds\n"
    return CompileResult(
        returncode=124 if timed_out else process.returncode,
        timed_out=timed_out,
        duration_seconds=duration,
        stdout=stdout,
        stderr=stderr,
    )

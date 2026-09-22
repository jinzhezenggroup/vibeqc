"""Durable progress and a process-level deadline for GPU benchmark endpoints.

The watchdog uses a separate interpreter so a blocked native/CUDA call cannot
prevent cancellation. Only the benchmark process is killed; Slurm continues to
own device visibility and the allocation. Diagnostic stage tracing is opt-in
because synchronizing every stage changes the measurement.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import signal
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def save_record(path: Path, record: dict) -> None:
    """Publish a complete checkpoint; readers never observe truncated JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f".{os.getpid()}.tmp")
    pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def _append(path: Path, event: dict) -> None:
    # A single append keeps parent/watchdog records separate at the deadline.
    with path.open("a") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()


def _watch(
    connection: Any, pid: int, seconds: float, output: Path, record: dict
) -> None:
    """Stop one endpoint even if its calling interpreter cannot run Python."""
    try:
        if connection.poll(seconds):
            return  # Completion or EOF: the owner no longer needs a deadline.
        event = {
            "event": "timeout",
            "endpoint": record["active_endpoint"],
            "limit_seconds": seconds,
        }
        # Evidence writes are best-effort at the deadline: a full or unavailable
        # filesystem must not leave a blocked CUDA endpoint running indefinitely.
        try:
            try:
                _append(output.with_suffix(".progress.jsonl"), event)
            finally:
                record.update(
                    status="stopped", error="endpoint deadline exceeded", stop=event
                )
                save_record(output, record)
        finally:
            os.kill(pid, signal.SIGKILL)
    finally:
        connection.close()


class EndpointProgress:
    """Checkpoint each solve, with optional reference SCF cycle/stage evidence."""

    def __init__(self, output: Path, record: dict, seconds: float, trace: bool) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("endpoint limit must be finite and positive")
        self.output, self.record, self.seconds, self.trace = (
            output,
            record,
            seconds,
            trace,
        )
        self.journal = output.with_suffix(".progress.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.journal.write_text("")
        self.endpoint = "setup"
        self.started = time.monotonic()

    def emit(self, event: str, **values: Any) -> None:
        entry = {
            "event": event,
            "endpoint": self.endpoint,
            "elapsed_seconds": time.monotonic() - self.started,
            **values,
        }
        _append(self.journal, entry)
        print(json.dumps(entry, allow_nan=False), flush=True)

    @contextmanager
    def measure(self, endpoint: str) -> Iterator[None]:
        """Arm outside the solve timer and retain earlier successful samples."""
        self.endpoint, self.started = endpoint, time.monotonic()
        self.record.update(status="running", active_endpoint=endpoint)
        save_record(self.output, self.record)
        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        watchdog = context.Process(
            target=_watch,
            args=(receiver, os.getpid(), self.seconds, self.output, self.record),
            daemon=True,
        )
        try:
            watchdog.start()
        except BaseException:
            receiver.close()
            sender.close()
            raise
        receiver.close()
        try:
            self.emit("begin", limit_seconds=self.seconds)
            yield
        except BaseException as error:
            self.emit("failed", error=f"{type(error).__name__}: {error}")
            raise
        else:
            self.emit("end")
        finally:
            sender.close()  # EOF wakes the watchdog, including on exceptions.
            watchdog.join(timeout=2)
            if watchdog.is_alive():
                watchdog.terminate()
                watchdog.join()

    def instrument_reference(self, engine: Any, cupy: Any) -> None:
        """Chain the sample's convergence tracker without changing SCF policy.

        Kernel wrapping is necessary because the common warm-sample helper
        replaces ``engine.callback`` at the start of every solve. Restoring it
        afterwards prevents nested callback chains on repeated measurements.
        """
        if not self.trace:
            return
        original_kernel, original_veff = engine.kernel, engine.get_veff

        def kernel(*args: Any, **kwargs: Any) -> Any:
            previous = engine.callback

            def callback(environment: dict) -> None:
                if previous is not None:
                    previous(environment)
                values: dict[str, Any] = {}
                for name in ("cycle", "e_tot", "de", "norm_gorb", "norm_ddm"):
                    value = environment.get(name)
                    if value is not None:
                        scalar = float(value)
                        values[name] = scalar if math.isfinite(scalar) else str(scalar)
                self.emit("scf_cycle", **values)

            engine.callback = callback
            try:
                return original_kernel(*args, **kwargs)
            finally:
                engine.callback = previous

        def get_veff(*args: Any, **kwargs: Any) -> Any:
            cupy.cuda.Stream.null.synchronize()
            self.emit("get_veff_begin")
            start = time.monotonic()
            result = original_veff(*args, **kwargs)
            cupy.cuda.Stream.null.synchronize()
            self.emit("get_veff_end", seconds=time.monotonic() - start)
            return result

        engine.kernel, engine.get_veff = kernel, get_veff

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


def _latest_record(output: Path, fallback: dict) -> dict:
    """Read the newest parent checkpoint without trusting a stale spawn snapshot."""
    try:
        loaded = json.loads(output.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return dict(fallback)
    return loaded if isinstance(loaded, dict) else dict(fallback)


def _finite_scalar(value: Any) -> float | None:
    """Return a finite diagnostic scalar without coercing strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


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
        # Checkpoint recovery and evidence writes must not bypass termination
        # after the deadline, even if decoding or another read operation fails.
        try:
            # Preserve progress published after the watchdog was spawned.
            latest = _latest_record(output, record)
            try:
                _append(output.with_suffix(".progress.jsonl"), event)
            finally:
                latest.update(
                    status="stopped", error="endpoint deadline exceeded", stop=event
                )
                save_record(output, latest)
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
        self._diagnostic_summary: dict[str, Any] = {}
        self._last_de_sign: int | None = None

    def emit(self, event: str, **values: Any) -> dict[str, Any]:
        entry = {
            "event": event,
            "endpoint": self.endpoint,
            "elapsed_seconds": time.monotonic() - self.started,
            **values,
        }
        _append(self.journal, entry)
        print(json.dumps(entry, allow_nan=False), flush=True)
        return entry

    def _reset_diagnostic_summary(self, endpoint: str) -> None:
        """Begin a new observation interval, even when its display label is reused."""
        self._diagnostic_summary = {
            "endpoint": endpoint,
            "scf_cycles_observed": 0,
            "get_veff_completed": 0,
            "de_sign_changes": 0,
        }
        self._last_de_sign = None
        self.record["diagnostic_summary"] = dict(self._diagnostic_summary)

    def _update_diagnostic_summary(self, entry: dict[str, Any]) -> None:
        """Retain objective convergence signals without inventing a solver verdict."""
        endpoint = str(entry["endpoint"])
        if self._diagnostic_summary.get("endpoint") != endpoint:
            self._reset_diagnostic_summary(endpoint)

        summary = self._diagnostic_summary
        event = entry["event"]
        elapsed = _finite_scalar(entry.get("elapsed_seconds"))

        if event == "get_veff_begin" and elapsed is not None:
            summary.setdefault("first_get_veff_begin_elapsed_seconds", elapsed)
            summary["last_get_veff_begin_elapsed_seconds"] = elapsed
        elif event == "get_veff_end":
            seconds = _finite_scalar(entry.get("seconds"))
            if seconds is not None:
                count = int(summary["get_veff_completed"]) + 1
                total = float(summary.get("get_veff_total_seconds", 0.0)) + seconds
                summary.update(
                    get_veff_completed=count,
                    get_veff_total_seconds=total,
                    get_veff_mean_seconds=total / count,
                    get_veff_max_seconds=max(
                        float(summary.get("get_veff_max_seconds", 0.0)), seconds
                    ),
                )
        elif event == "scf_cycle":
            count = int(summary["scf_cycles_observed"]) + 1
            summary["scf_cycles_observed"] = count
            cycle = _finite_scalar(entry.get("cycle"))
            if cycle is not None:
                cycle_value: int | float = int(cycle) if cycle.is_integer() else cycle
                summary.setdefault("first_cycle", cycle_value)
                summary["last_cycle"] = cycle_value
            if elapsed is not None:
                summary.setdefault("first_cycle_elapsed_seconds", elapsed)
                summary["last_cycle_elapsed_seconds"] = elapsed
                first_elapsed = float(summary["first_cycle_elapsed_seconds"])
                summary["cycle_elapsed_span_seconds"] = elapsed - first_elapsed
                if count > 1:
                    summary["mean_cycle_interval_seconds"] = (
                        elapsed - first_elapsed
                    ) / (count - 1)

            latest: dict[str, float] = {}
            for name in ("e_tot", "de", "norm_gorb", "norm_ddm"):
                value = _finite_scalar(entry.get(name))
                if value is not None:
                    latest[name] = value
            if latest:
                summary["latest_cycle_values"] = latest

            de = latest.get("de")
            if de is not None and de != 0.0:
                sign = 1 if de > 0.0 else -1
                if self._last_de_sign is not None and sign != self._last_de_sign:
                    summary["de_sign_changes"] = int(summary["de_sign_changes"]) + 1
                self._last_de_sign = sign

            for name in ("norm_gorb", "norm_ddm"):
                value = latest.get(name)
                if value is None:
                    continue
                summary[f"{name}_latest"] = value
                minimum = summary.get(f"{name}_min")
                summary[f"{name}_min"] = (
                    value if minimum is None else min(float(minimum), value)
                )

        self.record["diagnostic_summary"] = dict(summary)

    def checkpoint(self, event: str, **values: Any) -> None:
        """Emit a trace event and persist it in the latest result checkpoint."""
        entry = self.emit(event, **values)
        self.record["diagnostic_progress"] = entry
        self._update_diagnostic_summary(entry)
        save_record(self.output, self.record)

    @contextmanager
    def measure(self, endpoint: str) -> Iterator[None]:
        """Arm outside the solve timer and retain earlier successful samples."""
        self.endpoint, self.started = endpoint, time.monotonic()
        # A setup timeout may occur before any diagnostic callback. Publish the
        # new empty interval before spawning its watchdog, not on first callback.
        self._reset_diagnostic_summary(endpoint)
        self.record.pop("diagnostic_progress", None)
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
                self.checkpoint("scf_cycle", **values)

            engine.callback = callback
            try:
                return original_kernel(*args, **kwargs)
            finally:
                engine.callback = previous

        def get_veff(*args: Any, **kwargs: Any) -> Any:
            cupy.cuda.Stream.null.synchronize()
            self.checkpoint("get_veff_begin")
            start = time.monotonic()
            result = original_veff(*args, **kwargs)
            cupy.cuda.Stream.null.synchronize()
            self.checkpoint("get_veff_end", seconds=time.monotonic() - start)
            return result

        engine.kernel, engine.get_veff = kernel, get_veff

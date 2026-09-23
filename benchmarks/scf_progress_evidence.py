"""Analyze retained SCF progress journals without qualifying benchmark samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def read_progress_journal(path: Path) -> list[dict[str, Any]]:
    """Read an append-only progress journal with line-local diagnostics."""
    events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path}:{line_number}: invalid progress JSON: {error.msg}"
            ) from error
        if not isinstance(event, dict):
            raise ValueError(f"{path}:{line_number}: progress event must be an object")
        events.append(event)
    return events


def _segments(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split measured invocations at begin events, including repeated labels."""
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for event in events:
        if event.get("event") == "begin":
            if current:
                segments.append(current)
            current = [event]
        elif current is not None:
            current.append(event)
    if current:
        segments.append(current)
    return segments


def _residual_summary(values: list[tuple[int | float | None, float]]) -> dict[str, Any]:
    first_cycle, first_value = values[0]
    latest_cycle, latest_value = values[-1]
    min_cycle, min_value = min(values, key=lambda item: item[1])
    improving = 0
    non_improving = 0
    longest_non_improving = 0
    current_non_improving = 0
    for (_, previous), (_, current) in zip(values, values[1:], strict=False):
        if current < previous:
            improving += 1
            current_non_improving = 0
        else:
            non_improving += 1
            current_non_improving += 1
            longest_non_improving = max(longest_non_improving, current_non_improving)
    return {
        "samples": len(values),
        "first": first_value,
        "latest": latest_value,
        "minimum": min_value,
        "first_cycle": first_cycle,
        "latest_cycle": latest_cycle,
        "minimum_cycle": min_cycle,
        "improving_steps": improving,
        "non_improving_steps": non_improving,
        "longest_non_improving_run": longest_non_improving,
    }


def analyze_progress_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return objective per-invocation evidence without a convergence verdict."""
    results: list[dict[str, Any]] = []
    for invocation, segment in enumerate(_segments(events)):
        endpoint = str(segment[0].get("endpoint", "unknown"))
        if any(str(event.get("endpoint", endpoint)) != endpoint for event in segment):
            raise ValueError(f"invocation {invocation} mixes endpoint labels")

        veff_seconds = [
            seconds
            for event in segment
            if event.get("event") == "get_veff_end"
            and (seconds := _finite_number(event.get("seconds"))) is not None
            and seconds >= 0.0
        ]
        cycle_events = [event for event in segment if event.get("event") == "scf_cycle"]
        cycle_times = [
            elapsed
            for event in cycle_events
            if (elapsed := _finite_number(event.get("elapsed_seconds"))) is not None
        ]
        cycle_intervals = [
            current - previous
            for previous, current in zip(cycle_times, cycle_times[1:], strict=False)
            if current >= previous
        ]
        first_veff = next(
            (
                elapsed
                for event in segment
                if event.get("event") == "get_veff_begin"
                and (elapsed := _finite_number(event.get("elapsed_seconds")))
                is not None
                and elapsed >= 0.0
            ),
            None,
        )
        de_values = [
            de
            for event in cycle_events
            if (de := _finite_number(event.get("de"))) is not None and de != 0.0
        ]
        signs = [1 if value > 0.0 else -1 for value in de_values]
        de_sign_changes = sum(
            current != previous
            for previous, current in zip(signs, signs[1:], strict=False)
        )

        residuals: dict[str, Any] = {}
        for name in ("norm_gorb", "norm_ddm"):
            values: list[tuple[int | float | None, float]] = []
            for event in cycle_events:
                value = _finite_number(event.get(name))
                if value is None:
                    continue
                cycle = _finite_number(event.get("cycle"))
                cycle_value: int | float | None
                if cycle is None:
                    cycle_value = None
                else:
                    cycle_value = int(cycle) if cycle.is_integer() else cycle
                values.append((cycle_value, value))
            if values:
                residuals[name] = _residual_summary(values)

        terminal = next(
            (
                str(event.get("event"))
                for event in reversed(segment)
                if event.get("event") in {"end", "failed", "timeout"}
            ),
            "incomplete",
        )
        result: dict[str, Any] = {
            "invocation": invocation,
            "endpoint": endpoint,
            "terminal_event": terminal,
            "benchmark_qualification": "diagnostic-only",
            "scf_cycles_observed": len(cycle_events),
            "get_veff_completed": len(veff_seconds),
            "de_sign_changes": de_sign_changes,
            "residuals": residuals,
        }
        if first_veff is not None:
            result["setup_before_first_get_veff_seconds"] = first_veff
        if cycle_times:
            result["first_cycle_elapsed_seconds"] = cycle_times[0]
            result["last_cycle_elapsed_seconds"] = cycle_times[-1]
        if veff_seconds:
            result["get_veff_total_seconds"] = sum(veff_seconds)
            result["get_veff_mean_seconds"] = sum(veff_seconds) / len(veff_seconds)
            result["get_veff_max_seconds"] = max(veff_seconds)
        if cycle_intervals:
            result["cycle_interval_mean_seconds"] = sum(cycle_intervals) / len(
                cycle_intervals
            )
            result["cycle_interval_min_seconds"] = min(cycle_intervals)
            result["cycle_interval_max_seconds"] = max(cycle_intervals)
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema": "vibeqc.scf-diagnostic-evidence.v1",
        "source": str(args.journal),
        "invocations": analyze_progress_events(read_progress_journal(args.journal)),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

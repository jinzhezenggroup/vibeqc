"""Synchronized sample ordering/statistics shared by every tuning consumer."""

import statistics
from collections.abc import Sequence
from typing import Any

BASELINE = "baseline"
CANDIDATE = "candidate"


def interleaved_selection_order(repeats: int, style: str = "abba") -> tuple[str, ...]:
    """Return exactly ``repeats`` baseline and candidate labels.

    ``abba`` uses balanced four-sample blocks and is the default because it
    places each selection on both sides of its counterpart.  ``ab`` is useful
    when a profiler needs a strictly alternating stream.  A short final block
    is truncated without ever changing the requested sample count.
    """

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if style not in {"abba", "ab"}:
        raise ValueError("selection order must be 'abba' or 'ab'")
    block = (BASELINE, CANDIDATE, CANDIDATE, BASELINE)
    if style == "ab":
        block = (BASELINE, CANDIDATE)
    counts = {BASELINE: 0, CANDIDATE: 0}
    order: list[str] = []
    while counts[BASELINE] < repeats or counts[CANDIDATE] < repeats:
        for label in block:
            if counts[label] >= repeats:
                continue
            order.append(label)
            counts[label] += 1
    return tuple(order)


def timing_summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw synchronized samples without discarding any sample."""

    seconds = [float(sample["seconds"]) for sample in samples]
    if not seconds:
        raise ValueError("at least one timing sample is required")
    return {
        "samples": len(seconds),
        "median_seconds": float(statistics.median(seconds)),
        "minimum_seconds": float(min(seconds)),
        "maximum_seconds": float(max(seconds)),
        "raw_seconds": seconds,
    }

"""Reusable synchronized A/B measurements using the existing shell gate order."""

from __future__ import annotations

import time
from math import isfinite
from statistics import median

from benchmarks.aot_shell_batch_gate import interleaved_selection_order, timing_summary

from .schema import WORKLOADS, outcome


def measure_interleaved(
    evaluate,
    synchronize,
    *,
    workload: str,
    inputs_hash: str,
    repeats: int = 5,
    clock=time.perf_counter,
    prepare=None,
) -> list[dict]:
    """Measure identical inputs; evaluate returns serializable result diagnostics.

    Kernel callers must freeze amplitudes/densities. Solver callers must keep
    iteration histories and final residuals in the returned diagnostics. Cold
    callers construct/destroy their executor inside evaluate; replay callers
    retain it. Synchronization brackets every sample, including failures.
    """
    if repeats < 5 or workload not in WORKLOADS:
        raise ValueError("use a known workload and at least five A/B measurements")
    rows = []
    for selection in interleaved_selection_order(repeats):
        if prepare is not None:
            prepare(selection)
        synchronize()
        start = clock()
        try:
            diagnostics = evaluate(selection)
        finally:
            synchronize()
        rows.append(
            {
                "selection": selection,
                "seconds": clock() - start,
                "inputs_hash": inputs_hash,
                "workload": workload,
                "synchronized": True,
                "diagnostics": diagnostics,
            }
        )
    return rows


def assess_comparison(samples: list[dict]) -> dict:
    """Require >2% improvement exceeding robust sample noise for every workload.

    Median absolute deviation / median estimates relative spread. These are
    descriptive gates, not confidence intervals. Runtime alone does not
    authorize promotion; schema validation additionally requires memory,
    compilation cost, independent accuracy, and complete provenance.
    """
    if not samples:
        return outcome("not-run", "no raw measurements")
    summaries = {}
    for workload in dict.fromkeys(row["workload"] for row in samples):
        rows = [row for row in samples if row["workload"] == workload]
        groups = {
            side: [r for r in rows if r["selection"] == side]
            for side in ("baseline", "candidate")
        }
        n = len(groups["baseline"])
        if n < 5 or len(groups["candidate"]) != n:
            return outcome("not-run", "at least five paired A/B measurements required")
        if tuple(r["selection"] for r in rows) not in (
            interleaved_selection_order(n),
            interleaved_selection_order(n, "ab"),
        ):
            return outcome("fail", "measurements must be interleaved")
        if len({r["inputs_hash"] for r in rows}) != 1 or not all(
            r["synchronized"] for r in rows
        ):
            return outcome(
                "fail", "comparison requires identical inputs and synchronization"
            )
        if any(not isfinite(r["seconds"]) or r["seconds"] <= 0 for r in rows):
            return outcome("fail", "timings must be positive and finite")
        sides = {side: timing_summary(group) for side, group in groups.items()}
        spread = {}
        for side, summary in sides.items():
            middle = summary["median_seconds"]
            spread[side] = (
                median(abs(t - middle) for t in summary["raw_seconds"]) / middle
            )
            summary["relative_mad"] = spread[side]
        improvement = (
            1
            - sides["candidate"]["median_seconds"] / sides["baseline"]["median_seconds"]
        )
        noise = max(0.02, 2 * (spread["baseline"] + spread["candidate"]))
        summaries[workload] = {
            **sides,
            "relative_improvement": improvement,
            "noise_floor": noise,
            "significant": improvement > noise,
        }
    passed = all(s["significant"] for s in summaries.values())
    return outcome(
        "pass" if passed else "not-run",
        None
        if passed
        else "difference within 2%, measurement noise, or no improvement",
        workloads=summaries,
    )

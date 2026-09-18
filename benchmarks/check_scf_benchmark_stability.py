"""Validate whether an SCF benchmark supports a cross-engine performance claim.

The normal-convergence benchmark intentionally measures real end-to-end latency,
but repeated fixed-dm0 replays can still follow different SCF branches near a
convergence threshold. Pooling different iteration counts into one timing median
mixes different amounts of work. This checker makes that condition explicit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def iteration_branch(sample: dict[str, Any]) -> tuple[int, ...]:
    """Return the per-system SCF iteration tuple for one warm sample."""

    convergence = sample.get("convergence")
    if not isinstance(convergence, list) or not convergence:
        raise ValueError("each warm sample requires nonempty convergence records")
    counts = []
    for item in convergence:
        if not isinstance(item, dict):
            raise TypeError("convergence records must be objects")
        value = item.get("iterations")
        if type(value) is not int or value < 0:
            raise ValueError("SCF iteration counts must be nonnegative integers")
        counts.append(value)
    return tuple(counts)


def branch_histogram(samples: list[dict[str, Any]]) -> dict[str, int]:
    """Return a JSON-friendly histogram of iteration branches."""

    counts = Counter(iteration_branch(sample) for sample in samples)
    return {
        ",".join(str(value) for value in branch): count
        for branch, count in sorted(counts.items())
    }


def stability_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether the result supports a stable cross-engine ratio."""

    vibeqc_samples = list(payload["vibeqc"]["warm_samples"])
    gpu_samples = list(payload["gpu4pyscf"]["warm_samples"])
    if not vibeqc_samples or not gpu_samples:
        raise ValueError("benchmark must contain warm samples for both engines")

    vibeqc_branches = {iteration_branch(sample) for sample in vibeqc_samples}
    gpu_branches = {iteration_branch(sample) for sample in gpu_samples}
    vibeqc_stable = len(vibeqc_branches) == 1
    gpu_stable = len(gpu_branches) == 1
    shared = vibeqc_branches & gpu_branches
    shared_stable_branch = (
        next(iter(shared))
        if vibeqc_stable and gpu_stable and len(shared) == 1
        else None
    )

    reasons: list[str] = []
    for engine, samples in (("VibeQC", vibeqc_samples), ("GPU4PySCF", gpu_samples)):
        if len(samples) < 2:
            reasons.append(
                f"{engine} needs at least two warm repeats to test stability"
            )
        if any(
            item.get("converged") is not True
            for sample in samples
            for item in sample["convergence"]
        ):
            reasons.append(f"{engine} lacks confirmed convergence for every warm item")
    if not vibeqc_stable:
        reasons.append("VibeQC warm replays follow multiple SCF iteration branches")
    if not gpu_stable:
        reasons.append("GPU4PySCF warm replays follow multiple SCF iteration branches")
    if not shared:
        reasons.append("the two engines have no shared SCF iteration branch")
    elif shared_stable_branch is None:
        reasons.append(
            "the shared branch is not the unique stable branch of both engines"
        )

    headline_valid = not reasons
    return {
        "headline_cross_engine_ratio_valid": headline_valid,
        "scope": "iteration-workload stability; independent numerical gates are still required",
        "interpretation": (
            "stable iteration-matched normal-convergence comparison"
            if headline_valid
            else "inconclusive for a cross-engine SCF performance ratio"
        ),
        "vibeqc": {
            "stable": vibeqc_stable,
            "branch_histogram": branch_histogram(vibeqc_samples),
        },
        "gpu4pyscf": {
            "stable": gpu_stable,
            "branch_histogram": branch_histogram(gpu_samples),
        },
        "shared_stable_branch": (
            list(shared_stable_branch) if shared_stable_branch is not None else None
        ),
        "reasons": reasons,
        "ordinary_ratio_is_diagnostic_only": not headline_valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the stability summary JSON",
    )
    parser.add_argument(
        "--allow-inconclusive",
        action="store_true",
        help="return success even when the result cannot support a headline ratio",
    )
    args = parser.parse_args()

    payload = json.loads(args.result.read_text())
    summary = stability_summary(payload)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")

    if not summary["headline_cross_engine_ratio_valid"] and not args.allow_inconclusive:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

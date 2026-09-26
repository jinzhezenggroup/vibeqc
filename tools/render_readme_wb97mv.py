"""Retain WB97M-V endpoint evidence and plot only completed accepted samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def collect(directory: Path) -> list[dict[str, Any]]:
    """Keep every attempted size, including killed or numerically failed points.

    A running JSON accompanied by exit 124/137 is a timeout, never a timing.
    Missing artifacts stay missing; partial repeats are not turned into medians.
    """
    rows = []
    for atoms in (3, 6, 12, 24, 48, 96):
        row: dict[str, Any] = {"atoms": atoms, "aos": 8 * atoms}
        for group in ("wb97mv", "wb97mv-reference"):
            path = directory / group / f"direct-{atoms}.json"
            outcome_path = path.with_suffix(".outcome")
            outcome = (
                json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
            )
            if not path.exists():
                row[group] = {
                    "status": outcome.get("status", "missing"),
                    "outcome": outcome,
                }
                continue
            raw = path.read_bytes()
            record = json.loads(raw)
            status = record["status"]
            if outcome.get("status") == "interrupted":
                status = "interrupted"
            elif outcome.get("exit_code") in (124, 137):
                status = "timeout"
            elif outcome.get("exit_code", 0) != 0:
                status = "failed"
            row[group] = {
                "status": status,
                "outcome": outcome,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "record": record,
            }
        rows.append(row)
    return rows


def timing(row: dict[str, Any], engine: str) -> dict[str, float] | None:
    """Require three finished repeats and the native independent accuracy gate."""
    groups = ("wb97mv",) if engine == "vibeqc" else ("wb97mv", "wb97mv-reference")
    for group in groups:
        source = row[group]
        record = source.get("record", {})
        samples = record.get(
            "native_samples" if engine == "vibeqc" else "reference_samples", []
        )
        if (
            source["status"] == "measured"
            and len(samples) == 3
            and (engine != "vibeqc" or record.get("accepted") is True)
        ):
            return record["timing"].get(engine)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.raw_directory)
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "endpoints.json").write_text(
        json.dumps(
            {"schema": "vibeqc.readme-wb97mv.evidence.v1", "rows": rows},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    lines = [
        "| Atoms / AOs | VibeQC (s) | GPU4PySCF (s) | Paired status | Reference status |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        native, reference = timing(row, "vibeqc"), timing(row, "gpu4pyscf")
        values = [
            f"{sample['median_seconds']:.3f}" if sample else "—"
            for sample in (native, reference)
        ]
        lines.append(
            f"| {row['atoms']} / {row['aos']} | {' | '.join(values)} | {row['wb97mv']['status']} | "
            f"{'measured' if reference else row['wb97mv-reference']['status']} |"
        )
    (args.destination / "table.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for engine, label, color in (
        ("vibeqc", "VibeQC (accuracy accepted)", "#1967a3"),
        ("gpu4pyscf", "GPU4PySCF", "#d46b18"),
    ):
        measured = [(r["aos"], timing(r, engine)) for r in rows if timing(r, engine)]
        if not measured:
            continue
        x = [aos for aos, _ in measured]
        y = [sample["median_seconds"] for _, sample in measured]
        lower = [
            sample["median_seconds"] - sample["min_seconds"] for _, sample in measured
        ]
        upper = [
            sample["max_seconds"] - sample["median_seconds"] for _, sample in measured
        ]
        ax.errorbar(
            x, y, yerr=[lower, upper], fmt="o-", capsize=4, color=color, label=label
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(
        [24, 48, 96, 192, 384, 768], labels=["24", "48", "96", "192", "384", "768"]
    )
    ax.set_xlabel("Spherical def2-SVP AOs (3–96 atoms)")
    ax.set_ylabel("Complete warm SCF + analytic forces (seconds)")
    ax.set_title("WB97M-V · RTX 5090 · three interleaved repeats")
    ax.grid(alpha=0.2)
    if ax.lines:
        ax.legend()
    fig.text(
        0.5,
        -0.015,
        "Missing/failed/timed-out points are omitted; see the evidence table.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(args.destination / "latency.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

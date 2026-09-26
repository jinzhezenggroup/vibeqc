"""Retain every complete HF endpoint and render the README's direct/DF figure.

This reducer verifies all samples against their approximation's independent
reference, including cold, moved and diagnostic calls. Timings are never divided
by iteration counts, and trace samples never enter clean endpoint medians.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from benchmarks.compare_df_direct_endpoint import check_endpoint


def digest(path: Path) -> str:
    """Pin the raw measurements used for a compact retained record."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_sample(row: dict) -> dict:
    """Keep all scalar convergence/work evidence without repeated force arrays."""
    return {
        key: value
        for key, value in row.items()
        if key not in ("forces", "trace", "metric")
    }


def trace_work(records: list[dict]) -> dict:
    """Separate graph construction from actual stream observations and work."""
    grouped: dict[str, Any] = {}
    for operation in records:
        if (
            not operation["valid"]
            or operation["cuda_error"]
            or operation["dropped_regions"]
            or operation["dropped_tiles"]
        ):
            raise ValueError("incomplete DF work trace")
        key = operation["operation"] + "/" + operation["execution"]
        bucket = grouped.setdefault(key, {"records": 0, "counters": Counter()})
        bucket["records"] += 1
        bucket["counters"].update(operation["counters"])
    return grouped


def checked_run(
    native_path: Path, reference_path: Path, repeats: int
) -> tuple[dict, dict]:
    """Reject mismatched protocols, missing phases or even one inaccurate sample."""
    native = json.loads(native_path.read_text())
    reference = json.loads(reference_path.read_text())
    identity = native["identity"]
    if identity["protocol"] != reference["identity"]["protocol"]:
        raise ValueError("native/reference scientific protocol mismatch")
    if identity["reference_sha256"] != digest(reference_path):
        raise ValueError("native run used a different independent reference")
    if len(reference["records"]) != 2:
        raise ValueError("missing original or moved independent reference")
    all_reference = reference["records"] + reference["reference_samples"]
    groups: dict[str, list] = defaultdict(list)
    reference_groups: dict[str, list] = defaultdict(list)
    for row in native["records"]:
        index = int("moved" in row["phase"])
        checked = check_endpoint(
            SimpleNamespace(
                energy=row["energy"],
                forces=np.asarray(row["forces"]),
                converged=row["converged"],
                succeeded=row["status"] == 0,
            ),
            reference["records"][index],
        )
        if not checked["gate"] or row["gate"] is not True:
            raise ValueError(
                "a native endpoint failed independent energy/force acceptance"
            )
        if not math.isfinite(row["complete_seconds"]) or row["complete_seconds"] <= 0:
            raise ValueError("invalid native endpoint timing")
        if not row["diagnostic"]:
            groups[row["phase"]].append(row)
    for row in all_reference:
        checked = check_endpoint(
            SimpleNamespace(
                energy=row["energy"],
                forces=np.asarray(row["forces"]),
                converged=row["converged"],
                succeeded=True,
            ),
            reference["records"][row["geometry"]],
        )
        if (
            not checked["gate"]
            or not math.isfinite(row["complete_seconds"])
            or row["complete_seconds"] <= 0
        ):
            raise ValueError("invalid reference endpoint")
        reference_groups[row["phase"]].append(row)
    expected = {"cold": 1, "warm": repeats, "moved": 1, "moved-warm": repeats}
    if (
        dict(Counter({k: len(v) for k, v in groups.items()})) != expected
        or {k: len(v) for k, v in reference_groups.items()} != expected
    ):
        raise ValueError("missing/duplicate cold, warm or changed-geometry endpoints")
    compact = {
        "native_identity": identity,
        "reference_identity": reference["identity"],
        "raw_native_sha256": digest(native_path),
        "raw_reference_sha256": digest(reference_path),
        "native_samples": [scalar_sample(row) for row in native["records"]],
        "reference_samples": [scalar_sample(row) for row in all_reference],
        "independent_references": reference["records"],
        "diagnostic_work": [
            {
                "phase": row["phase"],
                "iterations": row["iterations"],
                "operations": trace_work(row["trace"]),
            }
            for row in native["records"]
            if row["diagnostic"]
        ],
        "medians": {
            phase: {
                "vibeqc_seconds": statistics.median(
                    r["complete_seconds"] for r in rows
                ),
                "gpu4pyscf_seconds": statistics.median(
                    r["complete_seconds"] for r in reference_groups[phase]
                ),
                "native_iterations": sorted({r["iterations"] for r in rows}),
                "reference_iterations": sorted(
                    {r["iterations"] for r in reference_groups[phase]}
                ),
                "reference_scf_jk_builds": sorted(
                    {r["scf_jk_builds"] for r in reference_groups[phase]}
                ),
            }
            for phase, rows in groups.items()
        },
    }
    plot = {
        "aos": identity["protocol"]["aos"],
        "status": "measured",
        "engines": {},
    }
    for engine, rows in (
        ("VibeQC", groups["warm"]),
        ("GPU4PySCF", reference_groups["warm"]),
    ):
        plot["engines"][engine] = {
            "samples": [
                {
                    "ms": row["complete_seconds"] * 1000,
                    "convergence": [{"iterations": row["iterations"]}],
                }
                for row in rows
            ]
        }
    return compact, plot


def main() -> None:
    """Write small workload companions and a checksum-bound summary."""
    from tools.render_readme_benchmarks import plot_series, plt, save_svg, style_axes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    plots: dict[str, list] = defaultdict(list)
    summary: dict[str, Any] = {
        "schema": "vibeqc.hf-unified-acceptance.v1",
        "repeats": args.repeats,
        "scope": "same native FP64 SCF acceptance; complete energy and analytic forces; explicit packed-single fitted DF",
        "timing": "separate engine processes; cold includes prepare; five frozen warm replays; traces excluded from medians",
        "work": "SCF iterations are not Fock counts; GPU4PySCF get_veff includes pre-loop; native public Fock counts stay null; DF trace counters retained",
        "cases": [],
        "parts": [],
        "native_endpoint_samples": 0,
        "maximum_energy_error": 0.0,
        "maximum_force_error": 0.0,
    }
    hashes = set()
    for aos in (24, 48, 96, 192, 384, 768):
        for route in ("direct", "df"):
            parent = args.raw_directory / str(aos)
            record, plot = checked_run(
                parent / route / "results.json",
                parent / ("reference-" + route) / "results.json",
                args.repeats,
            )
            hashes.add(record["native_identity"]["native_build"]["library_sha256"])
            plots[route].append(plot)
            summary["cases"].append(
                {"aos": aos, "route": route, "medians": record["medians"]}
            )
            summary["native_endpoint_samples"] += len(record["native_samples"])
            for field in ("energy", "force"):
                key = "maximum_" + field + "_error"
                summary[key] = max(
                    summary[key],
                    *(r[field + "_error"] for r in record["native_samples"]),
                )
            if record["diagnostic_work"]:
                # Keep the full diagnostic work as an observable-specific list
                # using the repository's checksum-verified record loader.
                work_path = args.destination / f"{route}-{aos}-work.json"
                work_path.write_text(
                    json.dumps(record.pop("diagnostic_work"), indent=2, allow_nan=False)
                    + "\n"
                )
                part = {
                    "path": work_path.name,
                    "sha256": digest(work_path),
                    "bytes": work_path.stat().st_size,
                }
                record["record_parts"] = {"diagnostic_work": [part]}
                summary["parts"].append(part)
            filename = f"{route}-{aos}.json"
            path = args.destination / filename
            path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            summary["parts"].append(
                {"path": filename, "sha256": digest(path), "bytes": path.stat().st_size}
            )
    if len(hashes) != 1:
        raise ValueError("direct/DF were not measured on the same native library")
    summary["library_sha256"] = hashes.pop()
    (args.destination / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "vibeqc-hf-unified",
        }
    )
    # A shared time scale also makes direct versus DF readable across panels.
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8), sharey=True)
    for ax, route, title in zip(
        axes,
        ("direct", "df"),
        ("HF · direct J/K", "HF · DF J/K (cc-pVDZ-JKFIT)"),
        strict=True,
    ):
        for engine in ("VibeQC", "GPU4PySCF"):
            plot_series(ax, plots[route], engine, engine)
        style_axes(ax, title, [24, 48, 96, 192, 384, 768])
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle(
        f"RTX 5090 · spherical def2-SVP · complete warm energy + forces · {args.repeats} repeats",
        fontsize=12,
    )
    fig.tight_layout()
    save_svg(fig, args.destination / "hf.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()

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


def compact_table(samples: list[dict]) -> dict:
    """Keep exact scalar observations once per row, including missing work counts."""
    if not samples:
        raise ValueError("empty endpoint table")
    columns = list(samples[0])
    if any(set(sample) != set(columns) for sample in samples):
        raise ValueError("endpoint scalar fields differ")
    return {
        "columns": columns,
        "rows": [[sample[k] for k in columns] for sample in samples],
    }


def write_collection(path: Path, records: list[dict]) -> dict:
    """Retain one workload per line and verify lossless JSON before hashing it.

    Grouped work counters and independent arrays remain complete. This replaces
    per-size companion files without introducing archives or duplicating arrays.
    """
    path.write_text(
        "[\n" + ",\n".join(json.dumps(r, allow_nan=False) for r in records) + "\n]\n"
    )
    if json.loads(path.read_text()) != records:
        raise ValueError("compact retention changed measured values")
    return {"path": path.name, "sha256": digest(path), "bytes": path.stat().st_size}


def draw_hf(plots: dict[str, list], destination: Path, repeats: int) -> None:
    """Compare both engines and both J/K methods on one complete-endpoint scale.

    Every successful repeat enters the median/range, even when iteration counts
    differ. Crosses additionally expose those variable-work repeats; connecting
    their medians makes no equal-work or iteration-normalized timing claim.
    """
    from tools.render_readme_benchmarks import COLORS, plt, save_svg, style_axes

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "vibeqc-hf-comparison",
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    for engine in ("VibeQC", "GPU4PySCF"):
        for route, label, style in (
            ("direct", "direct J/K", "o-"),
            ("df", "DF J/K", "s--"),
        ):
            points = sorted(plots[route], key=lambda p: p["aos"])
            values = [
                [r["ms"] for r in p["engines"][engine]["samples"]] for p in points
            ]
            medians = [statistics.median(v) for v in values]
            ax.errorbar(
                [p["aos"] for p in points],
                medians,
                yerr=[
                    [m - min(v) for m, v in zip(medians, values, strict=True)],
                    [max(v) - m for m, v in zip(medians, values, strict=True)],
                ],
                fmt=style,
                color=COLORS[engine],
                label=f"{engine} · {label}",
                capsize=3,
                linewidth=1.8,
                markersize=4,
            )
            for point, times in zip(points, values, strict=True):
                samples = point["engines"][engine]["samples"]
                if len({r["convergence"][0]["iterations"] for r in samples}) > 1:
                    ax.scatter(
                        [point["aos"]] * len(times),
                        times,
                        marker="x",
                        color=COLORS[engine],
                        s=25,
                        zorder=4,
                    )
    style_axes(ax, "RHF · spherical def2-SVP · RTX 5090", [24, 48, 96, 192, 384, 768])
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.text(
        0.5,
        0.01,
        f"Warm energy + forces · median and min–max of {repeats} runs · × varying SCF iterations",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_svg(fig, destination / "hf.svg")
    plt.close(fig)


def main() -> None:
    """Validate native/reference endpoints and emit one consolidated comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument(
        "--reference-directory",
        type=Path,
        help="Defaults to raw-directory; allows independently retained reference runs",
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--include-warm-controls",
        action="store_true",
        help="Retain 384/768-AO disabled-reuse ablations in evidence, outside the README plot",
    )
    args = parser.parse_args()
    reference_directory = args.reference_directory or args.raw_directory
    args.destination.mkdir(parents=True, exist_ok=True)
    plots: dict[str, list] = defaultdict(list)
    summary = {
        "schema": "vibeqc.hf-comparison.v2",
        "repeats": args.repeats,
        "scope": "complete RHF energy and analytic forces; explicit packed-single fitted DF",
        "timing": "separate engine processes; cold includes prepare; frozen warm replays; traces excluded",
        "figure": "VibeQC/GPU4PySCF direct/DF medians and min-max over every warm repeat; crosses mark varying iteration counts",
        "cases": [],
        "parts": [],
        "native_endpoint_samples": 0,
        "reference_endpoint_samples": 0,
        "maximum_energy_error": 0.0,
        "maximum_force_error": 0.0,
    }
    native_records, reference_records, work_records = [], [], []
    for aos in (24, 48, 96, 192, 384, 768):
        routes = ["direct", "df"]
        if args.include_warm_controls and aos >= 384:
            routes.append("df-disabled")
        for route in routes:
            approximation = "direct" if route == "direct" else "df"
            record, plot = checked_run(
                args.raw_directory / str(aos) / route / "results.json",
                reference_directory
                / str(aos)
                / ("reference-" + approximation)
                / "results.json",
                args.repeats,
            )
            identity = record["native_identity"]
            disabled = identity["environment"].get("VIBEQC_DF_WARM_REUSE") == "0"
            if disabled != (route == "df-disabled"):
                raise ValueError("warm-reuse control differs from its label")
            shared = {
                key: identity[key]
                for key in (
                    "native_build",
                    "source_revision",
                    "native_acceptance",
                    "packages",
                    "slurm_job_id",
                )
            }
            if "native_identity" not in summary:
                summary["native_identity"] = shared
            if summary["native_identity"] != shared:
                raise ValueError(
                    "native source/build/acceptance/job differs across points"
                )
            native_records.append(
                {
                    "aos": aos,
                    "route": route,
                    "reference_route": approximation,
                    "raw_native_sha256": record["raw_native_sha256"],
                    "raw_reference_sha256": record["raw_reference_sha256"],
                    "environment": identity["environment"],
                    "samples": compact_table(record["native_samples"]),
                }
            )
            if route != "df-disabled":
                reference_records.append(
                    {
                        "aos": aos,
                        "route": route,
                        "raw_reference_sha256": record["raw_reference_sha256"],
                        "identity": record["reference_identity"],
                        "independent_references": record["independent_references"],
                        "samples": record["reference_samples"],
                    }
                )
                summary["reference_endpoint_samples"] += len(
                    record["reference_samples"]
                )
                plots[route].append(plot)
            if record["diagnostic_work"]:
                work_records.append(
                    {
                        "aos": aos,
                        "route": route,
                        "diagnostics": record["diagnostic_work"],
                    }
                )
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
    for filename, records in (
        ("samples.json", native_records),
        ("references.json", reference_records),
        ("work.json", work_records),
    ):
        summary["parts"].append(write_collection(args.destination / filename, records))
    (args.destination / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    draw_hf(plots, args.destination, args.repeats)


if __name__ == "__main__":
    main()

"""Retain and plot complete native direct/DF warm-reuse qualification.

Independent reference arrays remain in their earlier checksum-pinned records.
Every new endpoint is rechecked against those arrays before compact retention;
sample tables preserve all scalar values and every diagnostic work counter.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.render_hf_acceptance_benchmarks import checked_run, digest
from tools.vibeqc_validation.record import load_record


def write_samples(path: Path, metadata: dict, samples: list[dict]) -> None:
    """Use one row per endpoint, with explicit columns and exact JSON scalars."""
    columns = list(samples[0])
    if any(set(row) != set(columns) for row in samples):
        raise ValueError("endpoint scalar fields differ")
    rows = [[sample[key] for key in columns] for sample in samples]
    # Keeping scalar table rows on one line makes repeated endpoints easier to
    # compare while avoiding repeated field names and large whitespace blocks.
    # Diagnostic counters are already grouped by operation, not profiler rows.
    # Keep one complete phase per line so retention stays within the PR budget
    # without dropping counters or duplicating the independent force arrays.
    header = {k: v for k, v in metadata.items() if k != "diagnostic_work"}
    text = json.dumps(header, indent=2, allow_nan=False)[:-2]
    text += ',\n  "diagnostic_work": [\n'
    text += ",\n".join(
        "    " + json.dumps(row, separators=(",", ":"), allow_nan=False)
        for row in metadata["diagnostic_work"]
    )
    text += "\n  ]"
    text += ',\n  "columns": ' + json.dumps(columns) + ',\n  "samples": [\n'
    text += ",\n".join("    " + json.dumps(row, allow_nan=False) for row in rows)
    path.write_text(text + "\n  ]\n}\n")
    restored = json.loads(path.read_text())
    if [dict(zip(columns, row, strict=True)) for row in restored["samples"]] != samples:
        raise ValueError("compact retention changed scalar observations")
    if {
        k: v for k, v in restored.items() if k not in ("columns", "samples")
    } != metadata:
        raise ValueError("compact retention changed metadata or work counters")


def main() -> None:
    """Reject incomplete measurements and emit checksum-bound native evidence."""
    from tools.render_readme_benchmarks import plot_series, plt, save_svg, style_axes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "vibeqc.df-warm-reuse.v1",
        "repeats": 5,
        "timing": "complete native energy and analytic forces; frozen warm repeats; traces excluded",
        "control": "same native binary, independent owners; DF warm reuse disabled at 384/768 AOs",
        "reference": "independent approximation-matched arrays retained in the earlier qualification; no new reference timing claim",
        "cases": [],
        "parts": [],
        "native_samples": 0,
        "maximum_energy_error": 0.0,
        "maximum_force_error": 0.0,
    }
    plots = {route: [] for route in ("direct", "df", "df-disabled")}
    libraries = set()
    for aos in (24, 48, 96, 192, 384, 768):
        routes = ("direct", "df", "df-disabled") if aos >= 384 else ("direct", "df")
        for route in routes:
            approximation = "direct" if route == "direct" else "df"
            compact, plot = checked_run(
                args.raw_directory / str(aos) / route / "results.json",
                args.reference_directory
                / str(aos)
                / ("reference-" + approximation)
                / "results.json",
                5,
            )
            old_path = args.reference_records / f"{approximation}-{aos}.json"
            old = load_record(old_path)
            identity = compact["native_identity"]
            reuse_disabled = identity["environment"].get("VIBEQC_DF_WARM_REUSE") == "0"
            if reuse_disabled != (route == "df-disabled"):
                raise ValueError("warm-reuse control differs from its label")
            if (
                old["raw_reference_sha256"] != compact["raw_reference_sha256"]
                or old["independent_references"] != compact["independent_references"]
                or old["native_identity"]["protocol"] != identity["protocol"]
            ):
                raise ValueError(
                    "retained reference does not match the measured oracle"
                )
            libraries.add(identity["native_build"]["library_sha256"])
            if "native_build" not in summary:
                summary["native_build"] = identity["native_build"]
                summary["source_revision"] = identity["source_revision"]
                summary["native_acceptance"] = identity["native_acceptance"]
                summary["packages"] = identity["packages"]
                summary["slurm_job_id"] = identity["slurm_job_id"]
            if (
                summary["native_build"] != identity["native_build"]
                or summary["source_revision"] != identity["source_revision"]
                or summary["native_acceptance"] != identity["native_acceptance"]
                or summary["packages"] != identity["packages"]
                or summary["slurm_job_id"] != identity["slurm_job_id"]
            ):
                raise ValueError(
                    "native source/build/acceptance/job differs across points"
                )
            metadata = {
                "aos": aos,
                "route": route,
                "protocol_and_independent_reference": {
                    "path": os.path.relpath(old_path, args.destination),
                    "sha256": digest(old_path),
                    "bytes": old_path.stat().st_size,
                },
                "raw_native_sha256": compact["raw_native_sha256"],
                "raw_reference_sha256": compact["raw_reference_sha256"],
                "environment": identity["environment"],
                "diagnostic_work": compact["diagnostic_work"],
            }
            path = args.destination / f"{route}-{aos}.json"
            write_samples(path, metadata, compact["native_samples"])
            summary["parts"].append(
                {
                    "path": path.name,
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                }
            )
            medians = {
                phase: {
                    k: v
                    for k, v in values.items()
                    if k in ("vibeqc_seconds", "native_iterations")
                }
                for phase, values in compact["medians"].items()
            }
            summary["cases"].append({"aos": aos, "route": route, "medians": medians})
            summary["native_samples"] += len(compact["native_samples"])
            for field in ("energy", "force"):
                key = "maximum_" + field + "_error"
                summary[key] = max(
                    summary[key],
                    *(r[field + "_error"] for r in compact["native_samples"]),
                )
            plots[route].append(plot)
    if len(libraries) != 1:
        raise ValueError("warm control and direct/DF libraries differ")
    (args.destination / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "vibeqc-df-warm",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 3.8))
    plot_series(ax, plots["direct"], "VibeQC", "Direct J/K", color="#167568")
    plot_series(ax, plots["df"], "VibeQC", "DF J/K · warm reuse", color="#c95e22")
    plot_series(
        ax,
        plots["df-disabled"],
        "VibeQC",
        "DF J/K · reuse off",
        color="#777777",
        style="--",
    )
    style_axes(
        ax,
        "RTX 5090 · spherical def2-SVP · complete warm energy + forces",
        [24, 48, 96, 192, 384, 768],
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    save_svg(fig, args.destination / "hf.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()

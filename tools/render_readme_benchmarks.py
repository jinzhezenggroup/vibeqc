"""Reduce measured endpoint records and draw diagnostic latency figures.

This is an evidence reduction step, not a benchmark runner. Raw endpoint arrays stay in
ignored storage; their hashes and every scalar timing/convergence observation
remain in the compact summary. No cross-engine speedup is inferred here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"VibeQC": "#136f63", "GPU4PySCF": "#c05c26"}


def save_svg(fig: plt.Figure, path: Path) -> None:
    """Strip Matplotlib path-line whitespace for a clean, stable Git diff."""
    fig.savefig(path, metadata={"Date": None})
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


def convergence(rows: list[dict]) -> list[dict]:
    """Keep residuals and work counts; basis identities are retained once."""
    return [
        {key: value for key, value in item.items() if key != "basis_metadata"}
        for item in rows
    ]


def samples(rows: list[dict]) -> list[dict]:
    """Retain measured work and residuals while omitting large physical arrays."""
    return [
        {
            "ms": row["seconds"] * 1000,
            "convergence": convergence(row["convergence"]),
        }
        for row in rows
    ]


def reduce_point(path: Path, root: Path) -> tuple[dict, dict]:
    """Read one actual endpoint; failed points never acquire successful timings."""
    raw = json.loads(path.read_text())
    record = {
        "raw_file": str(path.relative_to(root)),
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    env = raw.get("environment", {}).copy()
    # Share static provenance while retaining each point's actual timestamp and
    # sampled device state. This avoids repeating the toolchain dozens of times.
    record["timestamp_utc"] = env.pop("timestamp_utc", None)
    if env.get("accelerator"):
        env["accelerator"] = env["accelerator"].copy()
        record["device_state"] = env["accelerator"].pop("nvidia_smi", None)
        if "workload" not in raw and record["device_state"]:
            # The DFT/CC driver calls this helper before running the endpoints;
            # its generic "after benchmark" label describes the HF caller only.
            state = record["device_state"]
            state["raw_sampling_point_label"] = state["sampling_point"]
            state["sampling_point"] = "before endpoint measurements"
    if "workload" in raw:
        work = raw["workload"]
        record.update(
            family="hf_energy" if work["properties"] == ["energy"] else "hf",
            method="RHF",
            mode="df" if work["density_fitting"] == "cuda" else "direct",
            atoms=len(work["geometries"][0]),
            aos=work["ao_count"],
            endpoint="SCF energy"
            if work["properties"] == ["energy"]
            else "energy + analytic forces",
            basis="def2-SVP spherical",
            auxiliary_basis=work["auxiliary_basis"],
            status="measured" if raw["gate"]["passed"] else "failed",
            accuracy=raw["accuracy"]["gate_selection"],
            accuracy_pairs=raw["accuracy"]["paired_warm_repeats"],
            gates=raw["settings"]["gates"],
            engines={},
        )
        for engine, label in (("vibeqc", "VibeQC"), ("gpu4pyscf", "GPU4PySCF")):
            record["engines"][label] = {
                "cold_ms": raw[engine]["cold_seconds"] * 1000,
                "cold_convergence": convergence(raw[engine]["cold_convergence"]),
                "samples": samples(raw[engine]["warm_samples"]),
            }
        build = raw.get("native_build")
        record["df_work"] = raw["settings"]["density_fitting_metric_diagnostics"]
        record["basis_identity"] = {
            role: {key: value for key, value in basis.items() if key != "electrons"}
            if isinstance(basis, dict)
            else basis
            for role, basis in raw["vibeqc"]["cold_convergence"][0]
            .get("basis_metadata", {})
            .items()
        }
    else:
        for key in (
            "method",
            "mode",
            "aos",
            "basis",
            "endpoint",
            "status",
            "accuracy",
            "accuracy_pairs",
            "gates",
            "grid_points",
            "grid_identity",
            "grid_export",
            "native_unavailable",
            "native_not_measured",
            "reference_strict_grid_order",
            "reference_empty_ao_block_adapter",
            "error",
            "maximum_energy_error_hartree",
            "maximum_triples_error_hartree",
            "convergence_gate_passed",
            "benchmark_sha256",
        ):
            if key in raw:
                record[key] = raw[key]
        build = raw.get("native_build")
        record["engines"] = {}
        if raw.get("method") == "CCSD(T)":
            record.update(
                family="cc", molecule=raw["molecule"], atoms=len(raw["atoms"])
            )
            record["validation_reference"] = {
                key: raw[key]
                for key in (
                    "input_sha256",
                    "triples_reference_sha256",
                    "reference_energy_hartree",
                    "reference_triples_hartree",
                    "solver_options",
                    "triples_virtual_chunk",
                )
                if key in raw
            }
            reference_shape = raw.get("native_cold", {})
            if "nocc" in reference_shape:
                record.update(
                    nocc=reference_shape["nocc"],
                    nvir=reference_shape["nvir"],
                    aos=reference_shape["nocc"] + reference_shape["nvir"],
                )
            for engine, label in (("native", "VibeQC"),):
                converted = []
                for row in raw.get(engine + "_samples", []):
                    diag = row.get("correlation", row)
                    converted.append(
                        {
                            "ms": row["seconds"] * 1000,
                            "converged": row["converged"],
                            "scf_iterations": row["scf_iterations"],
                            "ccsd_iterations": diag["ccsd_iterations"],
                            "triples_energy": diag.get(
                                "ccsd_t_triples_energy", diag.get("triples_energy")
                            ),
                            "virtual_triples": diag.get("virtual_triples"),
                            "replay_singles_residual_max": diag.get(
                                "replay_singles_residual_max"
                            ),
                            "replay_doubles_residual_max": diag.get(
                                "replay_doubles_residual_max"
                            ),
                            "provider_work": row.get("provider_work"),
                            "cuda_provenance": row.get("provenance"),
                        }
                    )
                if converted:
                    record["engines"][label] = {
                        "cold_ms": raw[engine + "_cold"]["seconds"] * 1000,
                        "cold_work": {
                            key: value
                            for key, value in raw[engine + "_cold"].items()
                            if key not in ("seconds", "provenance")
                        },
                        "samples": converted,
                    }
        else:
            record.update(
                family="dft",
                atoms=raw["atoms"] if "atoms" in raw else raw["arguments"]["atoms"],
            )
            for engine, label in (("native", "VibeQC"), ("reference", "GPU4PySCF")):
                rows = raw.get(engine + "_samples", [])
                if rows:
                    record["engines"][label] = {
                        "cold_ms": raw[engine + "_cold"]["seconds"] * 1000,
                        "cold_convergence": convergence(
                            raw[engine + "_cold"]["convergence"]
                        ),
                        "samples": samples(rows),
                    }
            record["native_prepare_ms"] = (
                1000 * raw["native_prepare_seconds"]
                if "native_prepare_seconds" in raw
                else None
            )
        record["gates_passed"] = raw.get("gates_passed", False)
    provenance = {"environment": env, "native_build": build}
    if raw.get("grid"):
        provenance["grid"] = raw["grid"]
    return record, provenance


def plot_series(
    ax: Any,
    points: list[dict],
    engine: str,
    label: str,
    *,
    style: str = "-",
    color: str | None = None,
) -> None:
    """Plot stable branch medians and min/max; unstable branches stay unpooled."""
    x, y, lower, upper = [], [], [], []
    unstable_x, unstable_y = [], []
    for point in sorted(points, key=lambda p: p.get("aos", 0)):
        if (
            point["status"] not in ("measured", "reference_only")
            or engine not in point["engines"]
            or not point.get("plot", True)
        ):
            continue
        observations = point["engines"][engine]["samples"]
        times = np.array([row["ms"] for row in observations])
        branches = {
            tuple(c["iterations"] for c in row["convergence"]) for row in observations
        }
        if len(branches) != 1:
            unstable_x.extend([point["aos"]] * len(times))
            unstable_y.extend(times)
            continue
        x.append(point["aos"])
        median = float(np.median(times))
        y.append(median)
        lower.append(median - min(times))
        upper.append(max(times) - median)
    if x:
        ax.errorbar(
            x,
            y,
            yerr=[lower, upper],
            fmt="o" + style,
            color=color or COLORS[engine],
            label=label,
            capsize=3,
            linewidth=1.8,
            markersize=4,
        )
    if unstable_x:
        ax.scatter(
            unstable_x,
            unstable_y,
            marker="x",
            color=color or COLORS[engine],
            label=None if x else label,
        )


def style_axes(ax: Any, title: str, ticks: list[int]) -> None:
    ax.set_title(title, loc="left", weight="bold", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ticks, [str(x) for x in ticks])
    ax.set_xlabel("Spherical AOs")
    ax.set_ylabel("Complete endpoint / ms")
    ax.grid(True, alpha=0.17, which="both")
    ax.spines[["top", "right"]].set_visible(False)


def figures(records: list[dict], destination: Path) -> None:
    """Keep hardware/property boundaries visible in every standalone artifact."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "vibeqc-readme-20260922",
        }
    )
    for family, filename, endpoint in (("hf", "hf.svg", "warm energy + forces"),):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6))
        for ax, mode, title in zip(
            axes,
            ("direct", "df"),
            ("HF · direct J/K", "HF · DF J/K (cc-pVDZ-JKFIT)"),
            strict=True,
        ):
            points = [p for p in records if p["family"] == family and p["mode"] == mode]
            for engine in ("VibeQC", "GPU4PySCF"):
                plot_series(ax, points, engine, engine)
            style_axes(ax, title, [24, 48, 96, 192, 384, 768])
            stopped = [p for p in points if p["status"] == "stopped"]
            if stopped:
                ax.text(
                    0.03,
                    0.96,
                    "96 atoms: run stopped at 900 s\nNo complete endpoint timing",
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                )
            ax.legend(frameon=False, loc="lower right")
        fig.suptitle(f"RTX 5090 · def2-SVP · {endpoint} · 3 repeats", fontsize=12)
        fig.tight_layout()
        save_svg(fig, destination / filename)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    for ax, method, title in zip(
        axes,
        ("pbe-rks", "r2scan-rks", "pbe0-rks"),
        ("PBE (GGA)", "r²SCAN (meta-GGA)", "PBE0 (hybrid)"),
        strict=True,
    ):
        points = [
            p for p in records if p["family"] == "dft" and p.get("method") == method
        ]
        direct = [p for p in points if p["mode"] == "direct"]
        fitted = [p for p in points if p["mode"] == "df"]
        plot_series(ax, direct, "VibeQC", "VibeQC direct")
        plot_series(ax, direct, "GPU4PySCF", "GPU4PySCF direct")
        plot_series(
            ax, fitted, "GPU4PySCF", "GPU4PySCF DF", style="--", color="#4669a1"
        )
        style_axes(ax, title, [24, 48, 96, 192, 384, 768])
        coverage = {
            "pbe-rks": "VibeQC direct: 3/6 atoms measured",
            "r2scan-rks": "96-atom reference run stopped at 120 s\nVibeQC direct: 3 atoms measured",
            "pbe0-rks": "Only 3-atom DF smoke completed\nFurther runs cancelled",
        }[method]
        ax.text(
            0.04,
            0.96,
            "VibeQC DF: unavailable"
            + ("\nVibeQC CUDA hybrid: unavailable" if method == "pbe0-rks" else "")
            + "\n"
            + coverage,
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
        )
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "RTX 5090 · def2-SVP · identical grids · warm SCF energy · partial campaign",
        fontsize=12,
    )
    fig.tight_layout()
    save_svg(fig, destination / "dft.svg")
    plt.close(fig)

    points = sorted(
        [p for p in records if p["family"] == "cc" and p["status"] == "measured"],
        key=lambda p: p["atoms"],
    )
    fig, ax = plt.subplots(figsize=(11.6, 3.1))
    for index, point in enumerate(points):
        observations = point["engines"]["VibeQC"]["samples"]
        times = np.array([row["ms"] for row in observations]) / 1000
        branches = {
            (row["scf_iterations"], row["ccsd_iterations"]) for row in observations
        }
        label = (
            "VibeQC CUDA composition (host preparation included)"
            if index == 0
            else None
        )
        if len(branches) == 1:
            median = float(np.median(times))
            ax.errorbar(
                [index],
                [median],
                yerr=[[median - min(times)], [max(times) - median]],
                fmt="o",
                color=COLORS["VibeQC"],
                label=label,
                capsize=5,
            )
            ax.annotate(
                f"{median:.3f} s",
                (index, max(times)),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        else:
            ax.scatter(
                [index] * len(times),
                times,
                marker="x",
                color=COLORS["VibeQC"],
                label=label,
            )
    ax.set_xticks(
        range(len(points)),
        [
            (
                p["molecule"]
                .upper()
                .replace("H2O", "H₂O")
                .replace("NH3", "NH₃")
                .replace("CH4", "CH₄")
                + f"\n{p['aos']} AOs ({p['nocc']} occupied, {p['nvir']} virtual)"
            )
            for p in points
        ],
    )
    ax.set_ylabel("Complete endpoint / s")
    ax.set_ylim(bottom=0)
    ax.margins(y=0.25)
    ax.set_title(
        "RTX 5090 · CCSD(T) CUDA composition · STO-3G · fresh complete endpoint · 3 repeats",
        loc="left",
        weight="bold",
    )
    ax.grid(axis="y", alpha=0.17)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_svg(fig, destination / "ccsd-t.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    records, provenance = [], {}
    for group in ("hf", "dft-reference", "dft", "cc"):
        for path in sorted((args.raw_directory / group).glob("*.json")):
            record, identity = reduce_point(path, args.raw_directory)
            records.append(record)
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            record["provenance"] = key
            provenance[key] = identity
    paired = {
        (row["method"], row["mode"], row["atoms"])
        for row in records
        if row["family"] == "dft" and row["status"] == "measured"
    }
    for row in records:
        if (
            row["family"] == "dft"
            and row["raw_file"].startswith("dft-reference/")
            and (row["method"], row["mode"], row["atoms"]) in paired
        ):
            row["plot"] = False
            row["plot_exclusion"] = (
                "paired run shown; independent reference run retained"
            )
    stops = args.raw_directory / "stopped-points.json"
    if stops.exists():
        records.extend(json.loads(stops.read_text()))
    if not records:
        parser.error("no actual measurements found")
    # Keep each method family in a readable companion below the repository's
    # evidence size limits. The index binds all samples and shared provenance.
    grouped = []
    for family, method, filename in (
        ("hf", None, "hf.json"),
        ("dft", "pbe-rks", "dft-pbe.json"),
        ("dft", "r2scan-rks", "dft-r2scan.json"),
        ("dft", "pbe0-rks", "dft-pbe0.json"),
        ("cc", None, "ccsd-t.json"),
    ):
        selected = [
            record
            for record in records
            if (
                record["family"] == family
                or (family == "hf" and record["family"] == "hf_energy")
            )
            and (method is None or record["method"] == method)
        ]
        grouped.append((filename, selected))
    if sum(len(selected) for _, selected in grouped) != len(records):
        raise ValueError("unassigned benchmark records; refusing incomplete reduction")
    args.destination.mkdir(parents=True, exist_ok=True)
    parts = []
    for filename, selected in grouped:
        payload = (json.dumps(selected, indent=2, allow_nan=False) + "\n").encode()
        (args.destination / filename).write_bytes(payload)
        parts.append(
            {
                "path": filename,
                "records": len(selected),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    summary = {
        "schema": "vibeqc.readme-benchmarks.v1",
        "timing": "warm endpoint latency; stable-branch median with min/max; x marks unpooled unstable samples",
        "comparison": "normal convergence, no cross-engine speedup claim; work counts retained per repeat",
        "samples": parts,
        "provenance": provenance,
    }
    (args.destination / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    figures(records, args.destination)


if __name__ == "__main__":
    main()

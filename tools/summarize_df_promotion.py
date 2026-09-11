"""Check archived DF value gates and summarize the measured promotion decision.

This host-only postcheck consumes complete reports; it never runs a GPU or
changes production policy. Memory checks apply to the measured homogeneous
batches, whose per-item diagnostics repeat their shared batch plan's peak.
"""

# Source-tree CLI bootstrap for transitive compiler clients.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.vibeqc_validation.f_shell_numerics import numerical_error
from tools.vibeqc_validation.schema import file_hash


def summarize(directory: Path) -> dict:
    """Require raw, endpoint, rank, fixed-geometry, budget, and memory gates."""
    reports = {
        name: json.loads((directory / f"{name}.json").read_text())
        for name in ("isolated", "source", "endpoints", "automatic")
    }
    if not all(report["passed"] for report in reports.values()):
        raise ValueError("all input validation reports must pass")
    endpoints = reports["endpoints"]["runs"]
    selected = [row for row in endpoints if row["route"] == "primitive"]
    result = {
        "schema": "vibeqc.df_value_promotion",
        "version": 1,
        "selected_mapping": "primitive",
        "input_hashes": {
            name: file_hash(directory / f"{name}.json") for name in reports
        },
        "isolated_count": reports["isolated"]["fixture_count"],
        "source_count": len(reports["source"]["runs"]),
        "endpoint_count": len(endpoints),
        "cross_budget": [],
        "fixed_geometry": [],
        "memory": [],
        "geometry": [],
        "speedups": {},
    }
    for method in ("rhf", "uhf"):
        rows = [r for r in selected if r["method"] == method]
        bounded = [r for r in rows if r["budget"] > 0]
        if (
            len({r["budget"] for r in bounded}) < 2
            or len({r["batch"] for r in bounded}) < 2
        ):
            raise ValueError("promotion needs two positive budgets and batch sizes")
        result["speedups"][method] = {
            phase: [
                min(r["speedup"][phase] for r in bounded),
                max(r["speedup"][phase] for r in bounded),
            ]
            for phase in ("cold", "changed_geometry", "warm", "changed_warm")
        }
        if any(
            not np.isfinite(speedup) or speedup < 1.0
            for row in bounded
            for speedup in row["speedup"].values()
        ):
            raise ValueError(
                "selected source mapping regressed a measured endpoint phase"
            )
        for row in rows:
            identity = {k: row[k] for k in ("method", "batch", "budget")}
            bulk = next(
                r for r in rows if r["batch"] == row["batch"] and not r["budget"]
            )
            for side in ("baseline", "generated"):
                phases = row[side]
                change = float(
                    np.min(
                        np.abs(
                            np.asarray(phases["cold"]["energies"])
                            - np.asarray(phases["changed_geometry"]["energies"])
                        )
                    )
                )
                if not np.isfinite(change) or change <= 1e-8:
                    raise ValueError("geometry change must affect every batch item")
                result["geometry"].append(
                    {**identity, "side": side, "minimum_energy_change": change}
                )
                for cold, warm in (
                    ("cold", "warm"),
                    ("changed_geometry", "changed_warm"),
                ):
                    for sample, record in enumerate(phases[warm]):
                        for quantity in ("energies", "forces"):
                            error = numerical_error(
                                np.asarray(record[quantity]),
                                np.asarray(phases[cold][quantity]),
                                atol=2e-8,
                                rtol=2e-9,
                            )
                            result["fixed_geometry"].append(
                                {
                                    **identity,
                                    "side": side,
                                    "phase": warm,
                                    "sample": sample,
                                    "quantity": quantity,
                                    **error,
                                }
                            )
            for phase in ("cold", "changed_geometry", "warm", "changed_warm"):
                records = row["generated"][phase]
                references = bulk["generated"][phase]
                if "warm" not in phase:
                    records, references = [records], [references]
                for sample, (record, reference) in enumerate(
                    zip(records, references, strict=True)
                ):
                    if row["budget"]:
                        peak = max(
                            d["peak_device_bytes"] + d["peak_host_bytes"]
                            for d in record["metrics"]
                        )
                        if peak > row["budget"]:
                            raise ValueError(
                                "reported host/device peak exceeds the budget"
                            )
                        result["memory"].append(
                            {
                                **identity,
                                "phase": phase,
                                "sample": sample,
                                "conservative_host_plus_device_peak": peak,
                            }
                        )
                        for quantity in ("energies", "forces"):
                            error = numerical_error(
                                np.asarray(record[quantity]),
                                np.asarray(reference[quantity]),
                                atol=2e-8,
                                rtol=2e-9,
                            )
                            result["cross_budget"].append(
                                {
                                    **identity,
                                    "phase": phase,
                                    "sample": sample,
                                    "quantity": quantity,
                                    **error,
                                }
                            )
    result["max_conservative_host_plus_device_peak"] = max(
        r["conservative_host_plus_device_peak"] for r in result["memory"]
    )
    for quantity in ("energies", "forces"):
        key = "energy" if quantity == "energies" else "force"
        result[f"endpoint_max_{key}_error"] = max(
            error["max_absolute_error"]
            for row in endpoints
            for name, error in row["errors"].items()
            if name.endswith(quantity)
        )
    result["passed"] = all(
        r["passed"] for gate in ("fixed_geometry", "cross_budget") for r in result[gate]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory)
    (args.directory / "promotion.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if not report["passed"]:
        raise SystemExit("DF promotion postcheck failed")
    print(json.dumps({k: report[k] for k in ("passed", "speedups")}))


if __name__ == "__main__":
    main()

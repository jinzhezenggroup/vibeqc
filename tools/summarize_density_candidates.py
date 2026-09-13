"""Reconstruct diagnostic D/C comparisons without asserting a promoted winner."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from vibeqc_compiler.common.evidence import validate_evidence


def summarize(report):
    """Keep all cases and raw repeats; report ratios only as descriptive statistics.

    The existing publication validator owns numerical acceptance. This summary
    additionally rejects missing/duplicated pairs and mismatched input identities
    before comparing timings. It does not supply #168's missing force gates.
    """
    validate_evidence(report)
    count = report["settings"]["samples_per_route"]
    if count < 5 or report["performance"]["status"] != "not-run":
        raise ValueError(
            "diagnostic D/C summary requires five repeats and no promotion"
        )
    groups = defaultdict(list)
    for row in report["timings"]:
        groups[row["case"]].append(row)
    cases = []
    for case, rows in sorted(groups.items()):
        expected = {(s, i) for s in ("baseline", "candidate") for i in range(count)}
        if (
            len(rows) != len(expected)
            or {(r["selection"], r["repeat"]) for r in rows} != expected
            or len({r["inputs_hash"] for r in rows}) != 1
            or any(r["seconds"] <= 0 for r in rows)
        ):
            raise ValueError("incomplete or mismatched D/C timing pairs")
        entry = {"case": case, "inputs_hash": rows[0]["inputs_hash"]}
        for selection, route in (
            ("baseline", "density_matrix"),
            ("candidate", "orbitals"),
        ):
            seconds = [r["seconds"] for r in rows if r["selection"] == selection]
            entry[route] = {
                "samples": len(seconds),
                "median_seconds": median(seconds),
                "min_seconds": min(seconds),
                "max_seconds": max(seconds),
                "sample_stdev_seconds": stdev(seconds),
            }
        entry["D_over_C_median_ratio"] = (
            entry["density_matrix"]["median_seconds"]
            / entry["orbitals"]["median_seconds"]
        )
        cases.append(entry)
    errors = defaultdict(list)
    for name, record in report["block_errors"].items():
        errors[name.rsplit("/", 1)[-1]].append(record)
    capacities = report.get("endpoint_cases", []) + report.get("batch_cases", [])
    return {
        "revision": report["revision"],
        "dirty": report["dirty"],
        "timing_count": len(report["timings"]),
        "block_gate_count": len(report["block_errors"]),
        "scope": report["settings"]["scope"],
        "decision": "numerical acceptance only; no performance promotion",
        "cases": cases,
        "errors": {
            name: {
                "gates": len(rows),
                "passed": all(r["passed"] for r in rows),
                "max_absolute_error": max(r["max_absolute_error"] for r in rows),
                "max_scaled_error": max(r["max_scaled_error"] for r in rows),
            }
            for name, rows in sorted(errors.items())
        },
        "capacity_bytes": (
            {
                kind: max(r["resource_plan"]["peak_bytes"][kind] for r in capacities)
                for kind in ("host", "device")
            }
            if capacities
            else {"host": report["memory"]["allocated_bytes"], "device": 0}
        ),
        "memory_scope": report["memory"]["reason"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            summarize(json.loads(args.evidence.read_text())), indent=2, sort_keys=True
        )
        + "\n"
    )

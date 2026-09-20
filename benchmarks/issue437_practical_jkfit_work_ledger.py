"""Build the Phase-A practical-JKFIT derivative work ledger for issue #437.

This reducer intentionally consumes intrusive diagnostics that already exist in
``benchmarks.df_shell_work_ledger`` and the matching CUDA trace. It does not
change production execution or invent DRAM traffic from logical source work.

For each cell it records the physical orbital shell-pair domain, auxiliary-shell
population, considered/executed shell and primitive work, nonzero public-weight
density, generated lowering/schedule, measured class GPU time/launches when an
Nsight-backed ledger is supplied, logical response-weight bytes, and the
fraction of the complete ``force_response`` GPU interval.

Distance/exponent bins, response-weight magnitude histograms, and measured
metadata DRAM bytes require additional instrumentation and are reported as
explicitly missing rather than inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import typing
from collections import Counter
from pathlib import Path

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

from benchmarks.df_component_ledger import read_trace

SCHEMA = "vibeqc.issue437_practical_jkfit_work"
VERSION = 1
_MISSING_PHASE_A_FIELDS = (
    "response_weight_magnitude_distribution",
    "distance_exponent_bins",
    "measured_shell_metadata_bytes_read",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _force_response_record(records: typing.Iterable[dict]) -> dict:
    selected = [
        row
        for row in records
        if row.get("operation") == "force_response" and row.get("execution") == "stream"
    ]
    if len(selected) != 1:
        raise ValueError("requires exactly one streamed force_response trace record")
    return selected[0]


def _nonnegative_number(value: typing.Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _positive_integer(value: typing.Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: typing.Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _orbital_pair_counts(
    shells: typing.Iterable[typing.Sequence[int]], pair_mode: int
) -> Counter:
    """Reconstruct the exact angular shell-pair domain used by the shell ledger."""
    if pair_mode not in (0, 1, 2):
        raise ValueError("unknown shell pair mode")
    groups = Counter((int(row[0]), int(row[1])) for row in shells)
    result: Counter = Counter()
    for first, nfirst in sorted(groups.items()):
        for second, nsecond in sorted(groups.items()):
            if pair_mode and first < second:
                continue
            pairs = (
                nfirst * (nfirst + 1) // 2
                if pair_mode and first == second
                else nfirst * nsecond
            )
            result[(first[0], second[0])] += pairs
    return result


def _auxiliary_shell_counts(
    shells: typing.Iterable[typing.Sequence[int]],
) -> Counter:
    return Counter(int(row[0]) for row in shells)


def _considered_primitive_products(work_ledger: dict) -> Counter:
    """Count the complete host primitive domain before zero-weight pruning."""
    result: Counter = Counter()
    rows = work_ledger["host_reconstruction"]["signature_tasks"]
    for row in rows:
        signature = tuple(int(value) for value in row["signature"])
        tasks = _nonnegative_integer(row["tasks"], "signature tasks")
        if len(signature) != 6 or tasks < 0:
            raise ValueError("invalid host signature-task reconstruction")
        primitives = math.prod(signature[3:])
        result[signature[:3]] += tasks * primitives
    return result


def _class_map(work_ledger: dict) -> dict[tuple[int, int, int], dict]:
    result: dict[tuple[int, int, int], dict] = {}
    for row in work_ledger["classes"]:
        angular = tuple(int(value) for value in row["angular"])
        if len(angular) != 3 or angular in result:
            raise ValueError("invalid or duplicate angular class in work ledger")
        result[angular] = row
    if not result:
        raise ValueError("work ledger has no derivative classes")
    return result


def summarize_cell(
    label: str,
    role: str,
    work_ledger: dict,
    trace_record: dict,
) -> dict:
    """Normalize one intrusive force call into the issue-437 Phase-A schema."""
    if not label or role not in {"equal", "practical"}:
        raise ValueError("cell requires a nonempty label and role equal|practical")

    nao = _positive_integer(trace_record.get("nbf"), "trace nbf")
    naux = _positive_integer(trace_record.get("naux"), "trace naux")
    if _positive_integer(work_ledger.get("aos"), "work-ledger aos") != nao:
        raise ValueError("trace/work-ledger AO dimensions differ")

    counters = trace_record.get("counters")
    if not isinstance(counters, dict):
        raise TypeError("trace is missing counters")
    pair_mode = counters.get("shell_work_pair_mode")
    if type(pair_mode) is not int:
        raise TypeError("trace is missing shell_work_pair_mode")

    reconstruction = work_ledger.get("host_reconstruction")
    if not isinstance(reconstruction, dict):
        raise TypeError("work ledger is missing host reconstruction")
    orbital_shells = reconstruction.get("shells")
    auxiliary_shells = reconstruction.get("auxiliary_shells")
    if not isinstance(orbital_shells, list) or not isinstance(auxiliary_shells, list):
        raise TypeError("work ledger is missing shell metadata")

    root_ms = _nonnegative_number(
        trace_record["regions"][0]["gpu_ms"], "force_response gpu_ms"
    )
    pair_counts = _orbital_pair_counts(orbital_shells, pair_mode)
    auxiliary_counts = _auxiliary_shell_counts(auxiliary_shells)
    considered_primitives = _considered_primitive_products(work_ledger)
    classes = _class_map(work_ledger)

    rows = []
    totals: Counter = Counter()
    class_gpu_ms = 0.0
    class_launches = 0
    complete_nsys = True
    for angular, source in sorted(classes.items()):
        work = source.get("work")
        if not isinstance(work, dict):
            raise TypeError("class is missing detailed work")
        shell_considered = _nonnegative_integer(work["shell_tasks"], "shell_tasks")
        shell_executed = _nonnegative_integer(
            work["active_shell_tasks"], "active_shell_tasks"
        )
        primitive_executed = _nonnegative_integer(
            work["primitive_products"], "primitive_products"
        )
        primitive_considered = int(considered_primitives[angular])
        public_weight_loads = _nonnegative_integer(
            work["public_weight_loads"], "public_weight_loads"
        )
        public_nonzero = _nonnegative_integer(
            work["public_nonzero_weights"], "public_nonzero_weights"
        )
        if not (
            0 <= shell_executed <= shell_considered
            and 0 <= primitive_executed <= primitive_considered
            and 0 <= public_nonzero <= public_weight_loads
        ):
            raise ValueError("executed/nonzero work exceeds its considered domain")

        nsys = source.get("nsys")
        if nsys is None:
            complete_nsys = False
            gpu_ms = _nonnegative_number(
                source["component_gpu_inclusive_ms"],
                "class component gpu_ms",
            )
            launches = None
            schedule = None
            timing_source = "cuda_event_inclusive"
        else:
            gpu_ms = _nonnegative_number(nsys["gpu_ms"], "Nsight class gpu_ms")
            launches = _positive_integer(nsys["launches"], "Nsight class launches")
            schedule = nsys.get("schedule")
            timing_source = "nsight_kernel_activity"
            class_launches += launches
        class_gpu_ms += gpu_ms

        row = {
            "angular": list(angular),
            "orbital_shell_pairs": int(pair_counts[(angular[0], angular[1])]),
            "auxiliary_shells": int(auxiliary_counts[angular[2]]),
            "shell_triples_considered": shell_considered,
            "shell_triples_executed": shell_executed,
            "primitive_products_considered": primitive_considered,
            "primitive_products_executed": primitive_executed,
            "public_weight_loads": public_weight_loads,
            "public_nonzero_weights": public_nonzero,
            "public_nonzero_fraction": _ratio(public_nonzero, public_weight_loads),
            "logical_response_weight_bytes": public_weight_loads * 8,
            "lowering": source["lowering"],
            "schedule": schedule,
            "gpu_ms": gpu_ms,
            "gpu_timing_source": timing_source,
            "launches": launches,
            "force_response_gpu_fraction": _ratio(gpu_ms, root_ms),
        }
        rows.append(row)
        for key in (
            "shell_triples_considered",
            "shell_triples_executed",
            "primitive_products_considered",
            "primitive_products_executed",
            "public_weight_loads",
            "public_nonzero_weights",
            "logical_response_weight_bytes",
        ):
            totals[key] += row[key]

    return {
        "label": label,
        "role": role,
        "nao": nao,
        "naux": naux,
        "pair_mode": pair_mode,
        "orbital_shell_domain": sorted([list(row) for row in orbital_shells]),
        "force_response_gpu_ms": root_ms,
        "classes_gpu_ms": class_gpu_ms,
        "classes_force_response_gpu_fraction": _ratio(class_gpu_ms, root_ms),
        "nsight_complete": complete_nsys,
        "nsight_launches": class_launches if complete_nsys else None,
        "totals": dict(totals),
        "classes": rows,
        "missing_phase_a_fields": list(_MISSING_PHASE_A_FIELDS),
        "interpretation": (
            "Considered primitive work is reconstructed from the complete host "
            "signature domain; executed work and nonzero weights come from device "
            "diagnostic counters. logical_response_weight_bytes is source-level "
            "8-byte weight traffic, not a measured DRAM transaction count. CUDA "
            "event class intervals may include stream idle time; Nsight kernel "
            "activity is used when present."
        ),
    }


def compare_cells(equal: dict, practical: dict) -> dict:
    """Separate Naux scaling from a change in the orbital shell-pair domain."""
    if equal["role"] != "equal" or practical["role"] != "practical":
        raise ValueError("comparison requires equal then practical roles")
    if equal["nao"] != practical["nao"]:
        raise ValueError("comparison requires the same orbital AO count")
    # Compare the complete host domain, not only the intersection of measured
    # angular classes. Otherwise a changed/missing orbital class or primitive
    # population can be mislabeled as auxiliary-basis scaling.
    if (
        equal["pair_mode"] != practical["pair_mode"]
        or equal["orbital_shell_domain"] != practical["orbital_shell_domain"]
    ):
        raise ValueError("orbital shell-pair domain changed between comparison cells")

    fields = (
        "shell_triples_considered",
        "shell_triples_executed",
        "primitive_products_considered",
        "primitive_products_executed",
        "public_weight_loads",
        "public_nonzero_weights",
        "logical_response_weight_bytes",
    )
    totals = {
        field: {
            "equal": equal["totals"][field],
            "practical": practical["totals"][field],
            "ratio": _ratio(practical["totals"][field], equal["totals"][field]),
        }
        for field in fields
    }
    totals["naux"] = {
        "equal": equal["naux"],
        "practical": practical["naux"],
        "ratio": _ratio(practical["naux"], equal["naux"]),
    }
    totals["force_response_gpu_ms"] = {
        "equal": equal["force_response_gpu_ms"],
        "practical": practical["force_response_gpu_ms"],
        "ratio": _ratio(
            practical["force_response_gpu_ms"], equal["force_response_gpu_ms"]
        ),
    }

    equal_classes = {tuple(row["angular"]): row for row in equal["classes"]}
    practical_classes = {tuple(row["angular"]): row for row in practical["classes"]}
    class_rows = []
    for angular in sorted(set(equal_classes) | set(practical_classes)):
        left, right = equal_classes.get(angular), practical_classes.get(angular)
        row = {
            "angular": list(angular),
            "present_in_equal": left is not None,
            "present_in_practical": right is not None,
        }
        if left is not None and right is not None:
            row["ratios"] = {
                field: _ratio(right[field], left[field])
                for field in (
                    "shell_triples_considered",
                    "primitive_products_considered",
                    "primitive_products_executed",
                    "public_weight_loads",
                    "gpu_ms",
                )
            }
        class_rows.append(row)

    return {
        "equal": equal["label"],
        "practical": practical["label"],
        "nao": equal["nao"],
        "totals": totals,
        "classes": class_rows,
        "interpretation": (
            "Ratios compare the same orbital AO count and verified orbital "
            "shell-pair domain. New auxiliary angular classes are reported "
            "without manufacturing a finite equal-basis ratio."
        ),
    }


def _parse_expectations(rows: list[list[str]]) -> dict[str, tuple[int, int]]:
    result = {}
    for label, nao, naux in rows:
        if label in result:
            raise ValueError(f"duplicate expectation for {label}")
        result[label] = (int(nao), int(naux))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        action="append",
        nargs=4,
        metavar=("LABEL", "ROLE", "WORK_LEDGER", "TRACE"),
        default=[],
        help=(
            "Add one equal/practical cell from a df_shell_work_ledger JSON "
            "and matching trace"
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("LABEL", "EQUAL_CELL", "PRACTICAL_CELL"),
        default=[],
        help="Add an equal-vs-practical comparison",
    )
    parser.add_argument(
        "--expect",
        action="append",
        nargs=3,
        metavar=("CELL", "NAO", "NAUX"),
        default=[],
        help="Require an exact AO/auxiliary dimension for a named cell",
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if not args.cell:
        parser.error("at least one --cell is required")

    expectations = _parse_expectations(args.expect)
    cells = {}
    provenance = {}
    for label, role, work_name, trace_name in args.cell:
        if label in cells:
            parser.error(f"duplicate cell label: {label}")
        work_path, trace_path = Path(work_name), Path(trace_name)
        work = json.loads(work_path.read_text())
        trace_hash = _sha256(trace_path)
        if work.get("trace_sha256") != trace_hash:
            raise ValueError(f"{label}: work ledger does not match the supplied trace")
        record = _force_response_record(read_trace(trace_path))
        cell = summarize_cell(label, role, work, record)
        if label in expectations and (cell["nao"], cell["naux"]) != expectations[label]:
            raise ValueError(
                f"{label}: observed {(cell['nao'], cell['naux'])} "
                f"!= expected {expectations[label]}"
            )
        cells[label] = cell
        provenance[label] = {
            "work_ledger": str(work_path),
            "work_ledger_sha256": _sha256(work_path),
            "trace": str(trace_path),
            "trace_sha256": trace_hash,
            "native_source_identity": work.get("native_source_identity"),
            "library_sha256": work.get("library_sha256"),
            "generated_schedule_sha256": work.get("generated_schedule_sha256"),
            "nsys_sqlite_sha256": work.get("nsys_sqlite_sha256"),
        }

    comparisons = {}
    for name, equal_label, practical_label in args.pair:
        if name in comparisons:
            parser.error(f"duplicate comparison label: {name}")
        try:
            comparisons[name] = compare_cells(
                cells[equal_label], cells[practical_label]
            )
        except KeyError as error:
            raise ValueError(f"unknown comparison cell: {error.args[0]}") from error

    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "scope": (
            "Issue #437 Phase-A intrusive derivative work ledger. This report "
            "does not enable screening or make clean endpoint performance claims."
        ),
        "cells": cells,
        "comparisons": comparisons,
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

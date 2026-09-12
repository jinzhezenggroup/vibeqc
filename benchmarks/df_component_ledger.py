"""Validate and aggregate opt-in CUDA component traces for issues #282/#283.

CUDA event intervals include stream idle time between submissions. Host and
GPU views overlap and must never be added together. Capture records describe
graph construction only; replay work requires a separate Nsight capture.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _milliseconds(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite nonnegative milliseconds")
    return float(value)


def _exclusive(regions: list[dict], field: str) -> list[float]:
    """Subtract immediate children, allowing only event/serialization roundoff.

    Retain tiny signed residuals so aggregation conserves the measured root
    time exactly. They indicate timer resolution, not negative execution cost.
    """
    children = defaultdict(list)
    for index, region in enumerate(regions[1:], 1):
        children[region["parent"]].append(index)
    result = []
    for index, region in enumerate(regions):
        inclusive = _milliseconds(region[field], field)
        exclusive = inclusive - sum(regions[c][field] for c in children[index])
        tolerance = max(0.01, inclusive * 2e-6)
        if exclusive < -tolerance:
            raise ValueError(f"{field}: children exceed their parent interval")
        result.append(exclusive)
    return result


def validate_record(record: dict) -> dict:
    """Reject partial, mistimed or structurally inconsistent native records."""
    try:
        if record["schema"] != "vibeqc.df_trace" or record["version"] != 1:
            raise ValueError("unsupported component trace schema/version")
        if record["valid"] is not True or record["cuda_error"] != 0:
            raise ValueError("invalid native component trace")
        for name in ("dropped_regions", "dropped_tiles"):
            if _integer(record[name], name) != 0:
                raise ValueError("truncated native component trace")
        for name in (
            "id",
            "systems",
            "system_offset",
            "nbf",
            "naux",
            "profiler_event_count",
        ):
            _integer(record[name], name)
        if not record["systems"] or not record["nbf"]:
            raise ValueError("empty component shape")
        for name in ("source_backed", "streamed", "nvtx"):
            if type(record[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
        execution = record["execution"]
        if execution not in ("stream", "graph_capture"):
            raise ValueError("unknown component execution mode")
        regions = record["regions"]
        if not isinstance(regions, list) or not regions:
            raise ValueError("missing component root")
        if regions[0]["name"] != record["operation"]:
            raise ValueError("operation/root mismatch")
        ancestors = []
        for index, region in enumerate(regions):
            if not isinstance(region["name"], str) or not region["name"]:
                raise ValueError("missing component name")
            parent = _integer(region["parent"], "parent", -1)
            if index == 0:
                if parent != -1:
                    raise ValueError("invalid root parent")
            elif parent not in ancestors:
                raise ValueError("invalid component parent hierarchy")
            while ancestors and ancestors[-1] != parent:
                ancestors.pop()
            ancestors.append(index)
            _milliseconds(region["host_ms"], "host_ms")
            if execution == "graph_capture":
                if region["gpu_ms"] is not None:
                    raise ValueError("graph capture must not report GPU execution time")
            else:
                _milliseconds(region["gpu_ms"], "gpu_ms")
        expected_events = 0 if execution == "graph_capture" else 2 * len(regions)
        if record["profiler_event_count"] != expected_events:
            raise ValueError("incomplete profiler event pairs")
        sync = _milliseconds(
            record["final_synchronization_ms"], "final_synchronization_ms"
        )
        completed = _milliseconds(record["host_completion_ms"], "host_completion_ms")
        if not math.isclose(
            completed, regions[0]["host_ms"] + sync, rel_tol=2e-7, abs_tol=1e-5
        ):
            raise ValueError("inconsistent host completion time")
        host_exclusive = _exclusive(regions, "host_ms")
        gpu_exclusive = (
            None if execution == "graph_capture" else _exclusive(regions, "gpu_ms")
        )
        counters = record["counters"]
        if not isinstance(counters, dict):
            raise TypeError("missing component counters")
        for name, value in counters.items():
            _integer(value, name)
        keys, expected = set(), defaultdict(int)
        for tile in record["tiles"]:
            for name in ("system", "pair_begin", "auxiliary_begin"):
                _integer(tile[name], name)
            for name in ("pair_count", "auxiliary_count", "productions"):
                _integer(tile[name], name, 1)
            _integer(tile["derivative_coordinate"], "derivative_coordinate", -1)
            if type(tile["transformed"]) is not bool:
                raise ValueError("tile transformed must be boolean")
            if (
                not record["system_offset"]
                <= tile["system"]
                < record["system_offset"] + record["systems"]
                or tile["pair_begin"] + tile["pair_count"] > record["nbf"] ** 2
                or tile["auxiliary_begin"] + tile["auxiliary_count"] > record["naux"]
            ):
                raise ValueError("logical tile is outside the operation shape")
            key = tuple(
                tile[k]
                for k in (
                    "system",
                    "pair_begin",
                    "pair_count",
                    "auxiliary_begin",
                    "auxiliary_count",
                    "derivative_coordinate",
                    "transformed",
                )
            )
            if key in keys:
                raise ValueError("duplicate logical tile key")
            keys.add(key)
            kind = (
                "derivative"
                if tile["derivative_coordinate"] >= 0
                else "transformed"
                if tile["transformed"]
                else "raw"
            )
            expected[f"{kind}_tile_productions"] += tile["productions"]
            expected[f"{kind}_value_bytes"] += (
                8 * tile["pair_count"] * tile["auxiliary_count"] * tile["productions"]
            )
        for kind in ("raw", "transformed", "derivative"):
            for suffix in ("tile_productions", "value_bytes"):
                key = f"{kind}_{suffix}"
                if counters.get(key, 0) != expected[key]:
                    raise ValueError("logical tile/counter totals disagree")
        return {"host_exclusive_ms": host_exclusive, "gpu_exclusive_ms": gpu_exclusive}
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("incomplete component trace record") from error


def read_trace(path: Path) -> list[dict]:
    """Require complete JSONL and unique operation IDs within one process run."""
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("missing or incomplete component trace")
    records, identifiers = [], set()
    for line in raw.splitlines():
        record = json.loads(line)
        validate_record(record)
        if record["id"] in identifiers:
            raise ValueError(
                "duplicate operation ID; use a fresh trace file per process"
            )
        identifiers.add(record["id"])
        records.append(record)
    return records


def aggregate(records: list[dict]) -> dict:
    """Group exclusive components without mixing capture counts with execution.

    Counter sums describe total work. Scratch counters additionally publish a
    per-operation maximum; their sum is never a claim about peak live memory.
    Full resource-ledger accounting remains a separate endpoint measurement.
    """
    if not records:
        raise ValueError("missing component trace records")
    groups = {}
    for record in records:
        exclusive = validate_record(record)
        key = (record["execution"], record["operation"])
        if key not in groups:
            groups[key] = {
                "execution": key[0],
                "operation": key[1],
                "calls": 0,
                "host_completion_ms": 0.0,
                "final_synchronization_ms": 0.0,
                "gpu_inclusive_ms": None if key[0] == "graph_capture" else 0.0,
                "host_exclusive_ms": defaultdict(float),
                "gpu_exclusive_ms": defaultdict(float),
                "counter_sums": defaultdict(int),
                "counter_maxima": defaultdict(int),
                "logical_tiles": 0,
                "repeated_tile_productions": 0,
            }
        group = groups[key]
        group["calls"] += 1
        group["host_completion_ms"] += record["host_completion_ms"]
        group["final_synchronization_ms"] += record["final_synchronization_ms"]
        if key[0] == "stream":
            group["gpu_inclusive_ms"] += record["regions"][0]["gpu_ms"]
        for index, region in enumerate(record["regions"]):
            for field, values in exclusive.items():
                if values is not None:
                    group[field][region["name"]] += values[index]
        for name, value in record["counters"].items():
            group["counter_sums"][name] += value
            group["counter_maxima"][name] = max(group["counter_maxima"][name], value)
        group["logical_tiles"] += len(record["tiles"])
        group["repeated_tile_productions"] += sum(
            t["productions"] - 1 for t in record["tiles"]
        )
    return {
        "groups": list(groups.values()),
        "interpretation": (
            "Host and GPU times overlap; do not add them. GPU intervals include stream idle time. "
            "Exclusive root residuals are unclassified runtime work. Capture counters describe "
            "construction only and exclude graph replay. Counter maxima are per operation, "
            "not the complete resource peak. Use unprofiled endpoints for performance claims."
        ),
    }


def trace_identity(path: Path) -> dict:
    """Retain an immutable reference to the raw trace alongside its summary."""
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def force_attribution(energy: dict, force: dict) -> dict:
    """Compare named host intervals to the same profiled pair's force delta.

    The root residual and time outside these operations stay unclassified.
    In particular, nuclear assembly cannot be relabeled as measured CUDA work.
    The ratio is diagnostic and may exceed one when the two SCF solves vary.
    """
    roots = [
        g
        for g in force["components"]["groups"]
        if g["execution"] == "stream"
        and g["operation"] in ("force_response", "one_electron_response")
    ]
    if {g["operation"] for g in roots} != {"force_response", "one_electron_response"}:
        raise ValueError("missing force operations for attribution")
    increment = 1000 * (force["seconds"] - energy["seconds"])
    completed = sum(g["host_completion_ms"] for g in roots)
    named = sum(
        sum(
            value
            for name, value in g["host_exclusive_ms"].items()
            if name != g["operation"]
        )
        + g["final_synchronization_ms"]
        for g in roots
    )
    return {
        "profiled_force_increment_ms": increment,
        "force_operations_host_completion_ms": completed,
        "named_host_and_synchronization_ms": named,
        "unclassified_inside_operations_ms": completed - named,
        "outside_operations_delta_ms": increment - completed,
        "named_fraction_of_increment": named / increment if increment > 0 else None,
    }

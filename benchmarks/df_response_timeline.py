"""Summarize executed Nsight activity without relabeling host waits as copies.

CUDA API, device execution and event intervals overlap. Report them separately.
Correlate raw-value copies with their actual DMA records and intersect the API
intervals with device work on that stream. A residual is explicitly unassigned
runtime/staging/descheduling time; only the packed ablation times a CPU gather.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import typing
from collections import defaultdict
from pathlib import Path

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path


def interval_union(intervals: typing.Any) -> typing.Any:
    """Merge overlapping clock intervals, preserving gaps and exact ns units."""
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("reversed timeline interval")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def intersect_duration(windows: typing.Any, activity: typing.Any) -> typing.Any:
    """Measure the union intersection, counting concurrent work only once."""
    left, right = interval_union(windows), interval_union(activity)
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] < b[1]:
            i += 1
        else:
            j += 1
    return total


def _nvtx_regions(connection: typing.Any, strings: typing.Any) -> typing.Any:
    """Index synchronous thread ranges; ignore marks without a complete end."""
    tables = {r[0] for r in connection.execute("select name from sqlite_master")}
    result = defaultdict(list)
    if "NVTX_EVENTS" in tables:
        for row in connection.execute("select * from NVTX_EVENTS"):
            if row["end"] is None or row["end"] < row["start"]:
                continue
            name = row["text"] or strings.get(row["textId"])
            if name:
                result[row["globalTid"]].append((row["start"], row["end"], name))
    return {tid: sorted(rows) for tid, rows in result.items()}


def execution_coverage(
    window: typing.Any, kernels: typing.Any, copies: typing.Any, memsets: typing.Any
) -> typing.Any:
    """Measure recorded activity unions within a host interval, without sums.

    The uncovered interval includes submission gaps and host preparation. It
    is not a utilization sample, and cannot identify why the GPU had no work.
    Memsets count as activity too; omitting them would invent tiny idle gaps.
    """
    windows = [window]
    kernel_intervals = [(r["start"], r["end"]) for r in kernels]
    copy_intervals = [(r["start"], r["end"]) for r in copies]
    memset_intervals = [(r["start"], r["end"]) for r in memsets]
    active = intersect_duration(
        windows, kernel_intervals + copy_intervals + memset_intervals
    )
    duration = window[1] - window[0]
    return {
        "host_window_ms": duration / 1e6,
        "kernel_union_ms": intersect_duration(windows, kernel_intervals) / 1e6,
        "copy_union_ms": intersect_duration(windows, copy_intervals) / 1e6,
        "memset_union_ms": intersect_duration(windows, memset_intervals) / 1e6,
        "active_union_ms": active / 1e6,
        "no_recorded_gpu_work_ms": (duration - active) / 1e6,
    }


def summarize(database: typing.Any) -> typing.Any:
    """Require actual kernel/API/copy activity, retaining exact correlation counts."""
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        strings = dict(connection.execute("select id,value from StringIds"))
        regions = _nvtx_regions(connection, strings)
        apis = {
            row["correlationId"]: dict(row)
            for row in connection.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME")
        }
        kernels = [
            dict(r)
            for r in connection.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL")
        ]
        copies = [
            dict(r)
            for r in connection.execute("select * from CUPTI_ACTIVITY_KIND_MEMCPY")
        ]
        tables = {r[0] for r in connection.execute("select name from sqlite_master")}
        memsets = (
            [
                dict(r)
                for r in connection.execute("select * from CUPTI_ACTIVITY_KIND_MEMSET")
            ]
            if "CUPTI_ACTIVITY_KIND_MEMSET" in tables
            else []
        )
    if not apis or not kernels or not copies:
        raise ValueError("missing executed CUDA activity")
    tids = {r["globalTid"] for r in apis.values()}
    if len(tids) != 1:
        raise ValueError("this response probe requires one CUDA submission thread")
    # Sweep nested ranges once. A captured force has tens of thousands of
    # regions/APIs; rescanning every prior range for each call is quadratic.
    rows = regions.get(next(iter(tids)), [])
    active, next_region, leaves = [], 0, {}
    for api in sorted(apis.values(), key=lambda r: r["start"]):
        while next_region < len(rows) and rows[next_region][0] <= api["start"]:
            current = rows[next_region]
            active = [r for r in active if r[1] >= current[0]]
            active.append(current)
            next_region += 1
        active = [r for r in active if r[1] >= api["end"]]
        leaf = min(active, key=lambda r: r[1] - r[0])[2] if active else None
        leaves[api["correlationId"]] = leaf

    def region(api: typing.Any) -> typing.Any:
        return leaves[api["correlationId"]]

    raw = []
    for copy in copies:
        api = apis.get(copy["correlationId"])
        if api is None:
            raise ValueError("uncorrelated device copy")
        name = strings[api["nameId"]]
        leaf = region(api)
        if leaf in ("raw_value_slice_upload", "raw_value_resident_upload") or (
            not regions and name.startswith("cudaMemcpy2DAsync")
        ):
            raw.append((copy, api))
    # A source-backed response regenerates raw columns instead of uploading
    # them. Require positively identified executed generation; an incomplete
    # trace must not become a fictitious zero-transfer result.
    regenerated = [
        kernel
        for kernel in kernels
        if kernel["correlationId"] in apis
        and region(apis[kernel["correlationId"]]) == "raw_three_center_generation"
    ]
    if not raw and not regenerated:
        raise ValueError(
            "missing raw-value copies or generation (requires component NVTX)"
        )
    if raw and regenerated:
        raise ValueError("mixed raw-value providers in one response probe")
    streams = {c["streamId"] for c, _ in raw} or {k["streamId"] for k in regenerated}
    if len(streams) != 1:
        raise ValueError("expected one response stream")
    stream = streams.pop()
    windows = [(r["start"], r["end"]) for _, r in raw]
    raw_host_ns = sum(end - start for start, end in interval_union(windows))
    stream_kernels = [k for k in kernels if k["streamId"] == stream]
    stream_copies = [c for c in copies if c["streamId"] == stream]
    work = [(r["start"], r["end"]) for r in stream_kernels + stream_copies]
    kernel_groups = defaultdict(list)
    for kernel in kernels:
        api = apis.get(kernel["correlationId"])
        leaf = region(api) if api else None
        kernel_groups[(leaf, strings[kernel["demangledName"]])].append(kernel)
    kernel_rows = []
    for (leaf, name), rows in kernel_groups.items():
        kernel_rows.append(
            {
                "component": leaf,
                "kernel": name,
                "calls": len(rows),
                "device_ms": sum(r["end"] - r["start"] for r in rows) / 1e6,
                "overlap_raw_copy_host_ms": intersect_duration(
                    windows,
                    [(r["start"], r["end"]) for r in rows if r["streamId"] == stream],
                )
                / 1e6,
            }
        )
    api_groups = defaultdict(list)
    for api in apis.values():
        api_groups[strings[api["nameId"]]].append(api)
    component_groups = defaultdict(list)
    for rows in regions.values():
        for start, end, name in rows:
            component_groups[name].append(end - start)
    raw_scopes = [
        (start, end)
        for rows in regions.values()
        for start, end, name in rows
        if name in ("raw_value_slice_upload", "raw_value_resident_upload")
    ]
    scope_summary = None
    if raw_scopes:
        if len(raw_scopes) != len({api["correlationId"] for _, api in raw}):
            raise ValueError("raw NVTX scopes do not match executed copy count")
        scope_ns = sum(end - start for start, end in interval_union(raw_scopes))
        scope_summary = {
            "host_nvtx_ms": scope_ns / 1e6,
            "host_overlap_same_stream_kernels_ms": intersect_duration(
                raw_scopes, [(r["start"], r["end"]) for r in stream_kernels]
            )
            / 1e6,
            "host_overlap_same_stream_copies_ms": intersect_duration(
                raw_scopes, [(r["start"], r["end"]) for r in stream_copies]
            )
            / 1e6,
            "host_residual_not_device_activity_ms": (
                scope_ns - intersect_duration(raw_scopes, work)
            )
            / 1e6,
            "scope": "includes progress fences when enabled; separate from the copy API interval",
        }
    force_windows = [
        (start, end)
        for ranges in regions.values()
        for start, end, name in ranges
        if name == "force_response"
    ]
    response_coverage, panel_rows = None, []
    if len(force_windows) == 1:
        force_start, force_end = force_windows[0]
        response_coverage = execution_coverage(
            force_windows[0], kernels, copies, memsets
        )
        charge_ends = [
            end
            for ranges in regions.values()
            for start, end, name in ranges
            if name == "coulomb_response" and force_start <= start <= end <= force_end
        ]
        derivative_ranges = sorted(
            (start, end)
            for ranges in regions.values()
            for start, end, name in ranges
            if name == "three_center_derivative_contraction"
            and force_start <= start <= end <= force_end
        )
        if len(charge_ends) == 1:
            begin = charge_ends[0]
            # Each panel begins after the preceding consumer submission. Use
            # API correlation to assign asynchronous work: clipping a kernel
            # to its host NVTX range loses execution queued after range exit.
            for index, (_, end) in enumerate(derivative_ranges):
                ids = {key for key, api in apis.items() if begin <= api["start"] < end}
                panel_kernels = [k for k in kernels if k["correlationId"] in ids]
                panel_copies = [c for c in copies if c["correlationId"] in ids]
                by_component = defaultdict(
                    lambda: {"kernel_calls": 0, "device_ms": 0.0}
                )
                for kernel in panel_kernels:
                    label = leaves[kernel["correlationId"]] or "unassigned"
                    by_component[label]["kernel_calls"] += 1
                    by_component[label]["device_ms"] += (
                        kernel["end"] - kernel["start"]
                    ) / 1e6
                raw_ids = {c["correlationId"] for c, _ in raw}
                panel_raw = [c for c in panel_copies if c["correlationId"] in raw_ids]
                host_components = defaultdict(lambda: {"calls": 0, "host_ms": 0.0})
                for ranges in regions.values():
                    for start, stop, name in ranges:
                        if begin <= start <= stop <= end:
                            host_components[name]["calls"] += 1
                            host_components[name]["host_ms"] += (stop - start) / 1e6
                panel_rows.append(
                    {
                        "index": index,
                        "host_submission_ms": (end - begin) / 1e6,
                        "raw_copy_calls": len(panel_raw),
                        "raw_copy_bytes": sum(c["bytes"] for c in panel_raw),
                        "raw_dma_ms": sum(c["end"] - c["start"] for c in panel_raw)
                        / 1e6,
                        "components": dict(by_component),
                        "host_components_inclusive": dict(host_components),
                    }
                )
                begin = end
    return {
        "scope": "actual Nsight device activity; host/device columns overlap and must not be added",
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "raw_value_provider": "source" if regenerated else "host",
        "raw_generation": {
            "calls": len(regenerated),
            "device_ms": sum(k["end"] - k["start"] for k in regenerated) / 1e6,
        },
        "raw_copies": {
            "calls": len(raw),
            "bytes": sum(c["bytes"] for c, _ in raw),
            "host_api_ms": raw_host_ns / 1e6,
            "device_dma_ms": sum(c["end"] - c["start"] for c, _ in raw) / 1e6,
            "host_overlap_same_stream_kernels_ms": intersect_duration(
                windows, [(r["start"], r["end"]) for r in stream_kernels]
            )
            / 1e6,
            "host_overlap_same_stream_copies_ms": intersect_duration(
                windows, [(r["start"], r["end"]) for r in stream_copies]
            )
            / 1e6,
            "host_residual_not_device_activity_ms": (
                raw_host_ns - intersect_duration(windows, work)
            )
            / 1e6,
            "residual_meaning": "runtime staging/control/descheduling; not a measured CPU packing duration",
        },
        "raw_copy_scope": scope_summary,
        "response_execution_coverage": response_coverage,
        "response_panels": panel_rows,
        "kernels": sorted(kernel_rows, key=lambda r: r["device_ms"], reverse=True),
        "api": sorted(
            [
                {
                    "name": name,
                    "calls": len(rows),
                    "host_ms": sum(r["end"] - r["start"] for r in rows) / 1e6,
                }
                for name, rows in api_groups.items()
            ],
            key=lambda r: r["host_ms"],
            reverse=True,
        ),
        "nvtx_host_inclusive": {
            name: {"calls": len(values), "host_ms": sum(values) / 1e6}
            for name, values in component_groups.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    payload = summarize(args.database)
    with args.output.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

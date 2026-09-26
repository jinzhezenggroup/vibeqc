"""Reduce Nsight CUDA activities; never interpret host waits as eigen CPU math.

Usage: python reduce_profiles.py ARTIFACT_PROFILES RETAINED_PROFILES
The CUDA API intervals and clipped GPU overlap are retained separately.
Neither occupancy estimates nor activity sums are clean endpoint timings.
"""

import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

source, destination = map(Path, sys.argv[1:])

# Check the complete input set before publishing any summaries. A failed
# collection must identify every missing dependency without leaving a partial
# output bundle that could be mistaken for a qualified profile.
required = [source / "environment.json"]
for aos in (384, 768):
    if (source / f"{aos}-diis.json").is_file():
        required.extend(
            source / f"{aos}-diis.0-{policy}.host.jsonl"
            for policy in ("serial", "auto")
        )
        required.append(source / f"{aos}-memory.csv")
missing = [path for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(
        "missing required profile artifacts: " + ", ".join(map(str, missing))
    )
destination.mkdir(parents=True, exist_ok=True)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(name, data):
    (destination / name).write_text(json.dumps(data, indent=2) + "\n")


for path in sorted(source.glob("*.sqlite")):
    database = sqlite3.connect(path)
    database.row_factory = sqlite3.Row
    kernels = database.execute(
        "SELECT k.start, k.end, s.value name, registersPerThread, "
        "staticSharedMemory, dynamicSharedMemory, blockX, blockY, blockZ "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
        "ORDER BY k.start"
    ).fetchall()
    grouped = {}
    for kernel in kernels:
        name = kernel["name"]
        row = grouped.setdefault(
            name,
            {
                "name": name,
                "calls": 0,
                "gpu_ms": 0,
                "registers_per_thread": [kernel["registersPerThread"]] * 2,
                "static_shared_peak_bytes": 0,
                "dynamic_shared_peak_bytes": 0,
                "block_threads": [],
            },
        )
        row["calls"] += 1
        row["gpu_ms"] += (kernel["end"] - kernel["start"]) / 1e6
        row["registers_per_thread"][0] = min(
            row["registers_per_thread"][0], kernel["registersPerThread"]
        )
        row["registers_per_thread"][1] = max(
            row["registers_per_thread"][1], kernel["registersPerThread"]
        )
        row["static_shared_peak_bytes"] = max(
            row["static_shared_peak_bytes"], kernel["staticSharedMemory"]
        )
        row["dynamic_shared_peak_bytes"] = max(
            row["dynamic_shared_peak_bytes"], kernel["dynamicSharedMemory"]
        )
        threads = kernel["blockX"] * kernel["blockY"] * kernel["blockZ"]
        if threads not in row["block_threads"]:
            row["block_threads"].append(threads)
    runtime = database.execute(
        "SELECT r.start, r.end, r.correlationId, s.value name FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId ORDER BY r.start"
    ).fetchall()
    apis = {}
    waits = []
    for event in runtime:
        duration = (event["end"] - event["start"]) / 1e6
        row = apis.setdefault(
            event["name"], {"calls": 0, "host_ms": 0, "maximum_ms": 0}
        )
        row["calls"] += 1
        row["host_ms"] += duration
        row["maximum_ms"] = max(row["maximum_ms"], duration)
        if (
            "Synchronize" not in event["name"] and "Memcpy" not in event["name"]
        ) or duration < 0.1:
            continue
        overlap = defaultdict(float)
        for kernel in kernels:
            ns = min(kernel["end"], event["end"]) - max(kernel["start"], event["start"])
            if ns > 0:
                overlap[kernel["name"]] += ns / 1e6
        waits.append(
            {
                "api": event["name"],
                "copies": [
                    dict(copy)
                    for copy in database.execute(
                        "SELECT m.bytes, e.label direction, (m.end-m.start)/1e6 gpu_ms "
                        "FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind "
                        "WHERE m.correlationId=?",
                        (event["correlationId"],),
                    )
                ],
                "host_ms": duration,
                "gpu_kernel_overlap_ms": dict(
                    sorted(overlap.items(), key=lambda x: -x[1])
                ),
            }
        )
    copies = [
        dict(row)
        for row in database.execute(
            "SELECT e.label direction, count(*) calls, sum(m.bytes) bytes, "
            "sum(m.end-m.start)/1e6 gpu_ms FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
            "JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind GROUP BY e.label"
        )
    ]
    write(
        path.stem + "-summary.json",
        {
            "scope": "Intrusive cudaProfilerApi window with host-only tracing; no VIBEQC_DF_TRACE fences. Kernel sums and clipped wait overlaps are GPU activity, not wall-time speedups.",
            "processes": [
                dict(row) for row in database.execute("SELECT * FROM PROCESSES")
            ],
            "sqlite_sha256": sha(path),
            "sqlite_bytes": path.stat().st_size,
            "kernels": sorted(grouped.values(), key=lambda row: -row["gpu_ms"]),
            "runtime_apis": apis,
            "blocking_api_intervals": waits,
            "copies": copies,
            "achieved_occupancy": None,
            "stall_reasons": None,
            "branch_efficiency": None,
            "dram_transactions": None,
        },
    )
    database.close()

for aos in (384, 768):
    path = source / f"{aos}-diis.json"
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    write(path.name, result)
    default = source / f"{aos}-default.json"
    if default.exists():
        write(default.name, json.loads(default.read_text()))
    host = {}
    for policy in ("serial", "auto"):
        trace = source / f"{aos}-diis.0-{policy}.host.jsonl"
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        regions = [region for record in records for region in record["regions"]]
        totals = {}
        for region in regions:
            row = totals.setdefault(
                region["name"], {"calls": 0, "wall_ms": 0, "cpu_ms": 0}
            )
            row["calls"] += 1
            row["wall_ms"] += region["wall_ms"]
            row["cpu_ms"] += region["cpu_ms"] or 0
        host[policy] = {
            "trace_sha256": sha(trace),
            "regions": regions,
            "inclusive_totals": totals,
        }
    write(f"{aos}-host.json", host)
    sampler = source / f"{aos}-memory.csv"
    maxima = {}
    readings = 0
    for line in sampler.read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            pid, mib = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        readings += 1
        maxima[pid] = max(maxima.get(pid, 0), mib * 1024**2)
    write(
        f"{aos}-memory.json",
        {
            "scope": "100 ms nvidia-smi process sampling during separate profiled cold and warm runs; sampled lower-bound high-water includes opaque runtime/module storage, not an exact allocation peak.",
            "sampler_sha256": sha(sampler),
            "readings": readings,
            "maximum_bytes_per_pid": maxima,
        },
    )
write("environment.json", json.loads((source / "environment.json").read_text()))

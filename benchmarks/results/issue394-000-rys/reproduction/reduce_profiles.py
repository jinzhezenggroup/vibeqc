"""Reduce separate Nsight captures to activity, transfer and memory evidence.

Activity sums and sampled high-water marks are diagnostic measurements, not
clean endpoint times or exact allocation peaks. Hardware occupancy is not
inferred from these traces.
"""

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from collect import compact_json

SOURCE = Path(".artifacts/issue394-000/profiles")
DESTINATION = Path("benchmarks/results/issue394-000-rys/profiles")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(name, data):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / name).write_text(compact_json(data) + "\n")


def shell_classes(kernels):
    """Sum every repeated panel in the uninstrumented CUDA activity capture.

    The detailed source ledger is a different execution. Do not join its event
    intervals with these kernels or infer per-signature timings from work counts.
    """
    classes = defaultdict(lambda: {"gpu_ms": 0.0, "launches": 0, "resources": []})
    for kernel in kernels:
        match = re.search(r"::shell_(?:panel|packet)<([^>]+)>", kernel["name"])
        if not match:
            continue
        parameters = [
            item.replace("(unsigned int)", "")
            .replace("(bool)", "")
            .strip()
            .removesuffix("u")
            for item in match[1].split(",")
        ]
        assert parameters[4] in ("0", "1", "false", "true")
        angular = tuple(map(int, parameters[:3]))
        row = classes[angular]
        row["gpu_ms"] += kernel["gpu_ms"]
        row["launches"] += kernel["calls"]
        row["resources"].append(
            {
                key: value
                for key, value in kernel.items()
                if key not in ("name", "gpu_ms", "calls")
            }
        )
        lowering = "rys" if parameters[4] in ("1", "true") else "polynomial"
        assert row.setdefault("lowering", lowering) == lowering
        row["variant"] = int(parameters[3])
    assert classes, "capture contains no derivative shell kernels"
    assert all(
        row["lowering"] == "polynomial" or angular == (0, 0, 0)
        for angular, row in classes.items()
    )
    return [
        {"angular": list(angular), **row} for angular, row in sorted(classes.items())
    ]


def main():
    """Keep every capture's identities and counters without committing databases."""
    target_pids = set()
    for path in sorted(SOURCE.glob("*.sqlite")):
        capture = int(path.stem.split(".")[-1])
        aos, policy = {
            1: (768, "polynomial"),
            2: (768, "rys"),
            3: (384, "polynomial"),
            4: (384, "rys"),
        }[capture]
        database = sqlite3.connect(path)
        database.row_factory = sqlite3.Row
        queries = {
            "kernels": """SELECT s.value name, count(*) calls,
                sum(k.end-k.start)/1e6 gpu_ms, min(registersPerThread) registers_min,
                max(registersPerThread) registers_max, max(staticSharedMemory) static_shared_bytes,
                max(dynamicSharedMemory) dynamic_shared_bytes,
                max(localMemoryPerThread) local_bytes_per_thread,
                min(blockX*blockY*blockZ) block_threads_min,
                max(blockX*blockY*blockZ) block_threads_max
                FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
                GROUP BY s.value ORDER BY gpu_ms DESC""",
            "runtime_apis": """SELECT s.value name, count(*) calls,
                sum(r.end-r.start)/1e6 host_ms FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds s ON s.id=r.nameId GROUP BY s.value ORDER BY host_ms DESC""",
            "copies": """SELECT e.label direction, count(*) calls, sum(m.bytes) bytes,
                sum(m.end-m.start)/1e6 gpu_ms FROM CUPTI_ACTIVITY_KIND_MEMCPY m
                JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind GROUP BY e.label""",
            "processes": "SELECT * FROM PROCESSES WHERE globalPid IN (SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL)",
        }
        result = {
            name: [dict(row) for row in database.execute(query)]
            for name, query in queries.items()
        }
        result.update(
            aos=aos,
            policy=policy,
            sqlite_sha256=sha(path),
            sqlite_bytes=path.stat().st_size,
            scope="Intrusive cudaProfilerApi capture; no component-trace fences",
        )
        # Match nvidia-smi's local timestamps to each capture's UTC origin;
        # retain the profiled process only, not the workstation process inventory.
        origin = database.execute(
            "SELECT * FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchone()
        offset = datetime.fromisoformat(origin["localTime"]) - datetime.fromisoformat(
            origin["utcTime"]
        )
        first, last = database.execute(
            "SELECT min(start), max(end) FROM (SELECT start,end FROM CUPTI_ACTIVITY_KIND_RUNTIME UNION ALL SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL)"
        ).fetchone()
        window = [origin["utcEpochNs"] + first, origin["utcEpochNs"] + last]
        pids = {row["pid"] for row in result["processes"]}
        target_pids.update(pids)
        result["shell_classes"] = shell_classes(result["kernels"])
        assert all(
            row["lowering"] == (policy if row["angular"] == [0, 0, 0] else "polynomial")
            for row in result["shell_classes"]
        )
        result["shell_kernel_gpu_ms"] = sum(
            row["gpu_ms"] for row in result["shell_classes"]
        )
        samples = []
        for line in (SOURCE / "memory.csv").read_text().splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            stamp = datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").replace(
                tzinfo=timezone(offset)
            )
            when = int(stamp.timestamp() * 1e9)
            if int(fields[1]) in pids and window[0] <= when <= window[1]:
                samples.append(int(fields[2]) * 1024**2)
        assert samples, "memory sampler did not overlap the CUDA activity window"
        result.update(
            cuda_activity_utc_ns=window,
            memory_sample_count=len(samples),
            sampled_device_peak_bytes=max(samples),
        )
        write(path.stem + ".json", result)
        database.close()
    for path in SOURCE.glob("*.host.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        write(path.stem + ".json", {"sha256": sha(path), "records": rows})
    for name in ("environment.json", "768.json", "384.json"):
        write(name, json.loads((SOURCE / name).read_text()))
    path = SOURCE / "memory.csv"
    maxima, readings = {}, 0
    for line in path.read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid, mib = int(fields[1]), int(fields[2])
        except ValueError:
            continue
        if pid not in target_pids:
            continue
        readings += 1
        maxima[pid] = max(maxima.get(pid, 0), mib * 1024**2)
    write(
        "memory.json",
        {
            "scope": "100 ms process device-memory sampling across initialization and warm captures; lower-bound high-water, including opaque runtime/module memory",
            "sha256": sha(path),
            "readings": readings,
            "maximum_bytes_per_pid": maxima,
            "profiler_launcher_time_report": (
                SOURCE / "process-residency.txt"
            ).read_text(),
            "host_rss_scope": "The time report measures the Nsight launcher, not the profiled Python endpoint; do not interpret its RSS as application residency.",
        },
    )


if __name__ == "__main__":
    main()

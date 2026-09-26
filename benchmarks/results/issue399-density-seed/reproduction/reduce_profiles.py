"""Reduce separate Nsight captures to activity, transfer and memory evidence.

Activity sums and sampled high-water marks are diagnostic measurements, not
clean endpoint times or exact allocation peaks. Hardware occupancy is not
inferred from these traces.
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from collect import compact_json

SOURCE = Path(".artifacts/issue399/profiles")
DESTINATION = Path("benchmarks/results/issue399-density-seed/profiles")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(name, data):
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / name).write_text(compact_json(data) + "\n")


def main():
    """Keep every capture's identities and counters without committing databases."""
    for path in sorted(SOURCE.glob("*.sqlite")):
        database = sqlite3.connect(path)
        database.row_factory = sqlite3.Row
        queries = {
            "kernels": """SELECT s.value name, count(*) calls,
                sum(k.end-k.start)/1e6 gpu_ms, min(registersPerThread) registers_min,
                max(registersPerThread) registers_max, max(staticSharedMemory) static_shared_bytes,
                max(dynamicSharedMemory) dynamic_shared_bytes,
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
    for path in SOURCE.glob("*.journal.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        stop = next(
            (i for i, row in enumerate(rows) if row.get("name") == "force_response"),
            len(rows),
        )
        write(
            path.stem + ".json",
            {
                "sha256": sha(path),
                "original_rows": len(rows),
                "scope": "SCF and final-state prefix, before force_response",
                "records": rows[:stop],
            },
        )
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
        readings += 1
        maxima[pid] = max(maxima.get(pid, 0), mib * 1024**2)
    write(
        "memory.json",
        {
            "scope": "100 ms process device-memory sampling across initialization and warm captures; lower-bound high-water, including opaque runtime/module memory",
            "sha256": sha(path),
            "readings": readings,
            "maximum_bytes_per_pid": maxima,
            "host_process_residency": (SOURCE / "process-residency.txt").read_text(),
        },
    )


if __name__ == "__main__":
    main()

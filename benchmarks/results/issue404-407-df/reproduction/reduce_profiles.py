"""Reduce the four separate Nsight warm captures without retaining databases.

Run from the repository root. Activity sums are intrusive diagnostics; 100 ms
process memory sampling is a lower bound on the simultaneous high-water mark.
"""

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for number, (aos, policy) in enumerate(
        (
            (384, "legacy-off"),
            (384, "candidate-reuse"),
            (768, "legacy-off"),
            (768, "candidate-reuse"),
        ),
        1,
    ):
        path = args.source / f"combined.{number}.sqlite"
        database = sqlite3.connect(path)
        database.row_factory = sqlite3.Row
        queries = {
            "kernels": """SELECT s.value name, count(*) calls,
                sum(k.end-k.start)/1e6 gpu_ms, min(registersPerThread) registers_min,
                max(registersPerThread) registers_max, max(staticSharedMemory) static_shared_bytes,
                max(dynamicSharedMemory) dynamic_shared_bytes
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
            key: [dict(row) for row in database.execute(query)]
            for key, query in queries.items()
        }
        with path.open("rb") as stream:
            result["sqlite_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
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
        for line in (args.source / "memory.csv").read_text().splitlines():
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
            aos=aos,
            policy=policy,
            cuda_activity_utc_ns=window,
            scope="Intrusive cudaProfilerApi capture; separate from clean timing",
            memory_sample_count=len(samples),
            sampled_device_peak_bytes=max(samples),
        )
        (args.destination / f"{aos}-{policy}.json").write_text(
            json.dumps(result, separators=(",", ":")) + "\n"
        )
        database.close()


if __name__ == "__main__":
    main()

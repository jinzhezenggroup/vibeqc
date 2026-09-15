"""Reduce an Nsight SQLite capture by synchronized GPU4PySCF NVTX scopes.

Usage: python reduce_profile.py capture.sqlite summary.json
No device is accessed. Copy directions come from the capture's enum table;
kernel times are summed GPU activity, not wall times or semantic work counts.
"""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path


def reduce_profile(path):
    """Retain copy counts/bytes and every kernel family within each force scope."""
    connection = sqlite3.connect(path)
    scopes = {}
    ranges = connection.execute(
        "SELECT start, end, text FROM NVTX_EVENTS "
        "WHERE text LIKE 'gpu4pyscf/%' ORDER BY start"
    )
    for begin, end, name in ranges:
        copies = connection.execute(
            "SELECT e.label, count(*), sum(m.bytes), sum(m.end-m.start) "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
            "JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind "
            "WHERE m.start>=? AND m.end<=? GROUP BY e.label",
            (begin, end),
        ).fetchall()
        kernels = connection.execute(
            "SELECT s.value, count(*), sum(k.end-k.start), "
            "min(registersPerThread), max(registersPerThread), "
            "max(staticSharedMemory), max(dynamicSharedMemory) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k.shortName "
            "WHERE k.start>=? AND k.end<=? GROUP BY s.value "
            "ORDER BY sum(k.end-k.start) DESC",
            (begin, end),
        ).fetchall()
        scopes[name] = {
            "wall_seconds": (end - begin) / 1e9,
            "copies": {
                label: {"calls": count, "bytes": size, "gpu_ms": ns / 1e6}
                for label, count, size, ns in copies
            },
            "kernels": [
                {
                    "name": label,
                    "calls": count,
                    "gpu_ms": ns / 1e6,
                    "registers_per_thread": [low, high],
                    "static_shared_peak_bytes": static,
                    "dynamic_shared_peak_bytes": dynamic,
                }
                for label, count, ns, low, high, static, dynamic in kernels
            ],
        }
    connection.close()
    return {
        "sqlite_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        "scope": "Intrusive profile. Synchronized NVTX boundaries; GPU activities entirely inside each range. Kernel totals exclude host overhead and are not complete force timings. Shell/primitive counts unavailable in GPU4PySCF.",
        "scopes": scopes,
    }


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(
        json.dumps(reduce_profile(Path(sys.argv[1])), indent=2, allow_nan=False) + "\n"
    )

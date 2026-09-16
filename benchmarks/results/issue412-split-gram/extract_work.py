"""Retain compact actual-work observations from local endpoint diagnostics.

Raw journals and component traces remain local; hashes bind these extracts to
their complete inputs. GPU regions are inclusive and must never be added into
an endpoint estimate. Captured construction records are not executed work.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

WORK_KEYS = {
    "device_iterations",
    "host_graph_replay",
    "graph_construction_attempt",
    "density_seed_rank",
    "accepted",
    "factorized",
    "converged",
    "compact_dispatch_status",
    "seed_generation",
}


def extract(directory):
    """Extract each arm without dropping failed, captured or zero-rank records."""
    results = {}
    for path in sorted(directory.glob("*-*.journal.jsonl")):
        stem = path.name.removesuffix(".journal.jsonl")
        separator = ".diagnostic-0-" if ".diagnostic-0-" in stem else ".0-"
        if separator not in stem:
            raise ValueError(f"unexpected diagnostic repetition: {path.name}")
        label, arm = stem.split(separator)
        if f"{label}-{arm}" in results:
            raise ValueError("multiple diagnostic records for one endpoint arm")
        trace = path.with_name(path.name.replace(".journal.jsonl", ".jsonl"))
        journal_data, trace_data = path.read_bytes(), trace.read_bytes()
        journal = [json.loads(line) for line in journal_data.splitlines()]
        records = [json.loads(line) for line in trace_data.splitlines()]
        regions = defaultdict(float)
        for row in records:
            if row["execution"] != "stream":
                continue
            for region in row["regions"]:
                if region["gpu_ms"] is not None:
                    regions[region["name"]] += region["gpu_ms"]
        results[f"{label}-{arm}"] = {
            "journal_sha256": hashlib.sha256(journal_data).hexdigest(),
            "trace_sha256": hashlib.sha256(trace_data).hexdigest(),
            "scope_counts": dict(
                Counter(row["name"] for row in journal if row["event"] == "BEGIN")
            ),
            "values": [
                {k: row[k] for k in ("name", "key", "value")}
                for row in journal
                if row["event"] == "VALUE"
                and (
                    row["key"] in WORK_KEYS
                    or row["key"].startswith(("final_", "seed_"))
                    or (row["key"] == "cuda_error" and row["value"] != 0)
                )
            ],
            "trace_failures": [
                {k: row[k] for k in ("operation", "valid", "cuda_error")}
                for row in records
                if row["execution"] == "stream"
                and (not row["valid"] or row["cuda_error"] != 0)
            ],
            "inclusive_gpu_region_ms": dict(regions),
            "occupied_gram_calls": [
                {
                    k: row[k]
                    for k in (
                        "execution",
                        "valid",
                        "cuda_error",
                        "nbf",
                        "naux",
                        "source_backed",
                        "streamed",
                        "counters",
                    )
                }
                for row in records
                if row["operation"] == "ri_k_occupied"
            ],
        }
    if not results:
        raise ValueError("no endpoint diagnostic journals found")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.directory), indent=2, allow_nan=False))

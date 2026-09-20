"""Read live #206/#308 phase journals without declaring unfinished work complete.

Unlike the completed CUDA/host ledgers, a killed run is an expected input here.
Its open scopes and truncated final line remain explicit. Timings are intrusive
diagnostic wall intervals; graph construction is never device execution.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path


def read_progress(path: Path) -> dict:
    """Validate scope lifetimes and preserve partial tails as incomplete evidence.

    The sink must be fresh and belong to one process. Concurrent roots are
    allowed, but their wall intervals must not be summed as endpoint time.
    Values remain ordered observations, including cumulative device readbacks.
    """
    content = Path(path).read_bytes()
    lines = content.splitlines(keepends=True)
    scopes: dict[int, dict] = {}
    truncated = bool(lines and not lines[-1].endswith(b"\n"))
    if truncated:
        lines.pop()
    for line in lines:
        row = json.loads(line)
        if row.get("schema") != "vibeqc.df_progress" or row.get("version") != 1:
            raise ValueError("unsupported progress schema")
        index, parent = row["id"], row["parent"]
        if type(index) is not int or index < 0 or type(parent) is not int:
            raise ValueError("invalid progress scope identity")
        if row["execution"] not in {"host", "stream", "graph_capture"}:
            raise ValueError("invalid progress execution mode")
        if type(row["time_ns"]) is not int or row["time_ns"] < 0:
            raise ValueError("invalid progress timestamp")
        if row["event"] == "BEGIN":
            if index in scopes:
                raise ValueError("duplicate progress BEGIN")
            if parent != -1 and (
                parent not in scopes or scopes[parent]["end"] is not None
            ):
                raise ValueError("progress parent is not active")
            scopes[index] = {"begin": row, "end": None, "values": [], "children": []}
            if parent != -1:
                scopes[parent]["children"].append(index)
            continue
        scope = scopes.get(index)
        if scope is None or scope["end"] is not None:
            raise ValueError("progress event outside active scope")
        begin = scope["begin"]
        if any(row[k] != begin[k] for k in ("parent", "name", "execution")):
            raise ValueError("progress scope identity changed")
        if row["time_ns"] < begin["time_ns"]:
            raise ValueError("progress clock went backwards")
        if row["event"] == "VALUE":
            if not isinstance(row.get("key"), str) or type(row.get("value")) not in (
                str,
                int,
            ):
                raise ValueError("invalid progress observation")
            scope["values"].append(row)
        elif row["event"] == "END":
            if any(scopes[child]["end"] is None for child in scope["children"]):
                raise ValueError("progress parent ended before child")
            if (
                row["execution"] == "graph_capture"
                and row["status"] == "stream_complete"
            ):
                raise ValueError("graph construction reported as device execution")
            scope["end"] = row
        else:
            raise ValueError("unknown progress event")
    pending = [scope["begin"] for scope in scopes.values() if scope["end"] is None]
    return {
        "scopes": scopes,
        "pending": pending,
        "truncated_tail": truncated,
        "complete": bool(scopes) and not pending and not truncated,
    }


def summarize_progress(journal: dict) -> dict:
    """Subtract completed immediate children; retain execution-qualified counts.

    Parentage links J/K to setup, retry or finalization. Device iteration
    readbacks are cumulative and are intentionally retained as observations;
    summing them or graph-construction counts would overstate physical work.
    """
    phases = defaultdict(lambda: {"calls": 0, "inclusive_ms": 0.0, "exclusive_ms": 0.0})
    scopes = journal["scopes"]
    observations = []
    for scope in scopes.values():
        begin, end = scope["begin"], scope["end"]
        observations.extend(scope["values"])
        if end is None:
            continue
        elapsed = (end["time_ns"] - begin["time_ns"]) / 1e6
        children = sum(
            (scopes[c]["end"]["time_ns"] - scopes[c]["begin"]["time_ns"]) / 1e6
            for c in scope["children"]
        )
        if children > elapsed + 1e-6:
            raise ValueError("progress child time exceeds parent")
        phase = phases[begin["execution"] + ":" + begin["name"]]
        phase["calls"] += 1
        phase["inclusive_ms"] += elapsed
        phase["exclusive_ms"] += max(0.0, elapsed - children)
    return {
        "schema": "vibeqc.df_progress_summary.v1",
        "complete_journal": journal["complete"],
        "truncated_tail": journal["truncated_tail"],
        "pending": journal["pending"],
        "phases": dict(phases),
        "observations": sorted(observations, key=lambda r: r["time_ns"]),
        "scope": "diagnostic wall time; completed journal does not prove convergence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    result = summarize_progress(read_progress(args.trace))
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()

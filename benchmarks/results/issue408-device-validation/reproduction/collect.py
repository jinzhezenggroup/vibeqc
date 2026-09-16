"""Publish completed #408 comparisons without retaining routine logs or traces.

Run from the repository root. Every clean sample, numerical result and identity
is retained; only repeated basis metadata is interned. Intrusive components are
separate records and never contribute to clean endpoint medians.
"""

import argparse
import copy
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.df_component_ledger import aggregate_host, read_host_trace


def digest(path):
    """Hash original bytes so reductions can be audited against local artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_json(value, level=0):
    """Keep numeric rows and flat event records on one line without losing data."""

    def numeric_tensor(item):
        return isinstance(item, (int, float)) or (
            isinstance(item, list) and all(numeric_tensor(child) for child in item)
        )

    if isinstance(value, dict):
        if all(not isinstance(item, (dict, list)) for item in value.values()):
            return json.dumps(value, allow_nan=False)
        fields = [
            json.dumps(str(key)) + ": " + compact_json(item, level + 2)
            for key, item in value.items()
        ]
        opening, closing = "{", "}"
    elif isinstance(value, list):
        if numeric_tensor(value) or all(
            not isinstance(item, (dict, list)) for item in value
        ):
            return json.dumps(value, allow_nan=False)
        fields = [compact_json(item, level + 2) for item in value]
        opening, closing = "[", "]"
    else:
        return json.dumps(value, allow_nan=False)
    return (
        opening
        + "\n"
        + ",\n".join(" " * (level + 2) + item for item in fields)
        + "\n"
        + " " * level
        + closing
    )


def write(path, data):
    """Refuse oversized evidence instead of silently discarding measurements."""
    raw = compact_json(data) + "\n"
    assert json.loads(raw) == data
    assert len(raw.encode()) < 1024**2, path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)


def retain(path, destination):
    """Validate the paired experiment and retain independent diagnostic ledgers."""
    data = json.loads(path.read_text())
    assert data["scope"] == "clean endpoint"
    assert data["policies"] == ["1", "0"]
    assert len(data["samples"]) == 10 and len(data["diagnostics"]) == 2
    # Declare the workload before measuring, rather than blessing whichever
    # iteration branch the first sample happened to take.
    expected = data["expected_iterations"]
    assert type(expected) is int and expected > 0, (
        "declare positive expected_iterations"
    )
    iterations = [expected]
    for row in data["samples"] + data["diagnostics"]:
        assert row["iterations"] == iterations
        assert row["prime_iterations"] == iterations
        assert row["maximum_energy_error"] <= 1e-9
        assert row["maximum_force_error"] is None or row["maximum_force_error"] <= 1e-8
        assert all(item["converged"] for item in row["convergence"])
    assert all(row["scope"] == "clean endpoint" for row in data["samples"])
    for repeat in range(5):
        assert [
            row["policy"] for row in data["samples"][2 * repeat : 2 * repeat + 2]
        ] == (["1", "0"] if repeat % 2 == 0 else ["0", "1"])
    retained = copy.deepcopy(data)
    retained["original_json_sha256"] = digest(path)
    retained["original_json_bytes"] = path.stat().st_size
    retained["basis_metadata"] = retained["initialization_convergence"][0][
        "basis_metadata"
    ]
    for rows in [
        retained["initialization_convergence"],
        retained.get("cold_convergence") or [],
        *(row["convergence"] for row in retained["samples"] + retained["diagnostics"]),
    ]:
        for row in rows:
            assert row.pop("basis_metadata") == retained["basis_metadata"]
    components = []
    for row in retained["diagnostics"]:
        policy = row["policy"]
        gpu = row.pop("components")
        host_path = path.with_suffix(f".diagnostic-0-{policy}.host.jsonl")
        host = read_host_trace(host_path)
        inclusive = defaultdict(lambda: {"calls": 0, "wall_ms": 0.0, "cpu_ms": 0.0})
        for record in host:
            for region in record["regions"]:
                target = inclusive[region["name"]]
                target["calls"] += 1
                target["wall_ms"] += region["wall_ms"]
                if region["cpu_ms"] is None:
                    target["cpu_ms"] = None
                elif target["cpu_ms"] is not None:
                    target["cpu_ms"] += region["cpu_ms"]
        observations = {
            item["key"]: item["value"] for item in row["final_state_observations"]
        }
        assert observations["final_fock_evaluations"] == 1
        for key in (
            "final_eigen_solves",
            "final_density_updates",
            "final_candidate_rejections",
        ):
            assert observations[key] == 0
        if data["measured_properties"] == ["energy"]:
            assert "final_state_weighted_density" not in inclusive
            assert "force_response" not in inclusive
            assert not any(
                group["operation"] == "final_state_weighted_density"
                for group in gpu["groups"]
            )
        components.append(
            {
                "policy": policy,
                "scope": "intrusive diagnostic; inclusive regions overlap",
                "gpu": gpu,
                "host": aggregate_host(host),
                "host_inclusive": dict(inclusive),
                "observations": observations,
                "original_trace_sha256": {
                    suffix: digest(path.with_suffix(f".diagnostic-0-{policy}.{suffix}"))
                    for suffix in ("host.jsonl", "jsonl", "journal.jsonl")
                },
            }
        )
    write(destination / "measurements" / path.name, retained)
    write(destination / "components" / path.name, components)
    timings = {}
    for policy in data["policies"]:
        rows = [row for row in data["samples"] if row["policy"] == policy]
        timings[policy] = {
            "seconds": [row["seconds"] for row in rows],
            "median_seconds": statistics.median(row["seconds"] for row in rows),
            "maximum_energy_error": max(row["maximum_energy_error"] for row in rows),
            "maximum_force_error": (
                None
                if data["measured_properties"] == ["energy"]
                else max(row["maximum_force_error"] for row in rows)
            ),
            "iterations": [row["iterations"] for row in rows],
        }
    timings["reduction_fraction"] = (
        1 - timings["0"]["median_seconds"] / timings["1"]["median_seconds"]
    )
    timings["warm_density_sha256"] = data["warm_density_sha256"]
    return timings


def main():
    """Require the full size/property matrix before publishing any summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    timings = {}
    for aos in (96, 192, 384, 768):
        for endpoint in ("energy", "forces"):
            name = f"{aos}-{endpoint}"
            timings[name] = retain(args.source / f"{name}.json", args.destination)
        assert (
            timings[f"{aos}-energy"]["warm_density_sha256"]
            == timings[f"{aos}-forces"]["warm_density_sha256"]
        )
        for policy in ("1", "0"):
            assert (
                timings[f"{aos}-energy"][policy]["iterations"]
                == timings[f"{aos}-forces"][policy]["iterations"]
            ), "energy and force endpoints must perform the same SCF work"
    write(args.destination / "timings.json", timings)
    for aos in (384, 768):
        path = args.source / f"{aos}-warm-memory.json"
        data = json.loads(path.read_text())
        assert len(data["samples"]) == 4
        iterations = timings[f"{aos}-energy"]["1"]["iterations"][0]
        for row in data["samples"]:
            assert row["iterations"] == iterations
            assert row["maximum_energy_error"] <= 1e-9
            assert (
                row["maximum_force_error"] is None or row["maximum_force_error"] <= 1e-8
            )
        data["original_json_sha256"] = digest(path)
        write(args.destination / "memory" / path.name, data)
    patch = args.destination / "reproduction" / "source.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_bytes((args.source / "source.patch").read_bytes())


if __name__ == "__main__":
    main()

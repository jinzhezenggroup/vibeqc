"""Publish complete #395 measurements and reconcile disabled/enabled work.

Usage: python collect_evidence.py ARTIFACT_ROOT OUTPUT
Inputs must be complete before any output is created. Logs, checkpoints,
libraries and profiler databases remain outside Git.
"""

import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

from benchmarks.df_component_ledger import read_trace
from benchmarks.df_shell_work_ledger import kernel_activity

root, destination = map(Path, sys.argv[1:])
qualification = root / "work-v1/qualification"
required = [
    qualification / f"{aos}-{name}.json"
    for aos in (384, 768)
    for name in (
        "baseline-2",
        "baseline-3",
        "candidate-2",
        "candidate-3",
        "overhead",
        "profile",
        "work-ledger",
    )
]
required += [qualification / f"{aos}-work.1.sqlite" for aos in (384, 768)]
required += [qualification / f"{aos}-profile.0-0.jsonl" for aos in (384, 768)]
required += [root / "work-v1/build.json", root / "work-v1/environment.json"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(", ".join(missing))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def force(sample):
    groups = [
        group
        for group in sample["components"]["groups"]
        if group["operation"] == "force_response" and group["execution"] == "stream"
    ]
    if len(groups) != 1 or groups[0]["calls"] != 1:
        raise ValueError("expected one complete force response")
    return groups[0]


def derivative_time(group):
    # The named parent is exclusive in this aggregate. Add its class children
    # exactly once; unrelated response products remain outside this component.
    return sum(
        value
        for name, value in group["gpu_exclusive_ms"].items()
        if name == "three_center_derivative_contraction"
        or re.fullmatch(r"shell_[0-3]{3}_(?:packet|p\d+_\d+_\d+)", name)
    )


def maximum_difference(first, second):
    """Compare nested retained result arrays without changing their precision."""
    if isinstance(first, list) and isinstance(second, list):
        return max(maximum_difference(a, b) for a, b in zip(first, second, strict=True))
    return abs(first - second)


records, summary = {}, {}
build = read(root / "work-v1/build.json")
for aos in (384, 768):
    measurements = {}
    for name in (
        "baseline-2",
        "baseline-3",
        "candidate-2",
        "candidate-3",
        "overhead",
        "profile",
    ):
        data = read(qualification / f"{aos}-{name}.json")
        for sample in data["samples"] + data.get("diagnostics", []):
            if (
                sample["iterations"] != [3]
                or not math.isfinite(sample["seconds"])
                or sample["seconds"] <= 0
                or not 0 <= sample["maximum_energy_error"] <= 1e-9
                or not 0 <= sample["maximum_force_error"] <= 1e-8
            ):
                raise ValueError(f"numerical/iteration gate failed: {aos}-{name}")
        records[f"measurements/{aos}-{name}.json"] = data
        measurements[name] = data
    ledger = read(qualification / f"{aos}-work-ledger.json")
    records[f"work/{aos}.json"] = ledger
    candidates = [
        measurements[name]
        for name in ("candidate-2", "candidate-3", "overhead", "profile")
    ]
    for field in ("native_source_identity", "library_sha256", "source_patch_sha256"):
        if len({data[field] for data in [*candidates, ledger]}) != 1:
            raise ValueError(f"candidate identity changed: {field}")
    if (
        ledger["library_sha256"] != build["libvibeqc.so"]["sha256"]
        or ledger["source_patch_sha256"] != build["source.patch"]["sha256"]
        or ledger["generated_schedule_sha256"]
        != build["generated_df_shell_derivatives.cuh"]["sha256"]
    ):
        raise ValueError("ledger identity differs from the frozen build")
    hashes = {
        json.dumps(data["warm_density_sha256"], sort_keys=True)
        for data in measurements.values()
    }
    if len(hashes) != 1:
        raise ValueError("qualification changed the frozen input density")
    result = {"warm_density_sha256": json.loads(hashes.pop()), "policies": {}}
    work = []
    for policy in ("baseline", "candidate"):
        documents = [measurements[f"{policy}-{n}"] for n in (2, 3)]
        samples = [sample for document in documents for sample in document["samples"]]
        if len(samples) != 5 or any(
            sample["scope"] != "clean endpoint"
            or sample["policy"] != "0"
            or sample["iterations"] != [3]
            for sample in samples
        ):
            raise ValueError(
                "requires five clean disabled-counter samples with three updates"
            )
        components = [force(document["diagnostics"][0]) for document in documents]
        work.extend(group["counter_sums"] for group in components)
        result["policies"][policy] = {
            "seconds": [sample["seconds"] for sample in samples],
            "median_seconds": statistics.median(
                sample["seconds"] for sample in samples
            ),
            "maximum_energy_error": max(
                sample["maximum_energy_error"] for sample in samples
            ),
            "maximum_force_error": max(
                sample["maximum_force_error"] for sample in samples
            ),
            "component_derivative_ms": [derivative_time(group) for group in components],
            "post_call_resident_bytes": [
                document["diagnostics"][0]["process_device_resident_bytes"]
                for document in documents
            ],
        }
    ratio = (
        result["policies"]["candidate"]["median_seconds"]
        / result["policies"]["baseline"]["median_seconds"]
    )
    result["disabled_endpoint_ratio"] = ratio
    result["disabled_endpoint_2_percent_gate_passed"] = ratio <= 1.02
    overhead = {
        sample["policy"]: sample for sample in measurements["overhead"]["samples"]
    }
    if set(overhead) != {"0", "1"}:
        raise ValueError("requires both intrusive counter policies")
    work.extend(force(sample)["counter_sums"] for sample in overhead.values())
    work.extend(
        force(sample)["counter_sums"] for sample in measurements["profile"]["samples"]
    )
    for field in (
        "shell_triples_visited",
        "shell_triples_nonzero",
        "shell_primitive_products",
        "shell_public_weights_consumed",
        "shell_public_weights_nonzero",
        "shell_cartesian_component_products",
    ):
        if len({row[field] for row in work}) != 1:
            raise ValueError(f"scientific work changed: {field}")
    result["scientific_work_identical"] = True
    result["intrusive_overhead"] = {
        "scope": "Separate traced runs; never clean endpoint timing or a performance claim.",
        "disabled_seconds": overhead["0"]["seconds"],
        "enabled_seconds": overhead["1"]["seconds"],
        "ratio": overhead["1"]["seconds"] / overhead["0"]["seconds"],
        "response_resources": {
            policy: {
                key: value
                for key, value in force(sample)["counter_maxima"].items()
                if key
                in (
                    "shell_work_diagnostic_bytes",
                    "response_scratch_bytes",
                    "response_probe_total_device_bytes",
                    "device_to_host_bytes",
                    "host_to_device_bytes",
                    "stream_synchronizations",
                )
            }
            for policy, sample in overhead.items()
        },
    }
    result["numerical_equivalence"] = {}
    for label, first, second in (
        (
            "clean_baseline_candidate",
            measurements["baseline-2"]["samples"][0],
            measurements["candidate-2"]["samples"][0],
        ),
        ("detailed_counter_disabled_enabled", overhead["0"], overhead["1"]),
    ):
        result["numerical_equivalence"][label] = {
            field: maximum_difference(first[field], second[field])
            for field in ("energies_hartree", "forces_hartree_per_bohr")
        }
    trace = qualification / f"{aos}-profile.0-0.jsonl"
    force_records = [
        row
        for row in read_trace(trace)
        if row["operation"] == "force_response" and row["execution"] == "stream"
    ]
    if len(force_records) != 1:
        raise ValueError("expected one profiled force call")
    database = qualification / f"{aos}-work.1.sqlite"
    activities = kernel_activity(database, force_records[0])
    records[f"profiles/{aos}-disabled.json"] = {
        "scope": "Actual Nsight kernel activities from the disabled detailed-counter capture.",
        "sqlite_sha256": digest(database),
        "trace_sha256": digest(trace),
        "classes": [
            {"angular": angular, **values} for angular, values in activities.items()
        ],
    }
    summary[str(aos)] = result

records["summary.json"] = summary
records["build.json"] = build
records["environment.json"] = read(root / "work-v1/environment.json")
encoded = {
    name: json.dumps(value, indent=2, allow_nan=False) + "\n"
    for name, value in records.items()
}
for name, value in encoded.items():
    if len(value.encode()) > 1024**2:
        raise ValueError(f"needs a smaller observable bundle: {name}")
if destination.exists() and any(destination.iterdir()):
    raise ValueError("refusing to overwrite a nonempty publication directory")
for name, value in encoded.items():
    path = destination / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
print(json.dumps(summary, indent=2))

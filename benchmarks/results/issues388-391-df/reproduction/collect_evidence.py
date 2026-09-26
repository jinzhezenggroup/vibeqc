"""Retain reviewed samples and derive summaries without changing their values.

Run after all finite Slurm jobs finish. Checkpoints, compiler products, full
traces and routine logs stay in the artifact directory. JSON measurements
retain their original ordering and numerical output; derived summaries label
branch mismatches rather than normalizing time by iteration count.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)

# An empty glob must not silently turn an unfinished job into a publication.
required = [
    f"{category}/{aos}-{suffix}.json"
    for category, suffix in (("baseline", "clean"), ("final", "normal"))
    for aos in (384, 768)
]
required += [
    f"final/ablations/{aos}-{control}.json"
    for aos in (384, 768)
    for control in (
        "VIBEQC_DF_RESIDENT_EXCHANGE",
        "VIBEQC_DF_RAW_REUSE",
        "VIBEQC_DF_RESPONSE_BATCHING",
        "VIBEQC_DF_DIIS_DOTS",
        "VIBEQC_DF_RESIDENT_EXCHANGE-energy",
    )
]
required += [
    "final/ablations/384-VIBEQC_DF_RESPONSE_STORAGE.json",
    "final/ablations/384-VIBEQC_DF_DIIS_DOTS-energy.json",
    "final/ablations/768-DIIS-legacy-endpoint.json",
    "final/ablations/768-DIIS-legacy-energy.json",
]
for name in required:
    if not (root / name).is_file():
        raise RuntimeError(f"missing required measurement: {name}")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def review_json(value, level=0):
    """Keep scalar arrays together without changing values or record order."""
    indent = "  " * level
    child = indent + "  "
    if isinstance(value, dict):
        if not value:
            return "{}"
        fields = (
            child + json.dumps(key) + ": " + review_json(item, level + 1)
            for key, item in value.items()
        )
        return "{\n" + ",\n".join(fields) + "\n" + indent + "}"
    if isinstance(value, list) and any(
        isinstance(item, (list, dict)) for item in value
    ):
        items = (child + review_json(item, level + 1) for item in value)
        return "[\n" + ",\n".join(items) + "\n" + indent + "]"
    return json.dumps(value, allow_nan=False)


def write(path, value):
    target = destination / path
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (review_json(value) + "\n").encode()
    if len(encoded) > 1024**2:
        raise RuntimeError(f"split a scientific record by observable: {path}")
    target.write_bytes(encoded)


def retain(source, path):
    if source.suffix == ".json":
        write(path, json.loads(source.read_text()))
        return
    target = destination / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_size > 1024**2:
        raise RuntimeError(f"oversized retained record: {source}")
    shutil.copyfile(source, target)


def summary(document):
    policies = {}
    for policy in document["policies"]:
        samples = [s for s in document["samples"] if s["policy"] == policy]
        seconds = [s["seconds"] for s in samples]
        policies[policy] = {
            "seconds": seconds,
            "median_seconds": statistics.median(seconds),
            "iterations": [s["iterations"] for s in samples],
            "maximum_energy_error": max(s["maximum_energy_error"] for s in samples),
            "maximum_force_error": max(
                (
                    s["maximum_force_error"]
                    for s in samples
                    if s["maximum_force_error"] is not None
                ),
                default=None,
            ),
            "density_rms": [
                [c["density_rms"] for c in s["convergence"]] for s in samples
            ],
        }
    branches = {tuple(s["iterations"]) for s in document["samples"]}
    return {
        "scope": document["scope"],
        "properties": document.get("measured_properties", ["energy", "forces"]),
        "common_iteration_branch": len(branches) == 1,
        "density_sha256": document.get("warm_density_sha256"),
        "cold_seconds": document.get("cold_seconds"),
        "cold_iterations": [
            s["iterations"] for s in document.get("cold_convergence") or []
        ],
        "policies": policies,
    }


def region_totals(trace):
    """Reduce executed stream regions; captured declarations are not timings."""
    totals = {}
    for line in trace.read_text().splitlines():
        record = json.loads(line)
        if record["execution"] != "stream":
            continue
        for region in record["regions"]:
            row = totals.setdefault(region["name"], {"calls": 0, "gpu_inclusive_ms": 0})
            row["calls"] += 1
            row["gpu_inclusive_ms"] += region["gpu_ms"]
    return {"trace_sha256": digest(trace), "inclusive_regions": totals}


timings = {}
component_records = {}
for category, paths in (
    ("baseline", sorted((root / "baseline").glob("*-clean.json"))),
    ("normal", sorted((root / "final").glob("*-normal.json"))),
    ("ablations", sorted((root / "final/ablations").glob("*.json"))),
    ("derivatives", sorted((root / "derivative-ablations").glob("*.json"))),
):
    for path in paths:
        document = json.loads(path.read_text())
        name = f"{category}/{path.stem}"
        expected = document.get("expected_iterations")
        if expected and any(
            count != expected
            for sample in document["samples"]
            for count in sample["iterations"]
        ):
            retain(path, f"rejected/{path.name}")
            continue
        if category != "derivatives" and any(
            len([s for s in document["samples"] if s["policy"] == policy]) < 5
            for policy in document["policies"]
        ):
            raise RuntimeError(f"incomplete comparison: {path}")
        retain(path, f"measurements/{name}.json")
        timings[name] = summary(document)
        for sample in document.get("diagnostics", []):
            component_records[f"{name}/{sample['policy']}"] = {
                "scope": "Separate intrusive component pass after all clean samples",
                **region_totals(
                    path.with_suffix(
                        f".diagnostic-{sample['repeat']}-{sample['policy']}.jsonl"
                    )
                ),
                "iterations": sample["iterations"],
                "force_stage_seconds": sample["force_stage_seconds"],
                "process_device_resident_bytes": sample[
                    "process_device_resident_bytes"
                ],
                "native_reservations": sample["metric"],
                "groups": sample["components"]["groups"],
            }
for aos in (384, 768):
    baseline = json.loads((root / f"baseline/{aos}-trace.json").read_text())
    retain(
        root / f"baseline/{aos}-trace.json",
        f"measurements/baseline/{aos}-components.json",
    )
    for sample in baseline["samples"]:
        component_records[f"baseline/{aos}/{sample['policy']}"] = {
            "scope": "Separate historical baseline intrusive component pass",
            **region_totals(
                (root / f"baseline/{aos}-trace.json").with_suffix(
                    f".{sample['repeat']}-{sample['policy']}.jsonl"
                )
            ),
            "iterations": sample["iterations"],
            "force_stage_seconds": sample.get("force_stage_seconds"),
            "process_device_resident_bytes": sample.get(
                "process_device_resident_bytes"
            ),
            "native_reservations": sample["metric"],
            "groups": sample["components"]["groups"],
        }
    for variant in ("legacy", "shared", "center"):
        entries = [
            timings[f"derivatives/{aos}-{phase}-{variant}"]["policies"]["auto"]
            for phase in ("forward", "reverse")
        ]
        seconds = [s for entry in entries for s in entry["seconds"]]
        if len(seconds) != 5:
            raise RuntimeError("derivative ABCCBA blocks must retain five samples")
        densities = {
            json.dumps(
                timings[f"derivatives/{aos}-{phase}-{v}"]["density_sha256"],
                sort_keys=True,
            )
            for phase in ("forward", "reverse")
            for v in ("legacy", "shared", "center")
        }
        if len(densities) != 1:
            raise RuntimeError("derivative starting density differs")
        timings[f"derivatives/{aos}-combined-{variant}"] = {
            "scope": "Sequential owner reconstruction, ABCCBA blocks; two then three samples per variant",
            "seconds": seconds,
            "median_seconds": statistics.median(seconds),
            "iterations": [i for entry in entries for i in entry["iterations"]],
            "density_sha256": json.loads(densities.pop()),
        }
write("timings.json", timings)
for aos in (384, 768):
    write(
        f"components/{aos}.json",
        {
            "scope": "Component records grouped by workload; original logical paths are preserved as keys.",
            "records": {
                f"{name}.json": record
                for name, record in component_records.items()
                if str(aos) in name
            },
        },
    )
for name in component_records:
    # Retire only the superseded layout after the complete bundle is written.
    (destination / f"components/{name}.json").unlink(missing_ok=True)

for name in ("final", "derivative-legacy", "derivative-shared", "derivative-center"):
    retain(root / name / "source.patch", f"reproduction/{name}-source.patch")
retain(root / "generated-work.json", "generated-work.json")
for script in (
    "baseline.sh",
    "final-carry-ablations.sh",
    "derivative-ablations.sh",
    "final-continuation.sh",
    "profile-final.sh",
    "collect_evidence.py",
    "reduce_profiles.py",
    "generated_work.py",
):
    retain(root / script, f"reproduction/{script}")
profiles = sorted((root / "profile-reduced").glob("*.json"))
for aos in (384, 768):
    write(
        f"profiles/{aos}.json",
        {
            "scope": "Reduced profiles grouped by workload; original logical paths are preserved as keys.",
            "records": {
                profile.name: json.loads(profile.read_text())
                for profile in profiles
                if profile.name.startswith(f"{aos}-")
            },
        },
    )
for profile in profiles:
    if profile.name.startswith(("384-", "768-")):
        (destination / f"profiles/{profile.name}").unlink(missing_ok=True)
    else:
        retain(profile, f"profiles/{profile.name}")

builds = {}
for name in (
    "baseline",
    "final",
    "derivative-legacy",
    "derivative-shared",
    "derivative-center",
):
    files = {}
    for filename in (
        "libvibeqc.so",
        "df_shell_derivatives.cu.o",
        "generated_df_shell_derivatives.cuh",
        "source.patch",
    ):
        path = root / name / filename
        if path.exists():
            files[filename] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    builds[name] = {
        "files": files,
        "configuration": "Release, sm_120, -O3 -DNDEBUG; CUDA fast compile disabled",
        "line_information": name == "baseline",
    }
write("builds.json", builds)

rejected = root / "qualified/ablations/768-VIBEQC_DF_RESIDENT_EXCHANGE.json"
retain(rejected, "rejected/panel-reassociation.json")
write(
    "rejected/decisions.json",
    {
        "panel_reassociation": "Stopped after identifying a three-versus-five iteration mismatch on identical density. Continuing the Q sum across storage panels fixes this. These partial samples are not promotion evidence.",
        "flat_dense": "Complete five-repeat ablation is retained with final runtime. A long-K flattened dense GEMM regresses total endpoint despite the occupied K improvement.",
        "candidate2_diis": "An early partial-dot grid treated physical slots as a prefix. It failed the independently accumulated poisoned circular-history regression. Excluded from all accepted final results; the fixed kernel checks the live circular window.",
        "warp_only_diis": "Insufficient parallelism. Replaced by deterministic block partials using already reserved temporary storage.",
        "historical_binary_size": "Baseline includes line information; only the three matched derivative variants establish binary-size deltas.",
    },
)
print(
    json.dumps(
        {"measurements": len(timings), "component_records": len(component_records)},
        indent=2,
    )
)

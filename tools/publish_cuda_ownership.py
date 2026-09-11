"""Publish a compact, lossless ownership comparison through the evidence API.

Raw worker JSON remains transient. Repeated input, plan, observation and build
records are interned by canonical hash; every timing, final result, residual
and iteration count is retained. Publication does not change source selectors.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

import numpy as np
from vibeqc_compiler.common.evidence import (
    block_error,
    canonical_hash,
    new_evidence,
    outcome,
    write_evidence,
)
from vibeqc_compiler.common.performance import assess_comparison
from vibeqc_compiler.common.timing import interleaved_selection_order

from tools.vibeqc_validation.publication import publish


def write(path, value):
    """Write deterministic finite JSON without discarding float precision."""
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )


def compact_comparison(directory):
    """Recompute every gate from raw workers, preserving the shared timing assessment."""
    original = json.loads((directory / "comparison.json").read_text())
    count = original["samples"]
    if type(count) is not int or count < 5:
        raise ValueError("at least five samples per source are required")
    records, counters = [], {"baseline": 0, "candidate": 0}
    for label in interleaved_selection_order(count):
        sample = counters[label]
        counters[label] += 1
        records.append(
            (
                label,
                sample,
                json.loads((directory / f"{label}-{sample}.json").read_text()),
            )
        )
    inventories = [[e["case"] for e in run["endpoints"]] for _, _, run in records]
    if any(ids != inventories[0] for ids in inventories) or not inventories[0]:
        raise ValueError("raw inventories differ or are empty")
    if len(set(inventories[0])) != len(inventories[0]):
        raise ValueError("duplicate case IDs")
    sources = {}
    interned = {}
    domain = records[0][2].get("domain", "one-electron")
    if domain not in ("one-electron", "df") or any(
        run.get("domain", "one-electron") != domain for _, _, run in records
    ):
        raise ValueError("unknown or inconsistent ownership domain")
    process_scope = records[0][2].get("process_scope", "inventory")
    driver = records[0][2].get("benchmark_driver_sha256")
    if (
        process_scope not in ("inventory", "case")
        or original.get("process_scope", "inventory") != process_scope
        or original.get("benchmark_driver_sha256") != driver
        or any(
            run.get("process_scope", "inventory") != process_scope
            or run.get("benchmark_driver_sha256") != driver
            for _, _, run in records
        )
        or (process_scope == "case" and not re.fullmatch(r"[0-9a-f]{64}", driver or ""))
    ):
        raise ValueError("inconsistent process scope or benchmark driver provenance")

    def intern(value):
        key = canonical_hash(value)
        interned[key] = value
        return key

    compact = {
        "schema": "vibeqc.cuda-ownership-samples.v1",
        "records": interned,
        "runs": [],
    }
    for label, sample, run in records:
        if domain == "df" and any("energy_only" not in e for e in run["endpoints"]):
            raise ValueError("DF publication requires every energy-only endpoint")
        source = (run["revision"], run["native_source_identity"], run["library_sha256"])
        if run["dirty"] or source != sources.setdefault(label, source):
            raise ValueError("dirty or changing measured source/binary")
        if (
            run["schema"] != "vibeqc.cuda-ownership-endpoints.v1"
            or run["selection"] != ("reference" if label == "baseline" else "generated")
            or not run["slurm_job_id"]
            or not run["gpu"]
            or "CMAKE_BUILD_TYPE:STRING=Release" not in run["build_settings"]
            or "VIBEQC_CUDA_FAST_COMPILE:BOOL=OFF" not in run["build_settings"]
        ):
            raise ValueError("missing optimized scheduled-worker provenance")
        compact["runs"].append(
            {
                "selection": label,
                "sample": sample,
                "provenance": intern(
                    {k: v for k, v in run.items() if k != "endpoints"}
                ),
                "endpoints": [
                    {
                        "case": e["case"],
                        "input": intern(
                            {
                                k: e[k]
                                for k in (
                                    "case",
                                    "atoms",
                                    "basis",
                                    "df_budget_bytes",
                                    "resource_scope_note",
                                )
                            }
                        ),
                        "resource_plan": intern(e["resource_plan"]),
                        "observed_resources": intern(e["observed_resources"]),
                        "density_fitting": intern(e["density_fitting"]),
                        "seconds": e["seconds"],
                        "results": e["results"],
                        **({"energy_only": e["energy_only"]} if domain == "df" else {}),
                    }
                    for e in run["endpoints"]
                ],
            }
        )
    errors, timings, rows = {}, [], []
    phases = {
        "cold": "cold-start",
        "warm": "unchanged-geometry",
        "moved": "changed-geometry",
        "restored": "restored-geometry",
    }
    for i, case in enumerate(inventories[0]):
        base = records[0][2]["endpoints"][i]
        case_timings = []
        for label, sample, run in records:
            endpoint = run["endpoints"][i]
            keys = ("case", "atoms", "basis", "df_budget_bytes", "resource_scope_note")
            inputs = {k: endpoint[k] for k in keys}
            if inputs != {k: base[k] for k in keys}:
                raise ValueError("compared mathematical inputs differ")
            # Validate individual intervals before combining prepare and cold:
            # an invalid negative interval must not cancel a positive one.
            if set(endpoint["seconds"]) != {*phases, "prepare", "complete"} or any(
                type(t) not in (int, float) or not np.isfinite(t) or t <= 0
                for t in endpoint["seconds"].values()
            ):
                raise ValueError("invalid individual timing interval")
            for phase, workload in phases.items():
                found, expected = endpoint["results"][phase], base["results"][phase]
                for quantity, tolerance in (("energy", 3e-10), ("forces", 3e-9)):
                    got = np.asarray([r[quantity] for r in found])
                    want = np.asarray([r[quantity] for r in expected])
                    if got.shape != want.shape:
                        raise ValueError("endpoint result dimensions differ")
                    errors[f"{case}/{label}/{sample}/{phase}/{quantity}"] = block_error(
                        got, want, atol=tolerance, rtol=0
                    )
                if any(
                    not np.isfinite([r["energy_change"], r["density_rms"]]).all()
                    or type(r["iterations"]) is not int
                    or r["iterations"] < 1
                    for r in found
                ):
                    raise ValueError("invalid final residual/count diagnostics")
                seconds = endpoint["seconds"][phase]
                if phase == "cold":
                    seconds += endpoint["seconds"]["prepare"]
                record = {
                    "selection": label,
                    "seconds": seconds,
                    "inputs_hash": canonical_hash(inputs),
                    "workload": workload,
                    "synchronized": True,
                    "case": case,
                    "sample": sample,
                    "phase": phase,
                }
                case_timings.append(record)
                # v1's cross-method envelope calls restoration a geometry
                # change. Keep its explicit phase and the unmodified shared
                # assessment so this never merges the actual timing gates.
                timings.append(
                    {
                        **record,
                        "workload": "changed-geometry"
                        if phase == "restored"
                        else workload,
                    }
                )
            if domain == "df":
                # Value-path regressions must not be hidden by faster force
                # finalization. Retain and independently gate the fresh public
                # energy-only calls, including every residual and iteration.
                value = endpoint["energy_only"]
                if value["properties"] != ["energy"]:
                    raise ValueError("invalid energy-only properties")
                found, expected = value["results"], base["energy_only"]["results"]
                got = np.asarray([r["energy"] for r in found])
                want = np.asarray([r["energy"] for r in expected])
                if got.shape != want.shape or not got.size:
                    raise ValueError("energy-only result dimensions differ")
                errors[f"{case}/{label}/{sample}/energy-only/energy"] = block_error(
                    got, want, atol=3e-10, rtol=0
                )
                if any(
                    not np.isfinite([r["energy_change"], r["density_rms"]]).all()
                    or type(r["iterations"]) is not int
                    or r["iterations"] < 1
                    for r in found
                ):
                    raise ValueError("invalid energy-only residual/count diagnostics")
                record = {
                    "selection": label,
                    "seconds": value["seconds"],
                    "inputs_hash": canonical_hash(
                        {**inputs, "properties": value["properties"]}
                    ),
                    "workload": "energy-only-singlepoints",
                    "synchronized": True,
                    "case": case,
                    "sample": sample,
                    "phase": "energy-only",
                }
                case_timings.append(record)
                # The shared evidence envelope calls each fresh public call a
                # cold start; the explicit phase and assessment retain its
                # separate identity and never pool it with prepared forces.
                timings.append({**record, "workload": "cold-start"})
        assessment = assess_comparison(case_timings)
        if "workloads" not in assessment:
            raise ValueError(f"shared timing validation failed: {assessment}")
        ratios = {
            k: 1 - v["relative_improvement"] for k, v in assessment["workloads"].items()
        }
        rows.append(
            {
                "case": case,
                "phase_ratios": ratios,
                "nonregression_passed": all(v <= 1.02 for v in ratios.values()),
                "shared_comparison": assessment,
            }
        )
    if not all(e["passed"] for e in errors.values()) or not all(
        r["nonregression_passed"] for r in rows
    ):
        raise ValueError("raw numerical or nonregression gate failed")
    # A modified summary cannot override the recomputed raw worker evidence.
    if original.get("passed") is not True or original.get("endpoint_ceiling") != 1.02:
        raise ValueError("stored comparison is not the accepted unchanged gate")
    return compact, records, errors, timings, rows


def validate_resources(resources, baseline, candidate, *, domain="one-electron"):
    """Bind object measurements to the exact worker source and build contract.

    An exact-source kernel reconstruction is explicit when the historical linked
    candidate library was replaced by the final retirement build. It is resource
    evidence, not a claim that the reconstructed object was the timed binary.
    """
    expected = {
        "candidate-pair": candidate,
        "baseline-pair": baseline,
        "baseline-reference": baseline,
    }
    if domain == "df":
        expected = {
            "baseline-rhf": baseline,
            "candidate-rhf": candidate,
            "baseline-weighted": baseline,
            "candidate-weighted": candidate,
        }
    elif domain != "one-electron":
        raise ValueError("unknown resource domain")
    if set(resources) != set(expected):
        raise ValueError("resource inventory differs")
    for name, worker in expected.items():
        row = resources[name]
        provenance = row.get("provenance", {})
        for field in (
            "revision",
            "native_source_identity",
            "library_sha256",
            "build_settings",
        ):
            if provenance.get(field) != worker[field]:
                raise ValueError(f"resource/worker provenance mismatch: {name}/{field}")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", row.get("object_sha256", ""))
            or type(row.get("object_bytes")) is not int
            or row["object_bytes"] <= 0
            or row.get("measurement")
            not in ("original-object", "exact-source-kernel-reconstruction")
            or not row.get("compiler")
            or not row.get("resources")
        ):
            raise ValueError(f"incomplete native resource measurement: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--candidate-ref")
    args = parser.parse_args()
    compact, records, errors, timings, rows = compact_comparison(args.comparison)
    candidate = next(run for label, _, run in records if label == "candidate")
    baseline = records[0][2]
    domain = candidate.get("domain", "one-electron")
    stage = args.comparison / "publication"
    stage.mkdir(exist_ok=True)
    write(stage / "samples.json", compact)
    summary = {
        "cases": len(rows),
        "samples_per_selection": len(records) // 2,
        "energy_atol": 3e-10,
        "force_atol": 3e-9,
        "endpoint_ceiling": 1.02,
        "nonregression_passed": True,
        "speedup_claim": False,
        "baseline_revision": baseline["revision"],
        "candidate_revision": candidate["revision"],
        "process_scope": candidate.get("process_scope", "inventory"),
        "endpoints": rows,
    }
    write(stage / "summary.json", summary)
    resources = (
        json.loads(args.resources.read_text())
        if args.resources
        else {
            name: json.loads(
                (ROOT / ".artifacts" / f"231-{name}-resources.json").read_text()
            )
            for name in ("candidate-pair", "baseline-pair", "baseline-reference")
        }
    )
    validate_resources(resources, baseline, candidate, domain=domain)
    write(stage / "resources.json", resources)
    record = new_evidence(
        tier="endpoint",
        subject=f"{domain} CUDA ownership retirement",
        inputs_hash=canonical_hash([r["inputs_hash"] for r in timings]),
    )
    record.update(
        revision=candidate["revision"],
        backend_selected="cuda",
        device=candidate["gpu"],
        toolchain={k: candidate[k] for k in ("python", "numpy", "build_settings")},
        hardware=outcome(
            "pass", "finite Slurm allocation with preserved device visibility"
        ),
        settings={
            "device": "cuda",
            "fast_compile": False,
            "baseline_revision": baseline["revision"],
            "structural_retirement_nonregression": summary,
            "source_clean": True,
            "slurm_job_id": candidate["slurm_job_id"],
            "cuda_visible_devices": candidate["cuda_visible_devices"],
        },
        block_errors=errors,
        timings=timings,
    )
    record["hashes"]["source"] = candidate["native_source_identity"]
    record["hashes"]["schedule"] = canonical_hash(
        {"mapping": candidate["mapping"], "build": candidate["build_settings"]}
    )
    for key in ("equation", "ir"):
        record["hash_reasons"][key] = (
            "The full HF endpoint has no single scalar IR artifact; exact clean source and generated policy identities bind all contributing equations."
        )
    record["memory"]["reason"] = (
        "Per-case shared plans, scoped observations, and explicit resource-scope limitations are retained losslessly in samples.json. Cases exceeding the HF inventory have no total-budget guarantee. Native kernel resources are in resources.json."
    )
    record["compilation"]["reason"] = (
        "Original full build wall time was not captured; Release/FAST_COMPILE=OFF provenance and native kernel resources are retained."
    )
    record["solver_trace_reason"] = (
        "Every final residual and iteration count is retained in samples.json; the public endpoint does not expose per-iteration histories."
    )
    for name in ("representation", "source", "compilation", "numerical", "endpoint"):
        record["stages"][name] = outcome(
            "pass",
            "clean optimized worker comparison; see retained samples and summary",
        )
    record["performance"] = outcome(
        "not-run",
        "No significant-speedup promotion is claimed; every workload passed the separate fixed 2% structural-retirement nonregression ceiling.",
    )
    record["stages"]["production"] = outcome(
        "not-run",
        "This shared stage requires significant speedup; source retirement is governed by #231's explicitly retained nonregression decision.",
    )
    write_evidence(stage / "evidence.json", record)
    command = [
        "srun",
        "--partition=main",
        "--gres=gpu:5090:1",
        "--nodes=1",
        "--ntasks=1",
        "--time=02:00:00" if domain == "df" else "--time=01:00:00",
        "python",
        "tools/benchmark_cuda_ownership.py",
        "compare",
        "--domain",
        domain,
        "--baseline-root",
        "<baseline-checkout>",
        "--baseline-build",
        "<baseline-optimized-build>",
        "--candidate-root",
        "<candidate-checkout>",
        "--candidate-build",
        "<candidate-optimized-build>",
        "--samples",
        str(len(records) // 2),
        "--output",
        ".artifacts/ownership-reproduction",
    ]
    # Historical measured checkouts predate this option; their omitted field
    # already means inventory scope and must keep the original replay argv.
    if "process_scope" in candidate:
        command += ["--process-scope", candidate["process_scope"]]
    specification = {
        "source": {"revision": candidate["revision"], "dirty": False},
        "reproduction": {
            "command": command,
            "baseline_source": {
                "repository": "https://github.com/njzjz-bot/vibeqc",
                "ref": args.baseline_ref or "refs/heads/evidence/issue-231-baseline",
                "revision": baseline["revision"],
            },
            "candidate_source": {
                "repository": "https://github.com/njzjz-bot/vibeqc",
                "ref": args.candidate_ref
                or "refs/heads/codex/issue-231-cuda-ownership",
                "revision": candidate["revision"],
            },
            "note": "Check out the exact recorded baseline/candidate revisions and use Release, CUDA 12.9.1, sm_120, FAST_COMPILE=OFF; set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1.",
        },
        "decision": {
            "status": "accepted",
            "scope": "numerical",
            "reason": "All unchanged numerical and separate per-workload 2% nonregression gates passed. Accept ownership retirement without a significant-speedup claim.",
        },
        "files": [
            {"path": f"{name}.json", "role": role}
            for name, role in (
                ("evidence", "evidence"),
                ("samples", "samples"),
                ("summary", "summary"),
                ("resources", "summary"),
            )
        ],
        "archives": [],
    }
    print(publish(stage, specification, args.destination))


if __name__ == "__main__":
    main()

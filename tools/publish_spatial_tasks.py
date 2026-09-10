"""Publish bounded spatial candidates with explicit arithmetic/approximation scopes."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.common.evidence import (
    canonical_hash,
    new_evidence,
    outcome,
    write_evidence,
)
from vibeqc_compiler.common.resources import ResourcePlan

from tools.vibeqc_validation.publication import publish


def write(path, value):
    """Retain full float precision and deterministic finite JSON."""
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )


def validate_run(run, *, dense_only=False):
    """Reject partial inventories, stale resource accounting and failed numerical gates."""
    if (
        run["schema"] != "vibeqc.spatial-task-benchmark.v1"
        or run["dirty"] is not False
        or not re.fullmatch(r"[0-9a-f]{40}", run["revision"])
    ):
        raise ValueError("invalid clean worker provenance")
    if run["backend"] not in ("cpu", "cuda"):
        raise ValueError("unsupported worker backend")
    for key in ("source_identity", "library_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", run.get(key, "")):
            raise ValueError("missing source/library provenance")
    if run["backend"] == "cuda" and (
        not run["slurm_job_id"] or not run["gpu"] or not run["artifact"]
    ):
        raise ValueError("missing scheduled CUDA provenance")
    modes = ("dense",) if dense_only else ("dense", "local_off", "local_screened")
    expected = [
        f"{n}/{r}/tile{t}/{m}"
        for n in (4, 16)
        for r in ("cartesian", "spherical")
        for t in (16, 64)
        for m in modes
    ]
    if [r["case"] for r in run["cases"]] != expected:
        raise ValueError("worker inventory differs")
    for row in run["cases"]:
        atom, representation, tile, mode = row["case"].split("/")
        if (
            row["mode"] != mode
            or row["natom"] != int(atom)
            or row["nao"] != int(atom) * (14 if representation == "cartesian" else 11)
            or row["npoint"] != 32 * int(atom)
            or row["tile_points"] != int(tile.removeprefix("tile"))
            or row["tile_plan"]["tile_points"] != row["tile_points"]
            or row["tile_plan"]["nao"] != row["nao"]
            or row["tile_plan"]["backend"] != run["backend"]
        ):
            raise ValueError("case label and execution metadata differ")
        if [r["sample"] for r in row["samples"]] != list(range(5)):
            raise ValueError("five complete samples per case required")
        if row["mode"] != "dense":
            plan = ResourcePlan.from_dict(row["resource_plan"]).require_feasible()
            if (
                plan.budget.host_bytes != row["budgets"]["host_bytes"]
                or plan.budget.device_bytes != row["budgets"]["device_bytes"]
            ):
                raise ValueError("resource plan/worker budget mismatch")
            # A self-consistent plan can still belong to a different workload.
            # Bind its declared topology and native capacities to this worker.
            request = next(
                (r for r in plan.requests if r.name == "spatial_execution"), None
            )
            if request is None or len(request.candidates) != 1:
                raise ValueError(
                    "resource plan requires one spatial execution candidate"
                )
            topology = json.loads(request.identity.topology)
            if (
                topology
                != {
                    "active": row["tile_plan"]["active_ao_capacity"],
                    "ao": row["nao"],
                    "jets": 4,
                    "points": row["tile_points"],
                }
                or request.identity.backend != run["backend"]
            ):
                raise ValueError("resource plan/worker topology mismatch")
            estimates = {e.name: e.bytes for e in request.candidates[0].estimates}
            if run["backend"] == "cuda" and (
                estimates["local_device_arena"] != row["tile_plan"]["allocation_bytes"]
                or estimates["cublas_allowance"] != row["tile_plan"]["provider_bytes"]
            ):
                raise ValueError("resource plan/native capacity mismatch")
        for sample in row["samples"]:
            errors = sample["errors"]
            required = {"rho", "gradient", "sigma", "tau"}
            if run["backend"] == "cpu":
                required |= {"energy", "potential"}
            elif row["mode"] != "dense":
                required.add("scatter")
            shapes = {
                "rho": [2, row["npoint"]],
                "gradient": [2, row["npoint"], 3],
                "sigma": [3, row["npoint"]],
                "tau": [2, row["npoint"]],
                "energy": [1],
                "potential": [2, row["nao"], row["nao"]],
                "scatter": [2, row["nao"], row["nao"]],
            }
            if set(errors) != required or not all(
                e["passed"] is True
                and e["atol"] == 1e-11
                and e["rtol"] == 1e-10
                and 0 <= e["max_scaled_error"] <= 1
                and e["shape"] == shapes[key]
                and all(
                    type(e[k]) in (int, float) and math.isfinite(e[k]) and e[k] >= 0
                    for k in ("max_absolute_error", "max_scaled_error", "rms_error")
                )
                for key, e in errors.items()
            ):
                raise ValueError("missing or failed unchanged arithmetic gates")
            timings = {"construction_seconds", "features_seconds"}
            if run["backend"] == "cpu":
                timings.add("xc_seconds")
            elif mode != "dense":
                timings |= {"device_consumer_scatter_seconds", "scatter_calls_seconds"}
            if {k for k in sample if k.endswith("_seconds")} != timings or any(
                type(v) not in (int, float) or not 0 < v < float("inf")
                for k, v in sample.items()
                if k.endswith("_seconds")
            ):
                raise ValueError("invalid execution timing")
            if run["backend"] == "cuda":
                metrics = sample["native_metrics"]
                if (
                    any(
                        type(metrics[k]) is not int or metrics[k] < 0
                        for k in ("owned_device_bytes", "provider_retained_bytes")
                    )
                    or metrics["owned_device_bytes"]
                    != row["tile_plan"]["allocation_bytes"]
                    or metrics["provider_retained_bytes"]
                    > row["tile_plan"]["provider_bytes"]
                ):
                    raise ValueError("native resource observation exceeds plan")
    return run


def summarize(run):
    """Summarize complete measurements without promoting a different scientific mask."""
    rows = []
    for row in run["cases"]:
        timing = {
            k: statistics.median(s[k] for s in row["samples"])
            for k in row["samples"][0]
            if k.endswith("_seconds")
        }
        summary = {
            k: row[k]
            for k in (
                "case",
                "mode",
                "nao",
                "npoint",
                "budgets",
                "approximation_difference",
            )
        }
        summary["median_seconds"] = timing
        summary["median_construction_plus_features_seconds"] = statistics.median(
            s["construction_seconds"] + s["features_seconds"] for s in row["samples"]
        )
        if run["backend"] == "cpu":
            summary["median_construction_plus_xc_seconds"] = statistics.median(
                s["construction_seconds"] + s["xc_seconds"] for s in row["samples"]
            )
        if row["mode"] != "dense":
            summary["planned_peak_bytes"] = row["resource_plan"]["peak_bytes"]
            summary["metadata_bytes"] = row["metadata_bytes"]
            summary["active_aos"] = row["active_aos"]
        rows.append(summary)
    return {
        "schema": "vibeqc.spatial-task-summary.v1",
        "backend": run["backend"],
        "endpoints": rows,
        "production_promoted": False,
    }


def dense_comparison(directory):
    """Descriptive historical dense sweep: three ABBA process blocks, five samples each.

    This grouped design is retained as measured; it is not relabeled as fifteen
    independent interleaved process trials or a shared significant-speedup gate.
    """
    groups = {
        side: [
            validate_run(
                json.loads(
                    (directory / f"234-optimized-dense-{side}-{i}.json").read_text()
                ),
                dense_only=True,
            )
            for i in range(3)
        ]
        for side in ("baseline", "candidate")
    }
    for group in groups.values():
        reference = group[0]
        for run in group:
            if run["backend"] != "cuda" or any(
                run[k] != reference[k]
                for k in (
                    "revision",
                    "source_identity",
                    "library_sha256",
                    "artifact",
                    "gpu",
                )
            ):
                raise ValueError("dense comparison source/build changes within a side")
    rows = []
    for index, base in enumerate(groups["baseline"][0]["cases"]):
        inputs = base["inputs"]
        samples = {
            side: [s for run in group for s in run["cases"][index]["samples"]]
            for side, group in groups.items()
        }
        for group in groups.values():
            for run in group:
                if run["cases"][index]["inputs"] != inputs:
                    raise ValueError("dense comparison mathematical inputs differ")
        medians = {
            side: {
                k: statistics.median(s[k] for s in values)
                for k in ("features_seconds", "construction_seconds")
            }
            for side, values in samples.items()
        }
        rows.append(
            {
                "case": base["case"],
                "medians": medians,
                "ratios": {
                    k: medians["candidate"][k] / medians["baseline"][k]
                    for k in medians["baseline"]
                },
            }
        )
    return {
        "process_order": [
            "baseline",
            "candidate",
            "candidate",
            "baseline",
            "baseline",
            "candidate",
        ],
        "runs": groups,
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    dense_samples, dense_rows = dense_comparison(args.artifacts)
    for backend in ("cpu", "cuda"):
        run = validate_run(
            json.loads((args.artifacts / f"234-{backend}-final.json").read_text())
        )
        stage = args.artifacts / f"spatial-publication-{backend}"
        stage.mkdir(exist_ok=True)
        write(stage / "samples.json", run)
        summary = summarize(run)
        if backend == "cuda":
            summary["historical_dense_comparison"] = dense_rows
            write(stage / "dense-comparison-samples.json", dense_samples)
        write(stage / "summary.json", summary)
        record = new_evidence(
            tier="endpoint",
            subject=f"{backend} bounded spatial AO/features"
            + (
                " and fixed-density PBE E/V"
                if backend == "cpu"
                else " and diagnostic scatter"
            ),
            inputs_hash=canonical_hash([r["inputs"] for r in run["cases"]]),
        )
        record.update(
            revision=run["revision"],
            backend_selected=backend,
            device=run["gpu"] or "CPU; processor model was not captured by the worker",
            toolchain={
                "python": run["python"],
                "numpy": run["numpy"],
                "native_library_sha256": run["library_sha256"],
                "cuda_artifact": run["artifact"],
            },
            hardware=outcome(
                "pass",
                "native worker executed successfully"
                if backend == "cpu"
                else "finite Slurm allocation, preserving assigned visibility",
            ),
        )
        record["hashes"]["source"] = run["source_identity"]
        record["hashes"]["equation"] = canonical_hash(
            [r["inputs"]["functional"] for r in run["cases"]]
        )
        record["hashes"]["schedule"] = canonical_hash(
            [
                {
                    "case": r["case"],
                    "tile": r["tile_plan"],
                    "mask": r.get("mask_identity"),
                }
                for r in run["cases"]
            ]
        )
        record["hash_reasons"]["ir"] = (
            "Complete AO/feature/CPU-XC endpoint has several existing compiler and native owners, with no single scalar IR artifact."
        )
        record["settings"] = {
            "fast_compile": False if backend == "cuda" else None,
            "slurm_job_id": run["slurm_job_id"],
            "cuda_visible_devices": run["cuda_visible_devices"],
            "approximation_scope": "Each screened arithmetic gate uses the identical fixed AO mask; differences from unscreened collocation are separate diagnostics, not error bounds.",
            "execution_scope": run["timing_scope"],
            "selection": "explicit local candidate; dense baseline retained",
        }
        for row in run["cases"]:
            for sample in row["samples"]:
                prefix = f"{row['case']}/{sample['sample']}"
                record["block_errors"].update(
                    {f"{prefix}/{k}": e for k, e in sample["errors"].items()}
                )
                record["timings"].append(
                    {
                        "selection": "candidate",
                        "seconds": sample["construction_seconds"]
                        + sample.get("xc_seconds", sample["features_seconds"]),
                        "inputs_hash": canonical_hash(
                            {"input": row["inputs"], "mask": row.get("mask_identity")}
                        ),
                        "workload": "cold-start",
                        "synchronized": True,
                        "case": row["case"],
                        "sample": sample["sample"],
                    }
                )
        record["memory"]["reason"] = (
            "Per-case shared plans include validated peak/resident accounting; native CUDA observations and explicit scope exclusions are retained in samples.json. Full global D/V remain O(NAO^2); no whole-process peak claim."
        )
        record["solver_trace_reason"] = (
            "Fixed-density primitives and XC integration have no SCF/Krylov iterations."
        )
        if run["artifact"]:
            record["compilation"] = {
                "seconds": run["artifact"]["compile_seconds"],
                "reason": "Cold runtime compilation time from the content-verified cache artifact.",
            }
        else:
            record["compilation"]["reason"] = (
                "The matching CPU native library was built separately; full CPU build wall time was not recorded."
            )
        for name in (
            "representation",
            "source",
            "compilation",
            "numerical",
            "endpoint",
        ):
            record["stages"][name] = outcome(
                "pass",
                "bounded worker passes unchanged arithmetic gates; endpoint scope is explicit",
            )
        record["performance"] = outcome(
            "not-run",
            "No significant-speedup or production promotion; complete construction costs can outweigh screened execution savings.",
        )
        record["stages"]["production"] = outcome(
            "not-run",
            "Local execution remains an explicit candidate for the existing selection owner.",
        )
        write_evidence(stage / "evidence.json", record)
        files = [
            {"path": f"{name}.json", "role": role}
            for name, role in (
                ("evidence", "evidence"),
                ("samples", "samples"),
                ("summary", "summary"),
            )
        ]
        if backend == "cuda":
            files.append({"path": "dense-comparison-samples.json", "role": "samples"})
        command = [
            "python",
            "tools/benchmark_spatial_tasks.py",
            "--root",
            "<measured-checkout>",
            "--library",
            "<matching-cpu-library>",
            "--cache",
            ".artifacts/spatial-cuda-cache",
            "--backend",
            backend,
            "--samples",
            "5",
            "--output",
            f".artifacts/234-{backend}-final.json",
        ]
        if backend == "cuda":
            command = [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:10:00",
                *command,
            ]
        specification = {
            "source": {"revision": run["revision"], "dirty": False},
            "reproduction": {
                "command": command,
                "source_repository": "https://github.com/njzjz-bot/vibeqc",
                "source_ref": "refs/heads/evidence/issue-234-measured",
                "note": "Fetch the durable measured-source branch and exact recorded revision; use OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Match the recorded native build identity. See README for historical dense sweep and its separate source ref.",
            },
            "decision": {
                "status": "accepted",
                "scope": "numerical",
                "reason": "All identical-mask arithmetic gates and resource checks passed; approximation differences and complete timing remain explicit. No local production promotion.",
            },
            "files": files,
            "archives": [],
        }
        print(publish(stage, specification, args.destination / backend))


if __name__ == "__main__":
    main()

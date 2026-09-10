"""Validate and publish compact CPU XC endpoint evidence via the shared policy."""

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
    block_error,
    new_evidence,
    outcome,
    write_evidence,
)
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.resources import ResourcePlan

from tools.vibeqc_validation.publication import publish

CASES = ("h2", "f_cartesian", "f_spherical", "separated_f")
FUNCTIONALS = ("LDA_XC_PW", "PBE")
OBSERVABLES = ("energy", "potential", "response", "geometry")


def checked_error(error, *, atol=1e-11, rtol=1e-10):
    """Require the unchanged quantitative gate, not a standalone success flag."""
    if (
        error["passed"] is not True
        or error["atol"] != atol
        or error["rtol"] != rtol
        or not 0 <= error["max_scaled_error"] <= 1
        or not 0 <= error["rms_error"] <= error["max_absolute_error"]
        or not error["shape"]
    ):
        raise ValueError("failed or altered numerical gate")


def validate_run(run):
    """Bind complete timing, source, derivative and resource inventories."""
    canonical_hash(run)  # Reject nonfinite values anywhere in retained evidence.
    if (
        run["schema"] != "vibeqc.xc-contraction-benchmark.v1"
        or run["dirty"] is not False
        or not re.fullmatch(r"[0-9a-f]{40}", run["revision"])
    ):
        raise ValueError("invalid clean source provenance")
    for key in ("source_identity", "library_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", run[key]):
            raise ValueError("missing source/library identity")
    if (
        not run["hardware"]
        or not run["toolchain"]
        or run["memory"]["process_lifetime_rss_high_water_bytes"] <= 0
    ):
        raise ValueError("missing observed hardware/memory evidence")
    expected_programs = {
        name + "/" + obs for name in FUNCTIONALS for obs in OBSERVABLES
    }
    if set(run["programs"]) != expected_programs:
        raise ValueError("incomplete native program inventory")
    for key, program in run["programs"].items():
        meta, artifact = program["metadata"], program["artifact"]
        if (
            canonical_hash(artifact["identity"]) != artifact["key"]
            or artifact["identity"]["source"] != meta["source_sha256"]
            or program["source_bytes"] != meta["source_bytes"]
            or program["source_bytes"] <= 0
            or artifact["compile_seconds"] <= 0
            or program["construction_seconds"] < artifact["compile_seconds"]
        ):
            raise ValueError("native source/build provenance mismatch")
        if (
            meta["contract"]["observable"] != key.split("/")[1]
            or not meta["host_source_hashes"]
            or not meta["layouts"]
        ):
            raise ValueError("native contract/source owner inventory differs")
    expected = [
        f"{case}/{name}/{obs}/{mode}/{tile}"
        for case in CASES
        for name in FUNCTIONALS
        for obs in OBSERVABLES
        for mode in ("dense", "local")
        for tile in (7, 19)
    ]
    if [r["case"] for r in run["cases"]] != expected:
        raise ValueError("incomplete endpoint case inventory")
    for row in run["cases"]:
        case, name, obs, mode, tile = row["case"].split("/")
        if (
            row["fixture"] != case
            or row["program"] != name + "/" + obs
            or row["tile_points"] != int(tile)
            or row["local"] != (mode == "local")
        ):
            raise ValueError("endpoint label and metadata differ")
        plan = ResourcePlan.from_dict(row["resource_plan"]).require_feasible()
        request = next((r for r in plan.requests if r.name == "xc_contractions"), None)
        if request is None or len(request.candidates) != 1:
            raise ValueError("missing XC numeric resource candidate")
        topology = json.loads(request.identity.topology)
        if (
            plan.budget.host_bytes != row["budget_bytes"]
            or topology["contract"]
            != canonical_hash(run["programs"][row["program"]]["metadata"]["contract"])
            or topology["grid"] != row["grid_identity"]
            or topology["mask"] != row["mask_identity"]
            or topology["tile_points"] != row["tile_points"]
        ):
            raise ValueError("XC resource plan belongs to different inputs")
        if (
            bool(row["mask_identity"]) != row["local"]
            or row["traced_execute_peak_bytes"] <= 0
        ):
            raise ValueError("missing mask/memory observation")
        if (
            row["local"]
            and case == "separated_f"
            and not all(0 < n < row["nao"] for n in row["active_ao_sizes"])
        ):
            raise ValueError(
                "separated f case must exercise nonempty strict AO subsets"
            )
        if [s["sample"] for s in row["samples"]] != list(range(5)):
            raise ValueError("five ordered samples required")
        required_errors = {"energy", "electrons"}
        required_errors |= (
            {"geometry/centers", "geometry/points", "geometry/weights"}
            if obs == "geometry"
            else {obs}
            if obs in ("potential", "response")
            else set()
        )
        if mode == "dense":
            required_errors.add("external_energy")
            if obs == "potential":
                required_errors.add("external_potential")
        for sample in row["samples"]:
            if set(sample["errors"]) != required_errors:
                raise ValueError("incomplete independent/arithmetic gate inventory")
            for error in sample["errors"].values():
                checked_error(error)
            for key in ("construction_seconds", "execute_seconds"):
                if not math.isfinite(sample[key]) or sample[key] <= 0:
                    raise ValueError("invalid complete endpoint timing")
            stats = sample["statistics"]
            if (
                stats["planned_host_peak_bytes"] != plan.peak_bytes["host"]
                or stats["scalar_calls"] > stats["tiles"]
                or stats["scalar_calls"] <= 0
            ):
                raise ValueError("resource/execution accounting mismatch")
            calls = stats["scalar_calls"]
            density_products = calls * (4 if obs == "response" else 2)
            geometry_products = (
                2 * calls * (1 if name.startswith("LDA") else 4)
                if obs == "geometry"
                else 0
            )
            if (
                stats["density_matrix_products"] != density_products
                or stats["geometry_matrix_products"] != geometry_products
                or stats["total_matrix_products"]
                != density_products
                + geometry_products
                + stats["matrix_assembly_products"]
            ):
                raise ValueError("incomplete matrix-product accounting")
            if (
                stats["point_coefficient_calls"] != (0 if obs == "energy" else calls)
                or stats["ao_pullback_calls"] != (2 * calls if obs == "geometry" else 0)
                or stats["matrix_assembly_products"]
                != (
                    2 * calls * (1 if name.startswith("LDA") else 2)
                    if obs in ("potential", "response")
                    else 0
                )
            ):
                raise ValueError("native/library call accounting mismatch")
    expected_derivatives = {case + "/" + name for case in CASES for name in FUNCTIONALS}
    if set(run["independent_derivatives"]) != expected_derivatives:
        raise ValueError("incomplete independent derivative inventory")
    for value in run["independent_derivatives"].values():
        projections = value["projections"]
        if {(p["axis"], p["step"]) for p in projections} != {
            (axis, step)
            for axis in (
                "density",
                "response",
                "centers",
                "points",
                "weights",
                "combined",
            )
            for step in (1e-3, 3e-4, 1e-4)
        } or len(projections) != 18:
            raise ValueError("independent derivative sources/steps differ")
        for p in projections:
            finite = (p["plus"] - p["minus"]) / (2 * p["step"])
            expected = block_error([p["analytic"]], [finite], atol=3e-8, rtol=0)
            if p["finite_difference"] != finite or p["error"] != expected:
                raise ValueError("independent derivative raw samples differ")
            checked_error(p["error"], atol=3e-8, rtol=0)
        checked_error(value["svec_transpose"], atol=1e-12, rtol=1e-10)
    return run


def summarize(run):
    """Reconstruct medians from retained samples without a speedup decision."""
    validate_run(run)
    return {
        "schema": "vibeqc.xc-contraction-summary.v1",
        "revision": run["revision"],
        "decision": "numerical acceptance; explicit experimental CPU candidate; no performance promotion",
        "cases": [
            {
                "case": r["case"],
                "construction_median_seconds": statistics.median(
                    s["construction_seconds"] for s in r["samples"]
                ),
                "execute_median_seconds": statistics.median(
                    s["execute_seconds"] for s in r["samples"]
                ),
                "numeric_plan_peak_bytes": r["resource_plan"]["peak_bytes"]["host"],
                "traced_execute_peak_bytes": r["traced_execute_peak_bytes"],
                "active_ao_min": min(r["active_ao_sizes"]),
                "active_ao_max": max(r["active_ao_sizes"]),
            }
            for r in run["cases"]
        ],
        "memory": run["memory"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    run = validate_run(json.loads(args.run.read_text()))
    args.stage.mkdir(parents=True, exist_ok=False)
    for name, data in (("samples", run), ("summary", summarize(run))):
        (args.stage / f"{name}.json").write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n"
        )
    evidence = new_evidence(
        tier="endpoint",
        subject="CPU fixed-density XC E/V, response actions and explicit geometry partials",
        inputs_hash=canonical_hash([r["inputs_hash"] for r in run["cases"]]),
    )
    evidence.update(
        revision=run["revision"],
        backend_selected="cpu",
        device=run["hardware"],
        toolchain=run["toolchain"],
        hardware=outcome("pass", "CPU worker executed all declared endpoints"),
    )
    evidence["hashes"] = {
        "source": run["source_identity"],
        "equation": canonical_hash(
            [p["metadata"]["contract"] for p in run["programs"].values()]
        ),
        "ir": canonical_hash(
            [p["metadata"]["layouts"] for p in run["programs"].values()]
        ),
        "schedule": canonical_hash([r["resource_plan"] for r in run["cases"]]),
    }
    evidence["settings"] = {
        "scope": "fixed D and mask; independent AO-center/point/weight gradients; no stationary molecular force, GPU XC or complete KS/CPKS capability",
        "selection": "explicit experimental CPU native candidate",
    }
    evidence["memory"] = {
        "allocated_bytes": None,
        "peak_bytes": run["memory"]["process_lifetime_rss_high_water_bytes"],
        "reason": run["memory"]["scope"],
    }
    evidence["compilation"] = {
        "seconds": sum(
            p["artifact"]["compile_seconds"] for p in run["programs"].values()
        ),
        "reason": "Sum of eight cold generated point-program compilations; full native AO library was built separately.",
    }
    for name in ("representation", "source", "compilation", "numerical", "endpoint"):
        evidence["stages"][name] = outcome(
            "pass",
            "complete scoped CPU endpoints and independent derivative gates pass",
        )
    evidence["stages"]["production"] = outcome(
        "not-run",
        "explicit experimental candidate; complete owning method tasks control public capabilities",
    )
    evidence["performance"] = outcome(
        "not-run",
        "timings characterize endpoints; no significant historical comparison or production promotion",
    )
    evidence["solver_trace_reason"] = (
        "fixed-density actions/partials have no iterative solver history"
    )
    for row in run["cases"]:
        for sample in row["samples"]:
            prefix = row["case"] + "/" + str(sample["sample"])
            evidence["block_errors"].update(
                {prefix + "/" + k: e for k, e in sample["errors"].items()}
            )
            evidence["timings"].append(
                {
                    "selection": "candidate",
                    "seconds": sample["construction_seconds"]
                    + sample["execute_seconds"],
                    "workload": "cold-start",
                    "inputs_hash": row["inputs_hash"],
                    "case": row["case"],
                    "sample": sample["sample"],
                }
            )
    for case, value in run["independent_derivatives"].items():
        evidence["block_errors"].update(
            {
                case + "/" + p["axis"] + "/" + str(p["step"]): p["error"]
                for p in value["projections"]
            }
        )
        evidence["block_errors"][case + "/svec_transpose"] = value["svec_transpose"]
    write_evidence(args.stage / "evidence.json", evidence)
    specification = {
        "source": {"revision": run["revision"], "dirty": False},
        "reproduction": {
            "command": [
                "python",
                "tools/benchmark_xc_contractions.py",
                "--library",
                "<matching-cpu-library>",
                "--cache",
                ".artifacts/xc-measured-cache",
                "--output",
                ".artifacts/xc-measured.json",
            ],
            "source_repository": "https://github.com/njzjz-bot/vibeqc",
            "source_ref": "refs/heads/evidence/issue-236-measured",
            "note": "Fetch exact measured revision; fresh cache; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; PySCF 2.14.0/Libxc 7.0.0.",
        },
        "decision": {
            "status": "accepted",
            "scope": "numerical",
            "reason": "Independent E/V and derivative projections, native identical-mask arithmetic and shared resource checks pass; performance remains experimental.",
        },
        "files": [
            {
                "path": name + ".json",
                "role": name,
                "reason": "Complete quantitative per-case gates and five raw timing samples retain independent numerical evidence.",
            }
            for name in ("evidence", "samples", "summary")
        ],
        "archives": [],
    }
    print(publish(args.stage, specification, args.destination))


if __name__ == "__main__":
    main()

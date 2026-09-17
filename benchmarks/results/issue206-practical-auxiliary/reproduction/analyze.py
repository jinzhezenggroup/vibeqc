"""Audit all paired raw arrays without importing either GPU implementation.

Accept the original Slurm output directory or its lossless retained copy.
Statistics describe the observed ordinary solves; no branch is discarded and
no post-hoc stability threshold is used to assert performance acceptance.
"""

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

# This historical audit intentionally uses assertions for executable invariants.
# Refuse optimized execution even when imported, so -O can never certify data
# after stripping hash, convergence, numerical, or cardinality checks.
if not __debug__:
    raise RuntimeError("evidence audit requires assertions; do not use python -O")


def raw_bytes(path):
    """Restore exact original bytes before checking the controller's hashes."""
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def read(path):
    return json.loads(raw_bytes(path))


def flatten(value):
    """Flatten nested force arrays while preserving every system and component."""
    if isinstance(value, list):
        for child in value:
            yield from flatten(child)
    else:
        assert isinstance(value, (int, float)) and math.isfinite(value)
        yield value


def maximum_error(first, second):
    return max(abs(a - b) for a, b in zip(flatten(first), flatten(second), strict=True))


def sample_statistics(samples):
    times = [s["seconds"] for s in samples]
    assert len(times) == 7 and all(math.isfinite(t) and t > 0 for t in times)
    branches = Counter(
        tuple(c["iterations"] for c in s["convergence"]) for s in samples
    )
    center = median(times)
    return {
        "seconds": times,
        "median_seconds": center,
        "minimum_seconds": min(times),
        "maximum_seconds": max(times),
        "relative_mad": median(abs(t - center) for t in times) / center,
        "branches": [
            {"iterations": list(branch), "count": count}
            for branch, count in sorted(branches.items())
        ],
        "final_residuals": [
            [c["final_residuals"] for c in s["convergence"]] for s in samples
        ],
    }


def matched_subset(samples):
    """Rebuild the comparator's most-supported common branch from raw solves.

    Selection maximizes the smaller engine sample count, with a lexicographic
    tie break. The result is checked for provenance only; a one-sample stock
    subset remains insufficient for a fixed-work performance claim.
    """
    groups = {}
    for engine, values in samples.items():
        groups[engine] = {}
        for sample in values:
            branch = tuple(c["iterations"] for c in sample["convergence"])
            assert all(type(n) is int and n >= 0 for n in branch)
            groups[engine].setdefault(branch, []).append(sample["seconds"])
    native, stock = groups["vibeqc"], groups["gpu4pyscf"]
    shared = native.keys() & stock.keys()
    if not shared:
        return None
    branch = min(shared, key=lambda b: (-min(len(native[b]), len(stock[b])), b))
    native_median, stock_median = median(native[branch]), median(stock[branch])
    return {
        "iteration_branch": list(branch),
        "vibeqc_sample_count": len(native[branch]),
        "gpu4pyscf_sample_count": len(stock[branch]),
        "vibeqc_median_seconds": native_median,
        "gpu4pyscf_median_seconds": stock_median,
        "speedup": stock_median / native_median,
    }


def analyze(directory):
    """Recompute numerical verdicts and medians from all 28 measured solves."""
    manifest = read(directory / "manifest.json")
    endpoints = [
        c for c in manifest["checks"] if c["scope"] == "clean external endpoint"
    ]
    assert len(endpoints) == 2
    points = []
    for row in endpoints:
        path = directory / (row["name"] + ".json")
        assert hashlib.sha256(raw_bytes(path)).hexdigest() == row["result_sha256"]
        data = read(path)
        workload = data["workload"]
        batch = workload["batch_size"]
        forces = workload["properties"] != ["energy"]
        qualifier = read(directory / f"{workload['case']}-b{batch}-qualification.json")
        assert qualifier["contract_engine"] == "cuTENSOR"
        assert len(qualifier["results"]) == batch
        metrics = data["settings"]["density_fitting_metric_diagnostics"]
        assert len(metrics) == batch
        for native, stock in zip(metrics, qualifier["results"], strict=True):
            assert stock["matching_full_rank"]
            assert native["effective_rank"] == stock["gpu4pyscf_effective_rank"]
            assert native["effective_rank"] == stock["naux"]
            assert stock["naux"] != workload["ao_count"]
        assert (
            data["native_build"]["library_sha256"]
            == manifest["build"]["library_sha256"]
        )
        assert (
            data["native_build"]["probe"]["source_identity"]
            == manifest["build"]["native_source_identity"]
        )
        samples = {
            engine: data[engine]["warm_samples"] for engine in ("vibeqc", "gpu4pyscf")
        }
        assert all(len(s) == 7 for s in samples.values())
        sequence = sorted(
            (s["sequence_index"], engine)
            for engine, values in samples.items()
            for s in values
        )
        assert [i for i, _ in sequence] == list(range(14))
        assert [engine for _, engine in sequence] == data["settings"][
            "measurement_order"
        ]
        for values in samples.values():
            for sample in values:
                assert (
                    len(sample["convergence"])
                    == len(sample["energies_hartree"])
                    == batch
                )
                assert all(c["converged"] for c in sample["convergence"])
                if forces:
                    assert len(sample["forces_hartree_per_bohr"]) == batch
        gates = data["settings"]["gates"]
        pairs = []
        for i, (native, stock, recorded) in enumerate(
            zip(
                samples["vibeqc"],
                samples["gpu4pyscf"],
                data["accuracy"]["paired_warm_repeats"],
                strict=True,
            )
        ):
            energy_error = maximum_error(
                native["energies_hartree"], stock["energies_hartree"]
            )
            force_error = (
                maximum_error(
                    native["forces_hartree_per_bohr"], stock["forces_hartree_per_bohr"]
                )
                if forces
                else None
            )
            assert energy_error == recorded["maximum_energy_error_hartree"]
            assert force_error == recorded["maximum_force_error_hartree_per_bohr"]
            branch_matches = [c["iterations"] for c in native["convergence"]] == [
                c["iterations"] for c in stock["convergence"]
            ]
            assert branch_matches == recorded["iteration_branches_match"]
            pairs.append(
                {
                    "repeat": i,
                    "energy_error_hartree": energy_error,
                    "force_error_hartree_per_bohr": force_error,
                    "iteration_branches_match": branch_matches,
                    "numerically_passed": energy_error
                    <= gates["maximum_energy_error_hartree"]
                    and (
                        not forces
                        or force_error <= gates["maximum_force_error_hartree_per_bohr"]
                    ),
                }
            )
        passed = all(p["numerically_passed"] for p in pairs)
        assert passed == row["numerically_qualified"] == data["gate"]["passed"]
        stats = {
            engine: sample_statistics(values) for engine, values in samples.items()
        }
        matched = matched_subset(samples)
        assert matched == data["timing_summary"]["iteration_matched"], (
            f"{row['name']}: recorded matched subset differs from raw samples"
        )
        for engine in stats:
            assert (
                stats[engine]["median_seconds"] == data[engine]["warm_median_seconds"]
            )
        native_time, stock_time = (
            stats[e]["median_seconds"] for e in ("vibeqc", "gpu4pyscf")
        )
        points.append(
            {
                "name": row["name"],
                "ao_count": workload["ao_count"],
                "batch_size": batch,
                "observable": "energy_and_force" if forces else "energy",
                "numeric_pass": passed,
                "failed_repeats": [
                    p["repeat"] for p in pairs if not p["numerically_passed"]
                ],
                "gates": gates,
                "paired_accuracy": pairs,
                "maximum_energy_error_hartree": max(
                    p["energy_error_hartree"] for p in pairs
                ),
                "maximum_force_error_hartree_per_bohr": max(
                    p["force_error_hartree_per_bohr"] for p in pairs
                )
                if forces
                else None,
                "ordinary": stats,
                "ordinary_native_reduction_fraction": 1 - native_time / stock_time,
                "paired_matching_iteration_count": sum(
                    p["iteration_branches_match"] for p in pairs
                ),
                "comparator_matched_subset": matched,
                "metric_and_native_reserved_memory": metrics,
                "stock_versions": qualifier["versions"],
                "stock_factor_records": [
                    {
                        k: v
                        for k, v in s.items()
                        if k not in ("atoms", "orbital_basis", "auxiliary_basis")
                    }
                    for s in qualifier["results"]
                ],
                "cold_single_observations_seconds": {
                    e: data[e]["cold_seconds"] for e in stats
                },
            }
        )
    return {
        "schema": "vibeqc.issue206.practical-auxiliary.audit.v1",
        "slurm_job_id": manifest["slurm_job_id"],
        "measured_head": manifest["git_head"],
        "native_source_identity": manifest["build"]["native_source_identity"],
        "library_sha256": manifest["build"]["library_sha256"],
        "endpoint_count": len(points),
        "paired_repeats": 14,
        "timed_solves": 28,
        "controller_exit_code": 1,
        "all_required_numeric_gates_pass": all(p["numeric_pass"] for p in points),
        "conclusion": "Both practical-basis force cells fail the predeclared 3e-11 force gate; all 14 pairs retained. Independent CPU diagnosis localizes the main force discrepancy to native results. Keeps #206 open.",
        "statistical_scope": "Predeclared ordinary-latency criterion: both relative MAD <= 3%, median reduction >= 2%, and paired-bootstrap 95% lower bound > 2%, 10000 resamples with NumPy default_rng(206). Numerical failure forbids performance admission. No fixed-work claim.",
        "points": points,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    manifest = read(args.directory / "manifest.json")
    protocol = manifest["predeclared_statistics"]
    assert protocol == {
        "repeats": 7,
        "maximum_relative_mad": 0.03,
        "minimum_reduction": 0.02,
        "bootstrap_pairs": 10000,
        "seed": 206,
        "confidence": 0.95,
    }
    oracle = read(args.directory.parent / "diagnosis/cpu-diagnosis-v1.json")
    oracle_by_case = {r["case"]: r for r in oracle["rows"]}
    for point in result["points"]:
        native = np.asarray(point["ordinary"]["vibeqc"]["seconds"])
        stock = np.asarray(point["ordinary"]["gpu4pyscf"]["seconds"])
        indices = np.random.default_rng(protocol["seed"]).integers(
            0, 7, size=(protocol["bootstrap_pairs"], 7)
        )
        reductions = 1 - np.median(native[indices], axis=1) / np.median(
            stock[indices], axis=1
        )
        interval = np.quantile(reductions, [0.025, 0.975]).tolist()
        stable = all(
            p["relative_mad"] <= protocol["maximum_relative_mad"]
            for p in point["ordinary"].values()
        )
        point["predeclared_ordinary_statistics"] = {
            "bootstrap95_reduction_interval": interval,
            "stable": stable,
            "timing_criterion_pass": stable
            and point["ordinary_native_reduction_fraction"] >= 0.02
            and interval[0] > 0.02,
            "qualified_ordinary_win": point["numeric_pass"]
            and stable
            and interval[0] > 0.02
            and point["ordinary_native_reduction_fraction"] >= 0.02,
        }
        data = read(args.directory / (point["name"] + ".json"))
        reference = oracle_by_case[data["workload"]["case"]]
        assert (
            reference["input_sha256"]
            == hashlib.sha256(
                raw_bytes(args.directory / (point["name"] + ".json"))
            ).hexdigest()
        )
        assert reference["converged"] and reference["orbital_gradient_norm"] < 1e-11
        errors = {}
        for engine in ("vibeqc", "gpu4pyscf"):
            values = []
            for sample, recorded in zip(
                data[engine]["warm_samples"], reference["engines"][engine], strict=True
            ):
                energy = maximum_error(
                    sample["energies_hartree"], [reference["energy_hartree"]]
                )
                force = maximum_error(
                    sample["forces_hartree_per_bohr"],
                    [reference["forces_hartree_per_bohr"]],
                )
                assert (
                    energy == recorded["energy_error"]
                    and force == recorded["force_error"]
                )
                values.append(force)
            errors[engine] = max(values)
        point["independent_cpu_max_force_errors"] = errors
    result["predeclared_statistics"] = protocol
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Audited {result['paired_repeats']} pairs; {sum(p['numeric_pass'] for p in result['points'])}/2 numerical cells pass; keeps #206 open."
    )

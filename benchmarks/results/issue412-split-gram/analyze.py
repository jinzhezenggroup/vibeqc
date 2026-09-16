"""Recompute the paired endpoint decision from retained samples and work records.

The seven paired repeats and bootstrap protocol were declared before measuring.
All numerical/work gates remain mandatory; failed rows are never discarded.
"""

import argparse
import itertools
import json
import statistics
from pathlib import Path

import numpy as np


def analyze(directory):
    """Check all samples independently before comparing their paired timings."""
    work = json.loads((directory / "work.json").read_text())
    results = {}
    for aos, observable in itertools.product((96, 192, 384, 768), ("forces", "energy")):
        label = f"{aos}-{observable}"
        data = json.loads((directory / f"{label}.json").read_text())
        assert len(data["samples"]) == 14 and len(data["diagnostics"]) == 2
        assert data["control"] == "VIBEQC_DF_RESIDENT_EXCHANGE"
        assert [(row["repeat"], row["policy"]) for row in data["samples"]] == [
            (repeat, policy)
            for repeat in range(7)
            for policy in (
                ("auto", "split4") if repeat % 2 == 0 else ("split4", "auto")
            )
        ]
        arms = {}
        for policy in ("auto", "split4"):
            rows = [row for row in data["samples"] if row["policy"] == policy]
            assert [row["repeat"] for row in rows] == list(range(7))
            arms[policy] = [row["seconds"] for row in rows]
            assert all(np.isfinite(t) and t > 0 for t in arms[policy])
        for row in data["samples"] + data["diagnostics"]:
            assert row["maximum_energy_error"] <= 1e-9
            assert (row["maximum_force_error"] or 0) <= 1e-8
            assert all(
                c["converged"] and not c["warm_start_fallback"]
                for c in row["convergence"]
            )
            if aos >= 384:
                assert row["iterations"] == row["prime_iterations"] == [3]
        ratios = [c / a for a, c in zip(arms["auto"], arms["split4"], strict=True)]
        rng = np.random.Generator(np.random.PCG64(412))
        indices = rng.integers(0, 7, size=(100000, 7))
        bootstrap = np.median(np.asarray(ratios)[indices], axis=1)
        bounds = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        median = {policy: statistics.median(times) for policy, times in arms.items()}
        diagnostic = {row["policy"]: row for row in data["diagnostics"]}
        per_arm = {}
        for policy, row in diagnostic.items():
            observed = work[f"{label}-{policy}"]
            assert not observed["trace_failures"]
            values = observed["values"]
            assert not any(v["key"] == "cuda_error" and v["value"] != 0 for v in values)
            starts = observed["scope_counts"]
            last_iteration = max(
                v["value"] for v in values if v["key"] == "device_iterations"
            )
            graph = sum(v["value"] for v in values if v["key"] == "host_graph_replay")
            assert not any(v["key"] == "graph_construction_attempt" for v in values)
            # A replay can tail-launch further iterations. Completed device
            # iterations, not host launch count, determine captured J/K work.
            captured_iterations = last_iteration if graph else 0
            counts = {
                "iterations": last_iteration,
                "host_graph_replays": graph,
                "j_builds": starts.get("ri_j", 0) + captured_iterations,
                "k_builds": starts.get("ri_k", 0)
                + starts.get("ri_k_occupied", 0)
                + captured_iterations,
                "eigensolves_including_density_factor_seed": starts.get(
                    "compact_eigensolve", 0
                )
                + captured_iterations,
                "final_fock_evaluations": next(
                    v["value"] for v in values if v["key"] == "final_fock_evaluations"
                ),
                "final_density_updates": next(
                    v["value"] for v in values if v["key"] == "final_density_updates"
                ),
                "final_candidate_rejections": next(
                    v["value"]
                    for v in values
                    if v["key"] == "final_candidate_rejections"
                ),
                "warm_start_fallback": any(
                    c["warm_start_fallback"] for c in row["convergence"]
                ),
            }
            groups = row["components"]["groups"]
            force = next(
                (g for g in groups if g["operation"] == "force_response"), None
            )
            grams = observed["occupied_gram_calls"]
            if aos == 768:
                # Four physical K builds remain four builds, even though each
                # candidate build now contains four full-matrix partials.
                assert len(grams) == counts["k_builds"] == 4
                for gram in grams:
                    assert gram["valid"] and gram["cuda_error"] == 0
                    assert gram["execution"] == "stream"
                    counters = gram["counters"]
                    assert counters["occupied_rank"] == 160
                    split = policy == "split4"
                    assert counters["occupied_exchange_split_count"] == (
                        4 if split else 0
                    )
                    assert counters["occupied_exchange_products"] == (4 if split else 1)
                    assert counters["occupied_exchange_blas_calls"] == 1
                    assert counters["occupied_exchange_extra_owned_bytes"] == 0
                    assert counters["occupied_exchange_partial_borrowed_bytes"] == (
                        18 * 1024**2 if split else 0
                    )
            per_arm[policy] = {
                "work": counts,
                "occupied_gram_calls": grams,
                "force_stage_seconds": row.get("force_stage_seconds"),
                "process_device_resident_bytes": row["process_device_resident_bytes"],
                "force_counter_sums": force["counter_sums"] if force else {},
                "inclusive_gpu_region_ms": observed["inclusive_gpu_region_ms"],
                "final_residual_observations": [
                    v
                    for v in values
                    if v["key"].startswith("final_")
                    and v["key"]
                    not in ("final_solve_epoch", "final_density_generation")
                ],
            }
        assert per_arm["auto"]["work"] == per_arm["split4"]["work"]
        conserved = (
            "shell_triples_visited",
            "shell_triples_nonzero",
            "shell_primitive_products",
            "shell_cartesian_component_products",
            "shell_public_weights_consumed",
            "three_center_shell_panels",
            "raw_value_upload_bytes",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "stream_synchronizations",
            "response_final_projection_reused",
        )
        for key in conserved:
            assert per_arm["auto"]["force_counter_sums"].get(key) == per_arm["split4"][
                "force_counter_sums"
            ].get(key), key
        results[label] = {
            "seconds": arms,
            "median_seconds": median,
            "ratio_of_medians": median["split4"] / median["auto"],
            "paired_ratios": ratios,
            "median_paired_ratio": statistics.median(ratios),
            "paired_ratio_range": [min(ratios), max(ratios)],
            "paired_bootstrap_95_percentile_interval": bounds,
            "maximum_energy_error": max(
                row["maximum_energy_error"]
                for row in data["samples"] + data["diagnostics"]
            ),
            "maximum_force_error": max(
                (row["maximum_force_error"] or 0)
                for row in data["samples"] + data["diagnostics"]
            ),
            "diagnostics": per_arm,
        }
    target = results["768-forces"]
    magnitude = target["median_paired_ratio"] <= 0.99
    stable = target["paired_bootstrap_95_percentile_interval"][1] < 1
    regressions = {
        label: result["ratio_of_medians"] <= 1.03
        for label, result in results.items()
        if label != "768-forces"
    }
    return {
        "predeclared_1_percent_force_gain_passed": magnitude,
        "paired_stability_gate_passed": stable,
        "regression_gates": regressions,
        "independent_numerical_and_work_gates_passed": True,
        "endpoint_promotion_gate_passed": magnitude
        and stable
        and all(regressions.values()),
        "interpretation": "Incremental Gram comparison with the post-#415 derivative mapping fixed. Seven pairs and their bootstrap interval describe this campaign, not an independent replication or stock GPU4PySCF comparison.",
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, nargs="?", default=Path(__file__).parent
    )
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2, allow_nan=False))

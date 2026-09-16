"""Recompute the split Gram rejection without treating changed work as a gain.

The original complete-campaign checker remains in candidate-analyze.py.txt.
This report preserves the stopped qualifying campaign and labels subsequent
intrusive diagnostics separately; it never synthesizes missing paired samples.
"""

import argparse
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def numerical(rows):
    """Enforce unchanged independent numerical gates and normal convergence."""
    assert rows
    for row in rows:
        assert row["maximum_energy_error"] <= 1e-9
        assert (row["maximum_force_error"] or 0) <= 1e-8
        assert all(
            c["converged"] and not c["warm_start_fallback"] for c in row["convergence"]
        )
    return {
        "maximum_energy_error": max(r["maximum_energy_error"] for r in rows),
        "maximum_force_error": max((r["maximum_force_error"] or 0) for r in rows),
    }


def actual_work(observed):
    """Count completed SCF/operator work, not partial products or graph launches."""
    assert not observed["trace_failures"]
    values, starts = observed["values"], observed["scope_counts"]
    assert not any(v["key"] == "cuda_error" and v["value"] != 0 for v in values)
    assert not any(v["key"] == "graph_construction_attempt" for v in values)
    iterations = max(v["value"] for v in values if v["key"] == "device_iterations")
    graph = sum(v["value"] for v in values if v["key"] == "host_graph_replay")
    captured = iterations if graph else 0
    result = {
        "iterations": iterations,
        "j_builds": starts.get("ri_j", 0) + captured,
        "k_builds": starts.get("ri_k", 0) + starts.get("ri_k_occupied", 0) + captured,
        "eigensolves_including_seed_factorization": starts.get("compact_eigensolve", 0)
        + captured,
    }
    for key in (
        "final_fock_evaluations",
        "final_density_updates",
        "final_candidate_rejections",
    ):
        result[key] = next(v["value"] for v in values if v["key"] == key)
    return result


def analyze(directory):
    """Reject promotion at the original work gate and retain all observed rows."""
    protocol = read(directory / "endpoint-protocol.json")
    assert protocol["repeats_per_arm"] == 7
    assert protocol["minimum_complete_force_gain_at_768"] == 0.01
    work = read(directory / "work.json")
    regressions = {}
    for observable in ("forces", "energy"):
        label = f"384-{observable}"
        data = read(directory / "endpoints" / f"{label}.json")
        assert len(data["samples"]) == 14 and len(data["diagnostics"]) == 2
        medians = {}
        for arm in ("auto", "split4"):
            rows = [r for r in data["samples"] if r["policy"] == arm]
            assert [r["repeat"] for r in rows] == list(range(7))
            assert all(r["iterations"] == r["prime_iterations"] == [3] for r in rows)
            medians[arm] = statistics.median(r["seconds"] for r in rows)
        counts = {arm: actual_work(work[f"{label}-{arm}"]) for arm in medians}
        assert counts["auto"] == counts["split4"]
        ratio = medians["split4"] / medians["auto"]
        regressions[label] = {
            "median_seconds": medians,
            "ratio_of_medians": ratio,
            "regression_gate_passed": ratio <= 1.03,
            "work": counts,
            **numerical(data["samples"] + data["diagnostics"]),
        }
    stopped = read(directory / "endpoints/768-forces.json")
    rows = stopped["samples"]
    assert [(r["policy"], r["repeat"]) for r in rows] == [("auto", 0), ("split4", 0)]
    assert rows[0]["iterations"] == rows[0]["prime_iterations"] == [3]
    assert rows[1]["iterations"] == rows[1]["prime_iterations"] == [5]
    result = {
        "decision": "Reject split4; retain the existing default and archive the candidate as patches.",
        "fixed_work_gate_passed": False,
        "complete_gain_gate_evaluated": False,
        "paired_stability_gate_evaluated": False,
        "automatic_promotion": False,
        "qualifying_campaign_slurm_job": stopped["slurm_job_id"],
        "completed_384_regressions": regressions,
        "stopped_768_force_pair": {
            "seconds": {r["policy"]: r["seconds"] for r in rows},
            "iterations": {r["policy"]: r["iterations"] for r in rows},
            "prime_iterations": {r["policy"]: r["prime_iterations"] for r in rows},
            "observed_single_pair_ratio": rows[1]["seconds"] / rows[0]["seconds"],
            "interpretation": "One changed-work pair, not a stable timing estimate or matched-work attribution.",
            **numerical(rows),
        },
        "unrun_qualifying_endpoints": [
            "96-forces",
            "96-energy",
            "192-forces",
            "192-energy",
            "768-energy",
        ],
        "followup": {},
    }
    for observable in ("forces", "energy"):
        label = f"768-{observable}"
        path = directory / "changed-work" / f"{label}.json"
        if not path.exists():
            result["followup"][label] = {"status": "pending intrusive follow-up"}
            continue
        data = read(path)
        assert data["library_sha256"] == stopped["library_sha256"]
        assert data["warm_density_sha256"] == stopped["warm_density_sha256"]
        assert len(data["samples"]) == 2
        arms = {}
        for row in data["samples"]:
            assert row["scope"] == "intrusive diagnostic"
            arm = row["policy"]
            observed = work[f"{label}-{arm}"]
            counts = actual_work(observed)
            grams = observed["occupied_gram_calls"]
            assert len(grams) == counts["k_builds"]
            for gram in grams:
                assert (
                    gram["execution"] == "stream"
                    and gram["valid"]
                    and gram["cuda_error"] == 0
                )
                c = gram["counters"]
                assert c["occupied_rank"] == 160
                assert c["occupied_exchange_split_count"] == (
                    4 if arm == "split4" else 0
                )
                assert c["occupied_exchange_extra_owned_bytes"] == 0
            groups = row["components"]["groups"]
            force = next(
                (g for g in groups if g["operation"] == "force_response"), None
            )
            arms[arm] = {
                "work": counts,
                "gram_partial_products": sum(
                    g["counters"]["occupied_exchange_products"] for g in grams
                ),
                "gram_flops": sum(
                    g["counters"]["occupied_exchange_flops"] for g in grams
                ),
                "gram_borrowed_bytes": max(
                    g["counters"]["occupied_exchange_partial_borrowed_bytes"]
                    for g in grams
                ),
                "gram_partial_write_bytes": sum(
                    g["counters"]["occupied_exchange_partial_write_bytes"]
                    for g in grams
                ),
                "gram_reduction_read_bytes": sum(
                    g["counters"]["occupied_exchange_reduction_read_bytes"]
                    for g in grams
                ),
                "final_projection_reused": force["counter_sums"].get(
                    "response_final_projection_reused", 0
                )
                if force
                else 0,
                "process_device_resident_bytes": row["process_device_resident_bytes"],
                "inclusive_gpu_region_ms": observed["inclusive_gpu_region_ms"],
            }
        result["followup"][label] = {
            "status": "intrusive diagnostic; all times excluded from performance qualification",
            "arms": arms,
            "work_differs": arms["auto"]["work"] != arms["split4"]["work"],
            **numerical(data["samples"]),
        }
    resources = directory / "changed-work/resources.json"
    if resources.exists():
        result["sampled_resources"] = read(resources)
        assert all(row["exit_code"] == 0 for row in result["sampled_resources"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, nargs="?", default=Path(__file__).parent
    )
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory), indent=2, allow_nan=False))

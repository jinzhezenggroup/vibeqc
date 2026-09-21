"""Build one #438 DF-SCF operator/final-state ledger from native trace journals.

This reducer keeps logical work, CUDA event timing and host eigensolve timing
separate. Graph-capture construction is structural evidence only; it is never
counted as executed GPU work. Clean endpoint timing remains a separate result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmarks._retention import raw_output_path
from benchmarks.df_component_ledger import aggregate_host, read_host_trace, read_trace
from benchmarks.df_progress_ledger import read_progress

SCHEMA = "vibeqc.issue438.operator-ledger.v1"
_FINAL_KEYS = (
    "final_fock_evaluations",
    "final_eigen_solves",
    "final_density_updates",
    "final_candidate_rejections",
    "final_fixed_point_checks",
    "final_fixed_point_eigen_solves",
    "final_fixed_point_rejections",
)


def _scope_values(scope: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in scope["values"]:
        result[row["key"]] = row["value"]
    return result


def _observations(journal: dict[str, Any], key: str) -> list[Any]:
    rows = [
        row
        for scope in journal["scopes"].values()
        for row in scope["values"]
        if row["key"] == key
    ]
    return [row["value"] for row in sorted(rows, key=lambda row: row["time_ns"])]


def _last_observation(journal: dict[str, Any], key: str) -> Any | None:
    rows = _observations(journal, key)
    return rows[-1] if rows else None


def _final_readbacks(journal: dict[str, Any]) -> dict[int, dict[str, int]]:
    rows: dict[int, tuple[int, dict[str, Any]]] = {}
    for scope in journal["scopes"].values():
        if (
            scope["begin"]["name"] != "compact_iteration_readback"
            or scope["end"] is None
        ):
            continue
        values = _scope_values(scope)
        if not {"system", "device_iterations", "converged"} <= values.keys():
            raise ValueError("incomplete compact iteration readback")
        system = values["system"]
        if type(system) is not int or system < 0:
            raise ValueError("invalid compact-SCF system index")
        stamp = scope["end"]["time_ns"]
        if system not in rows or stamp > rows[system][0]:
            rows[system] = (stamp, values)
    return {
        system: {
            "iterations": values["device_iterations"],
            "converged": values["converged"],
        }
        for system, (_, values) in sorted(rows.items())
    }


def _component_operations(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        op = record["operation"]
        group = groups.setdefault(
            op,
            {
                "stream_calls": 0,
                "graph_capture_calls": 0,
                "gpu_ms": 0.0,
                "shapes": set(),
                "counter_sums": defaultdict(int),
                "graph_capture_counter_sums": defaultdict(int),
            },
        )
        execution = record["execution"]
        group[f"{execution}_calls"] += 1
        if execution == "stream":
            group["gpu_ms"] += record["regions"][0]["gpu_ms"]
        group["shapes"].add(
            (
                record["systems"],
                record["nbf"],
                record["naux"],
                record["source_backed"],
                record["streamed"],
            )
        )
        counter_key = (
            "counter_sums" if execution == "stream" else "graph_capture_counter_sums"
        )
        for name, value in record["counters"].items():
            group[counter_key][name] += value
    return {
        op: {
            **row,
            "shapes": [
                {
                    "systems": shape[0],
                    "nao": shape[1],
                    "naux": shape[2],
                    "source_backed": shape[3],
                    "streamed": shape[4],
                }
                for shape in sorted(row["shapes"])
            ],
            "counter_sums": dict(row["counter_sums"]),
            "graph_capture_counter_sums": dict(row["graph_capture_counter_sums"]),
        }
        for op, row in sorted(groups.items())
    }


def summarize_call(
    progress: dict[str, Any],
    components: list[dict[str, Any]],
    host: list[dict[str, Any]],
    *,
    expected_nao: int,
    expected_naux: int,
) -> dict[str, Any]:
    if not progress["complete"]:
        raise ValueError("operator ledger requires a complete progress journal")
    if expected_nao <= 0 or expected_naux <= 0:
        raise ValueError("expected AO dimensions must be positive")
    for record in components:
        if record["operation"] in {
            "ri_j",
            "ri_k",
            "ri_k_occupied",
            "force_response",
            "final_state_retained_jk",
        } and (record["nbf"], record["naux"]) != (expected_nao, expected_naux):
            raise ValueError(
                "component trace dimensions do not match the requested cell"
            )

    readbacks = _final_readbacks(progress)
    if not readbacks:
        raise ValueError("missing compact-SCF iteration readback")
    provider_history = {
        "scf_exchange": _observations(progress, "scf_exchange_provider"),
        "final_exchange": _observations(progress, "final_exchange_provider"),
        "final_exchange_policy": _observations(progress, "final_exchange_policy"),
        "final_exchange_fallback": _observations(progress, "final_exchange_fallback"),
    }
    providers = {
        key: values[-1] if values else None for key, values in provider_history.items()
    }
    if providers["scf_exchange"] not in {"dense", "occupied"}:
        raise ValueError("missing SCF exchange-provider provenance")
    if providers["final_exchange"] not in {"dense", "occupied"}:
        raise ValueError("missing final exchange-provider provenance")
    if providers["final_exchange_fallback"] is None:
        raise ValueError("missing final exchange fallback provenance")

    final_state = {key: _last_observation(progress, key) for key in _FINAL_KEYS}
    missing_final = [key for key, value in final_state.items() if value is None]
    if missing_final:
        raise ValueError(f"missing final-state observations: {missing_final}")

    operations = _component_operations(components)
    force = operations.get("force_response", {})
    force_counters = force.get("counter_sums", {})
    reused = force_counters.get("response_final_projection_reused", 0)
    reconstructed = force_counters.get("response_final_projection_reconstructed", 0)
    if reused and reconstructed:
        raise ValueError(
            "force response reports both reused and reconstructed final projection"
        )

    host_summary = aggregate_host(host)
    iterations_by_system = {
        str(system): row["iterations"] for system, row in readbacks.items()
    }
    converged_by_system = {
        str(system): bool(row["converged"]) for system, row in readbacks.items()
    }
    return {
        "schema": SCHEMA,
        "dimensions": {"nao": expected_nao, "naux": expected_naux},
        "scf": {
            "iterations_by_system": iterations_by_system,
            "converged_by_system": converged_by_system,
            "update_systems_total": sum(iterations_by_system.values()),
            "seed_systems": sum(value > 0 for value in iterations_by_system.values()),
            "exchange_provider": providers["scf_exchange"],
            "exchange_provider_history": provider_history["scf_exchange"],
        },
        "final_state": {
            **final_state,
            "exchange_provider": providers["final_exchange"],
            "exchange_provider_history": provider_history["final_exchange"],
            "exchange_policy": providers["final_exchange_policy"],
            "exchange_policy_history": provider_history["final_exchange_policy"],
            "exchange_fallback": providers["final_exchange_fallback"],
            "exchange_fallback_history": provider_history["final_exchange_fallback"],
        },
        "operators": operations,
        "eigensolves": {
            "reference_by_reason": host_summary["eigensolves_by_reason"],
            "device_by_reason": host_summary["device_eigensolves_by_reason"],
        },
        "force_response": {
            "final_projection_reused": reused,
            "final_projection_reconstructed": reconstructed,
        },
        "interpretation": (
            "SCF iterations are logical per-system updates from final device readbacks. "
            "Operator stream_calls are observed native provider invocations; graph_capture_calls "
            "describe construction and are not executed-work counts. GPU milliseconds are CUDA "
            "event intervals and must not be added to host timings or clean endpoint latency. "
            "Use separate unprofiled runs for time-to-solution claims."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--nao", type=int, required=True)
    parser.add_argument("--naux", type=int, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    result = summarize_call(
        read_progress(args.progress),
        read_trace(args.components),
        read_host_trace(args.host),
        expected_nao=args.nao,
        expected_naux=args.naux,
    )
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()

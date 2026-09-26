"""Hardware-free contracts for the #438 practical operator ledger."""

from typing import Any

from benchmarks.issue438_practical_operator_ledger import summarize_call


def _progress() -> dict[str, Any]:
    def scope(
        index: int, name: str, values: dict[str, Any], time: int
    ) -> dict[str, Any]:
        return {
            "begin": {"id": index, "name": name, "time_ns": time},
            "end": {"time_ns": time + 10},
            "values": [
                {"key": key, "value": value, "time_ns": time + offset + 1}
                for offset, (key, value) in enumerate(values.items())
            ],
            "children": [],
        }

    scopes = {
        0: scope(0, "compact_rhf_scf", {"scf_exchange_provider": "occupied"}, 0),
        1: scope(
            1,
            "compact_iteration_readback",
            {"system": 0, "device_iterations": 2, "converged": 1},
            100,
        ),
        2: scope(
            2,
            "finalization",
            {
                "final_exchange_policy": "auto",
                "final_exchange_fallback": "none",
                "final_exchange_provider": "occupied",
                "final_fock_evaluations": 1,
                "final_eigen_solves": 1,
                "final_density_updates": 0,
                "final_candidate_rejections": 0,
                "final_fixed_point_checks": 1,
                "final_fixed_point_eigen_solves": 1,
                "final_fixed_point_rejections": 0,
            },
            200,
        ),
    }
    return {"scopes": scopes, "pending": [], "truncated_tail": False, "complete": True}


def _component(
    operation: str,
    *,
    counters: dict[str, int] | None = None,
    execution: str = "stream",
    identity: int = 1,
) -> dict[str, Any]:
    return {
        "schema": "vibeqc.df_trace",
        "version": 1,
        "id": identity,
        "operation": operation,
        "execution": execution,
        "valid": True,
        "cuda_error": 0,
        "nvtx": False,
        "systems": 1,
        "system_offset": 0,
        "nbf": 96,
        "naux": 464,
        "source_backed": True,
        "streamed": False,
        "final_synchronization_ms": 0.0,
        "host_completion_ms": 1.0,
        "profiler_event_count": 2 if execution == "stream" else 0,
        "dropped_regions": 0,
        "dropped_tiles": 0,
        "regions": [
            {
                "name": operation,
                "parent": -1,
                "host_ms": 1.0,
                "gpu_ms": 0.8 if execution == "stream" else None,
            }
        ],
        "counters": counters or {},
        "tiles": [],
    }


def _host() -> list[dict[str, Any]]:
    return [
        {
            "schema": "vibeqc.df_host_trace",
            "version": 1,
            "id": 1,
            "valid": True,
            "regions": [
                {
                    "name": "finalization",
                    "parent": -1,
                    "reason": "unspecified",
                    "item": -1,
                    "nbf": 96,
                    "finished": True,
                    "failed": False,
                    "wall_ms": 1.0,
                    "cpu_ms": 1.0,
                },
                {
                    "name": "device_eigensolve",
                    "parent": 0,
                    "reason": "final_fock",
                    "item": 0,
                    "nbf": 96,
                    "finished": True,
                    "failed": False,
                    "wall_ms": 0.2,
                    "cpu_ms": 0.1,
                },
            ],
        }
    ]


def test_practical_ledger_separates_logical_work_and_provider_provenance() -> None:
    result = summarize_call(
        _progress(),
        [
            _component("ri_j", identity=1),
            _component("ri_k_occupied", identity=2),
            _component(
                "force_response",
                counters={"response_final_projection_reused": 1},
                identity=3,
            ),
            _component("ri_j", execution="graph_capture", identity=4),
        ],
        _host(),
        expected_nao=96,
        expected_naux=464,
    )
    assert result["scf"]["iterations_by_system"] == {"0": 2}
    assert result["scf"]["update_systems_total"] == 2
    assert result["scf"]["exchange_provider"] == "occupied"
    assert result["final_state"]["exchange_provider"] == "occupied"
    assert result["final_state"]["exchange_provider_history"] == ["occupied"]
    assert result["final_state"]["exchange_fallback"] == "none"
    assert result["final_state"]["exchange_fallback_history"] == ["none"]
    assert result["operators"]["ri_j"]["stream_calls"] == 1
    assert result["operators"]["ri_j"]["graph_capture_calls"] == 1
    assert result["operators"]["ri_j"]["gpu_ms"] == 0.8
    assert result["force_response"]["final_projection_reused"] == 1
    assert result["force_response"]["final_projection_reconstructed"] == 0
    assert result["eigensolves"]["device_by_reason"]["final_fock"]["calls"] == 1


def test_reconstructed_final_projection_is_explicit() -> None:
    result = summarize_call(
        _progress(),
        [
            _component("ri_j", identity=1),
            _component("ri_k", identity=2),
            _component(
                "force_response",
                counters={"response_final_projection_reconstructed": 1},
                identity=3,
            ),
        ],
        _host(),
        expected_nao=96,
        expected_naux=464,
    )
    assert result["force_response"] == {
        "final_projection_reused": 0,
        "final_projection_reconstructed": 1,
    }

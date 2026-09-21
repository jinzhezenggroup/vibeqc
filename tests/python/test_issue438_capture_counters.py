"""Captured graph construction is not executed operator work."""

from test_issue438_practical_operator_ledger import _component, _host, _progress

from benchmarks.issue438_practical_operator_ledger import summarize_call


def test_capture_counters_do_not_pollute_executed_work() -> None:
    rows = [
        _component(
            "force_response",
            counters={"response_final_projection_reused": 1, "flops": 10},
        ),
        _component(
            "force_response",
            counters={"response_final_projection_reconstructed": 1, "flops": 100},
            execution="graph_capture",
            identity=2,
        ),
    ]
    result = summarize_call(
        _progress(), rows, _host(), expected_nao=96, expected_naux=464
    )
    assert result["force_response"] == {
        "final_projection_reused": 1,
        "final_projection_reconstructed": 0,
    }
    op = result["operators"]["force_response"]
    assert op["counter_sums"] == {"response_final_projection_reused": 1, "flops": 10}
    assert op["graph_capture_counter_sums"] == {
        "response_final_projection_reconstructed": 1,
        "flops": 100,
    }
    assert op["stream_calls"] == op["graph_capture_calls"] == 1

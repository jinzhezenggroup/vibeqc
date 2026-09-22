"""Deterministic compiler work admission, not numerical/production fallback."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from vibeqc_compiler.common.compiler_work import (
    CompilerWorkLimit,
    compiler_work_budget,
)
from vibeqc_compiler.integral.capabilities import (
    _check_recurrence,
    build_capability_report,
)
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.integral.expr import Graph, Node
from vibeqc_compiler.integral.shell_spec import PSSS_SPEC


def test_budget_stops_before_graph_mutation_and_restores_context() -> None:
    graph = Graph()  # Existing graphs must also respect the active scope.
    with compiler_work_budget(2) as count:
        x = graph.variable("x")
        assert graph._intern(Node("variable", payload="x")).identifier == x.identifier
        assert count.used == 2  # Interning work, not just unique-node allocation.
        with pytest.raises(CompilerWorkLimit, match="unqualified"):
            graph.variable("y")
        assert len(graph.nodes) == 1
    assert graph.variable("y").identifier == 1


def test_nested_budget_cannot_disable_outer_limit() -> None:
    with compiler_work_budget(1) as outer:
        with compiler_work_budget(None):
            Graph().variable("one")
            with pytest.raises(CompilerWorkLimit):
                Graph().variable("two")
        assert outer.used == 1
    with compiler_work_budget(3) as outer:
        with compiler_work_budget(1) as inner:
            Graph().variable("one")
            with pytest.raises(CompilerWorkLimit):
                Graph().variable("two")
        assert outer.used == inner.used == 1
        Graph().variable("three")
        assert outer.used == 2


def test_budget_scope_is_thread_local() -> None:
    with compiler_work_budget(1) as outer:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(lambda: [Graph().variable(str(i)) for i in range(4)]).result()
        assert outer.used == 0
        Graph().variable("parent")
        assert outer.used == 1


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "100"])
def test_invalid_work_budget_is_rejected_before_reporting(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_capability_report(
            architecture="sm_120", specifications=(), maximum_symbolic_work=limit
        )


def test_small_complete_emission_report_retains_unbounded_result() -> None:
    target = cuda_target_info("sm_120")
    bounded = _check_recurrence(PSSS_SPEC, "subset_wick", target)
    unbounded = _check_recurrence(
        PSSS_SPEC, "subset_wick", target, maximum_symbolic_work=None
    )
    assert bounded == unbounded
    assert "packed_tasks" in bounded.schedules


def test_exhausted_candidate_is_not_advertised_as_emitted() -> None:
    result = _check_recurrence(
        PSSS_SPEC, "subset_wick", cuda_target_info("sm_120"), maximum_symbolic_work=1
    )
    assert "packed_tasks" not in result.schedules
    assert any(
        "packed_tasks:" in reason and "unqualified" in reason
        for reason in result.reasons
    )
    report = build_capability_report(
        architecture="sm_120", specifications=(PSSS_SPEC,), maximum_symbolic_work=1
    )
    assert report["compiler_work_budget"] == {
        "unit": "symbolic_intern_attempt",
        "per_candidate": 1,
    }

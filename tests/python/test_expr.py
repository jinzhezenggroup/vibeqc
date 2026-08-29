"""Focused tests for symbolic DAG SSA and resource-model analysis."""

from __future__ import annotations

from tools.vibeqc_codegen import (
    PSSS_SPEC,
    MaterializationDecision,
    MaterializationPlan,
    RematerializationPolicy,
    SsaAnalysis,
    SsaValueLifetime,
    build_weighted_shell_contraction_kernel,
)
from tools.vibeqc_codegen.cuda import CudaEmitter
from tools.vibeqc_codegen.expr import Graph


def test_ssa_analysis_records_shared_last_uses_and_peak_liveness():
    """Count a shared operand through its final consumer and root output."""

    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    z = graph.variable("z")
    shared = x + y
    scaled = shared * z
    root = scaled + shared
    graph.exponential(graph.variable("unreachable"))

    analysis = graph.analyze_ssa((root,))
    assert isinstance(analysis, SsaAnalysis)
    assert all(isinstance(item, SsaValueLifetime) for item in analysis.lifetimes)
    assert analysis.root_count == 1
    assert analysis.reachable_node_count == 6
    assert analysis.operation_counts == (
        ("add", 2),
        ("multiply", 1),
        ("variable", 3),
    )
    assert analysis.arithmetic_operation_count == 3
    assert analysis.materialized_value_count == 3
    assert analysis.peak_live_values == 3

    by_identifier = {item.identifier: item for item in analysis.lifetimes}
    assert by_identifier[shared.identifier].use_count == 2
    assert by_identifier[shared.identifier].definition_index == 2
    assert by_identifier[shared.identifier].last_use_index == 5
    assert by_identifier[scaled.identifier].use_count == 1
    assert by_identifier[scaled.identifier].last_use_index == 5
    assert by_identifier[root.identifier].use_count == 1
    assert by_identifier[root.identifier].last_use_index == 6
    assert analysis.to_payload() == {
        "root_count": 1,
        "reachable_node_count": 6,
        "operation_counts": {"add": 2, "multiply": 1, "variable": 3},
        "arithmetic_operation_count": 3,
        "materialized_value_count": 3,
        "estimated_peak_live_values": 3,
    }


def test_ssa_analysis_counts_duplicate_edges_and_ordered_root_reads():
    """Treat repeated operands and repeated output consumers as real uses."""

    graph = Graph()
    shared = graph.variable("x") + graph.variable("y")
    square = shared * shared

    analysis = graph.analyze_ssa((square, shared))
    by_identifier = {item.identifier: item for item in analysis.lifetimes}
    assert by_identifier[shared.identifier].use_count == 3
    assert by_identifier[shared.identifier].last_use_index == 5
    assert by_identifier[square.identifier].use_count == 1
    assert by_identifier[square.identifier].last_use_index == 4
    assert analysis.peak_live_values == 2


def test_ssa_analysis_handles_empty_and_external_only_roots():
    """Keep degenerate static models well-defined without fake temporaries."""

    graph = Graph()
    assert graph.analyze_ssa(()).to_payload() == {
        "root_count": 0,
        "reachable_node_count": 0,
        "operation_counts": {},
        "arithmetic_operation_count": 0,
        "materialized_value_count": 0,
        "estimated_peak_live_values": 0,
    }

    external = graph.variable("external")
    analysis = graph.analyze_ssa((external,))
    assert analysis.operation_counts == (("variable", 1),)
    assert analysis.arithmetic_operation_count == 0
    assert analysis.materialized_value_count == 0
    assert analysis.peak_live_values == 0


def test_ssa_materialized_count_matches_current_cuda_emitter():
    """Anchor the static model to the temporaries emitted for a real shell DAG."""

    kernel = build_weighted_shell_contraction_kernel(PSSS_SPEC)
    roots = tuple(
        kernel.gradients[center][coordinate]
        for center in range(3)
        for coordinate in range(3)
    )
    analysis = kernel.graph.analyze_ssa(roots)
    emitter = CudaEmitter(kernel.graph, {})
    emitter.emit(roots)

    assert analysis.materialized_value_count == len(emitter.lines)
    assert analysis.arithmetic_operation_count == len(emitter.lines)
    assert 0 < analysis.peak_live_values < analysis.materialized_value_count
    rebuilt = build_weighted_shell_contraction_kernel(PSSS_SPEC)
    rebuilt_roots = tuple(
        rebuilt.gradients[center][coordinate]
        for center in range(3)
        for coordinate in range(3)
    )
    assert rebuilt.graph.analyze_ssa(rebuilt_roots).to_payload() == (
        analysis.to_payload()
    )


def test_materialized_cse_plan_preserves_existing_cuda_source_shape():
    """Keep the default placement byte-compatible with the legacy emitter."""

    graph = Graph()
    shared = graph.variable("x") + graph.variable("y")
    root = shared * graph.variable("z") + shared
    roots = (root,)
    plan = graph.materialization_plan(roots)

    assert isinstance(plan, MaterializationPlan)
    assert all(isinstance(item, MaterializationDecision) for item in plan.decisions)
    assert plan.policy.name == "materialized_cse"
    assert plan.baseline_arithmetic_operation_count == 3
    assert plan.arithmetic_operation_count == 3
    assert plan.baseline_materialized_value_count == 3
    assert plan.materialized_value_count == 3
    assert plan.inlined_value_count == 0
    assert plan.rematerialized_value_count == 0

    legacy = CudaEmitter(graph, {})
    legacy.emit(roots)
    planned = CudaEmitter(graph, {}, materialization_plan=plan)
    planned.emit(roots)
    assert planned.lines == legacy.lines
    assert planned.reference(root) == legacy.reference(root)


def test_single_use_plan_inlines_roots_with_exact_parentheses_and_metrics():
    """Inline a one-use chain without changing its arithmetic operation count."""

    graph = Graph()
    summed = graph.variable("x") + graph.variable("y")
    root = summed * graph.variable("z")
    roots = (root,)
    plan = graph.materialization_plan(
        roots,
        RematerializationPolicy.inline_single_use_values(),
    )

    assert plan.arithmetic_operation_count == 2
    assert plan.materialized_value_count == 0
    assert plan.peak_live_values == 0
    assert plan.inlined_value_count == 2
    assert {item.reason for item in plan.decisions} == {"single_use"}
    assert plan.to_payload()["post_optimization"] == {
        "operation_counts": {"add": 1, "multiply": 1},
        "arithmetic_operation_count": 2,
        "materialized_value_count": 0,
        "estimated_peak_live_values": 0,
    }

    emitter = CudaEmitter(
        graph,
        {"x": "input_x", "y": "input_y", "z": "input_z"},
        materialization_plan=plan,
    )
    emitter.emit(roots)
    assert emitter.lines == []
    assert emitter.reference(root) == "((input_x + input_y) * input_z)"


def test_pressure_plan_trades_bounded_recomputation_for_a_shorter_live_set():
    """Rematerialize a cheap long-lived shared value within the operation cap."""

    graph = Graph()
    shared = graph.variable("x") + graph.variable("y")
    scaled = shared * graph.variable("z")
    shifted = scaled + graph.variable("w")
    stretched = shifted * graph.variable("q")
    root = stretched + shared
    roots = (root,)

    single_use = graph.materialization_plan(
        roots,
        RematerializationPolicy.inline_single_use_values(),
    )
    pressure = graph.materialization_plan(
        roots,
        RematerializationPolicy.pressure_rematerialized(),
    )
    shared_decision = next(
        item for item in pressure.decisions if item.identifier == shared.identifier
    )

    assert single_use.arithmetic_operation_count == 5
    assert single_use.materialized_identifiers == frozenset({shared.identifier})
    assert pressure.arithmetic_operation_count == 6
    assert pressure.materialized_value_count == 0
    assert pressure.peak_live_values < single_use.peak_live_values
    assert pressure.rematerialized_value_count == 1
    assert shared_decision.use_count == 2
    assert shared_decision.lifetime_span >= 3
    assert shared_decision.estimated_live_range_cost > (
        shared_decision.estimated_recomputation_cost
    )
    assert shared_decision.reason == "live_range_benefit"

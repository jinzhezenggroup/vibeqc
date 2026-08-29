"""Focused tests for symbolic DAG SSA and resource-model analysis."""

from __future__ import annotations

import math
from fractions import Fraction

from tools.vibeqc_codegen import (
    PSSS_SPEC,
    AlgebraForm,
    AlgebraFusion,
    AlgebraOrdering,
    MaterializationDecision,
    MaterializationPlan,
    PowerLowering,
    RematerializationPolicy,
    SsaAnalysis,
    SsaValueLifetime,
    build_packed_force_geometry_algebra,
    build_psss_kernel,
    build_weighted_shell_contraction_kernel,
)
from tools.vibeqc_codegen.cuda import CudaEmitter, format_constant
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


def test_pressure_aware_ordering_reduces_exact_materialized_peak_liveness():
    """Delay ready definitions and free effective inlined dependencies early."""

    kernel = build_weighted_shell_contraction_kernel(PSSS_SPEC)
    roots = tuple(
        kernel.gradients[center][coordinate]
        for center in range(3)
        for coordinate in range(3)
    )
    policy = RematerializationPolicy.inline_single_use_values()
    topological = kernel.graph.materialization_plan(
        roots,
        policy,
        AlgebraOrdering.TOPOLOGICAL,
    )
    pressure_aware = kernel.graph.materialization_plan(
        roots,
        policy,
        AlgebraOrdering.PRESSURE_AWARE,
    )

    assert pressure_aware.materialized_identifiers == (
        topological.materialized_identifiers
    )
    assert pressure_aware.arithmetic_operation_count == (
        topological.arithmetic_operation_count
    )
    assert pressure_aware.emission_order != topological.emission_order
    assert pressure_aware.reordered_value_count > 0
    assert pressure_aware.peak_live_values < topological.peak_live_values
    assert pressure_aware.to_payload()["ordering"] == "pressure_aware"

    emitter = CudaEmitter(
        kernel.graph,
        {},
        materialization_plan=pressure_aware,
    )
    emitter.emit(roots)
    assert len(emitter.lines) == pressure_aware.materialized_value_count
    assert all(emitter.reference(root) for root in roots)


def test_pressure_aware_ordering_falls_back_when_exact_peak_does_not_improve():
    """Avoid source churn when the greedy candidate is not actually better."""

    kernel = build_weighted_shell_contraction_kernel(PSSS_SPEC)
    roots = tuple(
        kernel.gradients[center][coordinate]
        for center in range(3)
        for coordinate in range(3)
    )
    topological = kernel.graph.materialization_plan(roots)
    guarded = kernel.graph.materialization_plan(
        roots,
        ordering=AlgebraOrdering.PRESSURE_AWARE,
    )
    assert guarded.emission_order == topological.emission_order
    assert guarded.peak_live_values == topological.peak_live_values
    assert guarded.reordered_value_count == 0


def test_fma_fusion_removes_one_use_multiply_and_counts_one_operation():
    """Contract a direct multiply/add pair in both the plan and CUDA source."""

    graph = Graph()
    product = graph.variable("x") * graph.variable("y")
    root = product + graph.variable("z")
    roots = (root,)
    plan = graph.materialization_plan(roots, fusion=AlgebraFusion.FMA)
    product_decision = next(
        item for item in plan.decisions if item.identifier == product.identifier
    )

    assert plan.fusion == AlgebraFusion.FMA
    assert plan.fma_operations == ((root.identifier, product.identifier),)
    assert plan.operation_counts == (("fma", 1),)
    assert plan.arithmetic_operation_count == 1
    assert plan.materialized_identifiers == frozenset({root.identifier})
    assert plan.peak_live_values == 1
    assert plan.fma_operation_count == 1
    assert product_decision.reason == "fma_operand"

    emitter = CudaEmitter(graph, {}, materialization_plan=plan)
    emitter.emit(roots)
    assert emitter.lines == ["  const double v0 = fma(x, y, z);"]
    assert emitter.reference(root) == "v0"


def test_fma_fusion_preserves_shared_multiply_cse_and_supports_inline_root():
    """Do not duplicate shared products, but inline a fused single-use root."""

    graph = Graph()
    product = graph.variable("x") * graph.variable("y")
    first = product + graph.variable("z")
    shared_root = first + product
    shared_plan = graph.materialization_plan(
        (shared_root,),
        fusion=AlgebraFusion.FMA,
    )
    assert product.identifier in shared_plan.materialized_identifiers
    assert shared_plan.fma_operation_count == 0

    inline_graph = Graph()
    inline_root = (
        inline_graph.variable("a") * inline_graph.variable("b")
        + inline_graph.variable("c")
    )
    inline_plan = inline_graph.materialization_plan(
        (inline_root,),
        RematerializationPolicy.inline_single_use_values(),
        fusion=AlgebraFusion.FMA,
    )
    emitter = CudaEmitter(
        inline_graph,
        {},
        materialization_plan=inline_plan,
    )
    emitter.emit((inline_root,))
    assert emitter.lines == []
    assert emitter.reference(inline_root) == "fma(a, b, c)"


def test_canonical_nary_rebuild_flattens_and_folds_associative_regions():
    """Represent equal sums identically with one scalar-counted n-ary node."""

    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    root = ((x + 2.0) + (y + 3.0)) + x
    canonical, roots = graph.apply_algebra_form(
        (root,),
        AlgebraForm.CANONICAL_NARY,
    )
    rebuilt = roots[0]
    node = canonical.node(rebuilt)

    assert node.operation == "add"
    assert len(node.arguments) == 4
    assert sum(
        canonical.nodes[item].operation == "constant" for item in node.arguments
    ) == 1
    assert canonical.analyze_ssa(roots).arithmetic_operation_count == 3
    assert canonical.evaluate(rebuilt, {"x": 1.5, "y": -2.0}) == 6.0

    emitter = CudaEmitter(canonical, {})
    emitter.emit(roots)
    assert len(emitter.lines) == 1
    assert emitter.lines[0].count(" + ") == 3


def test_canonical_forms_ignore_binary_parenthesization():
    """Emit one stable associative form for equivalent binary source trees."""

    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    z = graph.variable("z")
    left_associative = (x + y) + z
    right_associative = x + (y + z)

    def emitted_form(root, form):
        canonical, roots = graph.apply_algebra_form((root,), form)
        emitter = CudaEmitter(canonical, {})
        emitter.emit(roots)
        return emitter.lines, emitter.reference(roots[0])

    for form in (AlgebraForm.CANONICAL_NARY, AlgebraForm.FACTORED_NARY):
        assert emitted_form(left_associative, form) == emitted_form(
            right_associative,
            form,
        )


def test_exact_rational_coefficients_fold_before_cuda_lowering():
    """Keep coefficient algebra exact until the final double literal."""

    graph = Graph()
    x = graph.variable("x")
    one_tenth = graph.coerce(0.1)
    two_tenths = graph.coerce(0.2)
    folded = graph.add(one_tenth, two_tenths)

    assert graph.node(one_tenth).payload == Fraction(1, 10)
    assert graph.node(folded).payload == Fraction(3, 10)
    assert format_constant(graph.node(folded).payload) == "0.29999999999999999"
    assert format_constant(Fraction(1, 3)) == "0.33333333333333331"
    rational = graph.constant(Fraction(2, 3))
    assert graph.node(graph.reciprocal(rational)).payload == Fraction(3, 2)
    assert graph.node(rational.pow(-2)).payload == Fraction(9, 4)

    root = Fraction(1, 3) * x + Fraction(2, 3) * x
    factored, roots = graph.apply_algebra_form(
        (root,),
        AlgebraForm.FACTORED_NARY,
    )
    assert factored.node(roots[0]).operation == "variable"
    assert factored.evaluate(roots[0], {"x": 1.25}) == 1.25

    transcendental = graph.exponential(graph.constant(1))
    assert isinstance(graph.node(transcendental).payload, float)

    weighted = build_weighted_shell_contraction_kernel(
        PSSS_SPEC,
        component_indices=(0,),
    )
    coefficients = (
        node.payload
        for node in weighted.graph.nodes
        if node.operation == "constant"
    )
    assert all(isinstance(coefficient, Fraction) for coefficient in coefficients)


def test_small_integer_power_lowering_reuses_squares_and_preserves_other_powers():
    """Expand bounded integer powers without duplicating shared squares."""

    graph = Graph()
    x = graph.variable("x")
    root = x.pow(4) + x.pow(-3) + x.pow(0.5)
    lowered, roots = graph.apply_algebra_form(
        (root,),
        AlgebraForm.BINARY,
        PowerLowering.SMALL_INTEGER,
    )

    counts = lowered.operation_counts(roots)
    assert counts["multiply"] == 3
    assert counts["reciprocal"] == 1
    assert counts["power"] == 1
    values = {"x": 1.75}
    assert lowered.evaluate(roots[0], values) == graph.evaluate(root, values)

    emitter = CudaEmitter(lowered, {})
    emitter.emit(roots)
    source = "\n".join(emitter.lines)
    assert "pow(x, 4" not in source
    assert "pow(x, -3" not in source
    assert "sqrt(x)" in source

    psss = build_psss_kernel("x")
    psss_roots = (
        psss.value,
        *(gradient for center in psss.gradients for gradient in center),
    )
    native_power_count = psss.graph.operation_counts(psss_roots)["power"]
    lowered_psss, lowered_psss_roots = psss.graph.apply_algebra_form(
        psss_roots,
        AlgebraForm.BINARY,
        PowerLowering.SMALL_INTEGER,
    )
    assert lowered_psss.operation_counts(lowered_psss_roots)["power"] < (
        native_power_count
    )


def test_cuda_emitter_assignment_binds_stored_root_for_later_cse():
    """Use a structured output field as the next expression's CSE input."""

    graph = Graph()
    x = graph.variable("x")
    stored = x + 1.0
    consumer = stored * 2.0
    emitter = CudaEmitter(graph, {})
    emitter.emit_assignment(stored, "geometry.stored")
    emitter.emit((consumer,))

    source = "\n".join(emitter.lines)
    assert "geometry.stored =" in source
    assert emitter.reference(stored) == "geometry.stored"
    assert "geometry.stored * 2" in source


def test_packed_force_geometry_algebra_matches_scalar_formulas():
    """Describe packed geometry completely before choosing CUDA storage."""

    geometry = build_packed_force_geometry_algebra()
    values = {
        "p": 1.7,
        "q": 2.3,
        "first_reduced_exponent": 0.4,
        "second_reduced_exponent": 0.6,
        "first_weighted_coefficient": 1.25,
        "second_weighted_coefficient": -0.75,
    }
    coordinates = {
        "first": (0.2, -0.3, 0.5),
        "second": (-0.4, 0.1, 0.7),
        "third": (0.8, -0.2, -0.6),
        "fourth": (0.3, 0.9, -0.1),
    }
    product_p = (0.05, -0.1, 0.6)
    product_q = (0.65, 0.25, -0.4)
    for center, items in coordinates.items():
        for axis, item in zip(("x", "y", "z"), items, strict=True):
            values[f"{center}_coordinate_{axis}"] = item
    for prefix, items in (("product_p", product_p), ("product_q", product_q)):
        for axis, item in zip(("x", "y", "z"), items, strict=True):
            values[f"{prefix}_{axis}"] = item

    evaluate = lambda expression: geometry.graph.evaluate(expression, values)
    expected_rho = values["p"] * values["q"] / (values["p"] + values["q"])
    difference = tuple(product_p[index] - product_q[index] for index in range(3))
    squared_distance = sum(item * item for item in difference)
    assert evaluate(geometry.rho) == expected_rho
    assert evaluate(geometry.inverse_two_p) == 0.5 / values["p"]
    assert evaluate(geometry.inverse_two_q) == 0.5 / values["q"]
    assert tuple(map(evaluate, geometry.difference)) == difference
    assert evaluate(geometry.argument_squared_distance) == squared_distance
    assert evaluate(geometry.boys_argument) == expected_rho * squared_distance
    assert evaluate(geometry.prefactor) == (
        34.986836655249725
        / (values["p"] * values["q"] * math.sqrt(values["p"] + values["q"]))
    )
    assert evaluate(geometry.primitive_coefficient) == -0.9375

    lowered, roots = geometry.graph.apply_algebra_form(
        geometry.roots,
        AlgebraForm.BINARY,
        PowerLowering.SMALL_INTEGER,
    )
    assert lowered.operation_counts(roots)["power"] == 1


def test_factored_nary_extracts_common_factors_and_collects_like_terms():
    """Turn repeated multiplicative terms into deterministic Horner-like sums."""

    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    z = graph.variable("z")
    root = 2.0 * x * y + 3.0 * x * z + x + x
    canonical, canonical_roots = graph.apply_algebra_form(
        (root,),
        AlgebraForm.CANONICAL_NARY,
    )
    factored, factored_roots = graph.apply_algebra_form(
        (root,),
        AlgebraForm.FACTORED_NARY,
    )

    values = {"x": 1.25, "y": -0.5, "z": 2.0}
    assert factored.evaluate(factored_roots[0], values) == canonical.evaluate(
        canonical_roots[0], values
    )
    assert factored.analyze_ssa(factored_roots).arithmetic_operation_count < (
        canonical.analyze_ssa(canonical_roots).arithmetic_operation_count
    )
    root_node = factored.node(factored_roots[0])
    assert root_node.operation == "multiply"
    assert any(
        factored.nodes[item].operation == "add" for item in root_node.arguments
    )


def test_nary_differentiation_and_fma_lowering_cover_variable_arity_nodes():
    """Keep symbolic AD and explicit contraction correct beyond binary nodes."""

    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    z = graph.variable("z")
    product = graph.multiply_many((x, y, z))
    derivative = graph.differentiate(product, x)
    assert graph.evaluate(derivative, {"x": 2.0, "y": 3.0, "z": 4.0}) == 12.0

    pair = x * y
    root = graph.add_many((pair, z, graph.variable("w")))
    plan = graph.materialization_plan((root,), fusion=AlgebraFusion.FMA)
    emitter = CudaEmitter(graph, {}, materialization_plan=plan)
    emitter.emit((root,))
    assert emitter.lines == ["  const double v0 = fma(x, y, (z + w));"]
    assert plan.operation_counts == (("add", 1), ("fma", 1))

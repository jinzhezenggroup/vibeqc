"""First-class TensorIR precision/cast contracts for #528."""

import typing

import numpy as np
import pytest
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    PrecisionDirective,
    Program,
    TensorSpec,
    add,
    cast,
    conservative_precision_variants,
    describe_precision,
    execute,
    input_tensor,
    jvp,
    linearize,
    lower_precision,
    multiply,
    reduce_sum,
    transpose_program,
    vjp,
)
from vibeqc_compiler.tensor.cuda_emit import emit_cuda
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.cuda_search import estimate_schedule, plan_schedule_search


def _parameter(name: str, *, dtype: str = "float64") -> typing.Any:
    space = IndexSpace("values", "batch", 4)
    return input_tensor(
        name,
        TensorSpec(
            (Index("i", space),),
            dtype=dtype,
            role="parameter",
            differentiable=True,
        ),
    )


def _mixed_program() -> Program:
    x = _parameter("x")
    low = cast(x, "float32")
    squared = multiply(low, low)
    return Program({"out": cast(squared, "float64")})


def test_cast_is_explicit_typed_replayable_and_never_implicit() -> None:
    x = _parameter("x")
    low = cast(x, "float32")
    restored = cast(low, "float64")
    assert low.spec.dtype == "float32"
    assert restored.spec.dtype == "float64"
    assert low.spec.indices == x.spec.indices
    assert low.spec.representation == x.spec.representation
    assert low.spec.differentiable
    with pytest.raises(ValueError, match="dtype"):
        add(x, low)

    program = Program({"low": low, "restored": restored})
    feeds = {"x": np.array([1.0, 1.0 / 3.0, -2.5, 1e-20], dtype=np.float64)}
    result = execute(program, feeds).outputs
    expected = feeds["x"].astype(np.float32)
    np.testing.assert_array_equal(result["low"], expected)
    np.testing.assert_array_equal(result["restored"], expected.astype(np.float64))
    replay = Program.loads(program.dumps())
    assert replay.logical_hash == program.logical_hash
    np.testing.assert_array_equal(execute(replay, feeds).outputs["low"], expected)


def test_cast_jvp_vjp_and_generated_derivative_dags_preserve_boundaries() -> None:
    program = _mixed_program()
    feeds = {"x": np.array([0.75, -1.25, 2.0, 3.5], dtype=np.float64)}
    tangent = {"x": np.array([1.0, 0.5, -2.0, 0.25], dtype=np.float64)}
    cotangent = {"out": np.array([0.5, -1.0, 0.75, 2.0], dtype=np.float64)}

    forward = jvp(program, feeds, tangent)
    reverse = vjp(program, feeds, cotangent)
    generated_forward = linearize(program, ["x"])
    generated_reverse = transpose_program(program, ["out"], inputs=["x"])

    forward_feeds = {**feeds, "d_x": tangent["x"]}
    reverse_feeds = {**feeds, "bar_out": cotangent["out"]}
    actual_forward = execute(generated_forward.program, forward_feeds).outputs["d_out"]
    actual_reverse = execute(generated_reverse.program, reverse_feeds).outputs["bar_x"]
    np.testing.assert_array_equal(actual_forward, forward.output_tangents["out"])
    np.testing.assert_array_equal(actual_reverse, reverse.input_cotangents["x"])
    assert any(node.op == "cast" for node in generated_forward.program.live_nodes)
    assert any(node.op == "cast" for node in generated_reverse.program.live_nodes)


def test_precision_schedule_and_cuda_cost_identity_include_casts() -> None:
    program = _mixed_program()
    precision = describe_precision(program)
    assert precision.strict_audit_dtype == "float64"
    assert precision.math_mode == "ieee-rn-no-tf32"
    assert [(c.source_dtype, c.target_dtype) for c in precision.casts] == [
        ("float64", "float32"),
        ("float32", "float64"),
    ]
    assert precision.cast_read_bytes == 4 * 8 + 4 * 4
    assert precision.cast_write_bytes == 4 * 4 + 4 * 8
    assert precision.maximum_cast_live_bytes == 4 * (8 + 4)
    assert precision.to_payload()["promotion"].startswith("requires-independent")

    plan = plan_cuda(program, cuda_target_info("sm_80"))
    precision_schedule = plan.precision_schedule
    assert plan.precision_schedule is precision_schedule
    precision_name = plan.precision
    assert plan.__dict__["precision"] == precision_name
    payload = plan.to_payload()
    assert payload["precision"] == "typed-fp32-fp64"
    assert payload["precision_schedule_identity"] == precision.identity
    assert payload["arithmetic"].endswith("no TF32 or implicit casts")
    traffic = plan.semantic_traffic
    assert traffic["precision_cast_read_bytes"] == precision.cast_read_bytes
    assert traffic["precision_cast_write_bytes"] == precision.cast_write_bytes
    estimate = estimate_schedule(plan)
    assert estimate["precision_schedule_identity"] == precision.identity
    assert estimate["estimated_precision_cast_simultaneous_bytes"] == 48

    source = emit_cuda(plan)
    assert "__double2float_rn" in source
    assert "static_cast<double>" in source


def test_precision_directives_lower_mixed_subgraphs_and_fail_closed() -> None:
    x = _parameter("x")
    product = multiply(x, x)
    reduced = reduce_sum(product, (0,))
    program = Program({"out": reduced})
    product_name = program.debug_names[product]

    lowered = lower_precision(
        program,
        {
            product_name: PrecisionDirective(
                "float32",
                "float32",
                "float32",
            )
        },
    )
    assert lowered.provenance["precision_source_equation"] == program.logical_hash
    assert any(
        node.op == "multiply" and node.spec.dtype == "float32"
        for node in lowered.live_nodes
    )
    assert any(
        node.op == "reduce" and node.spec.dtype == "float64"
        for node in lowered.live_nodes
    )
    assert [
        (c.source_dtype, c.target_dtype) for c in describe_precision(lowered).casts
    ] == [
        ("float64", "float32"),
        ("float32", "float64"),
    ]

    with pytest.raises(ValueError, match="compute/accumulation"):
        lower_precision(
            program,
            {
                product_name: PrecisionDirective(
                    "float32",
                    "float32",
                    "float64",
                )
            },
        )
    reduction_name = program.debug_names[reduced]
    with pytest.raises(ValueError, match="qualification"):
        lower_precision(
            program,
            {
                reduction_name: PrecisionDirective(
                    "float32",
                    "float32",
                    "float32",
                )
            },
        )


def test_existing_schedule_search_can_cross_precision_variants() -> None:
    x = _parameter("x")
    product = multiply(x, x)
    program = Program({"out": reduce_sum(product, (0,))})
    variants = conservative_precision_variants(program)
    assert len(variants) == 2
    baseline = plan_cuda(program, cuda_target_info("sm_80"))
    proposals = plan_schedule_search(
        baseline,
        (TensorSchedule(),),
        precision_programs=variants,
    )
    assert len(proposals) == 2
    mixed = [
        proposal
        for proposal in proposals
        if proposal.plan is not None
        and proposal.plan.program.logical_hash == variants[1].logical_hash
    ]
    assert len(mixed) == 1
    assert mixed[0].plan is not None
    assert mixed[0].plan.precision == "typed-fp32-fp64"
    assert mixed[0].plan.precision_schedule.source_equation == program.logical_hash


def _qualified_reduction(scope: str) -> Program:
    x = _parameter("x")
    result = reduce_sum(x, (0,))
    program = Program({"out": result})
    return lower_precision(
        program,
        {
            program.debug_names[result]: PrecisionDirective(
                "float32",
                "float32",
                "float32",
                qualification=scope,
            )
        },
    )


def test_qualification_scope_is_part_of_schedule_plan_and_search_identity() -> None:
    from vibeqc_compiler.tensor.cuda_search import execution_key

    first = _qualified_reduction("workload-a/sm80/evidence-a")
    second = _qualified_reduction("workload-b/sm120/evidence-b")
    assert first.logical_hash == second.logical_hash
    assert describe_precision(first).identity != describe_precision(second).identity
    left = plan_cuda(first, cuda_target_info("sm_80"))
    right = plan_cuda(second, cuda_target_info("sm_80"))
    assert left.identity != right.identity
    assert execution_key(left) != execution_key(right)
    assert describe_precision(Program.loads(first.dumps())) == describe_precision(first)


def test_relowering_retains_parent_precision_qualification_scope() -> None:
    first = lower_precision(_qualified_reduction("evidence-a"), {})
    second = lower_precision(_qualified_reduction("evidence-b"), {})
    assert first.logical_hash == second.logical_hash
    assert describe_precision(first).identity != describe_precision(second).identity


def test_precision_request_integrity_is_checked_before_planning() -> None:
    program = _qualified_reduction("evidence-a")
    provenance = program.provenance
    provenance["precision_request_identity"] = "0" * 64
    altered = Program(program.outputs, program.definitions, provenance=provenance)
    with pytest.raises(ValueError, match="precision request.*identity"):
        describe_precision(altered)


def test_cast_ad_matches_explicit_round_trip_rule_independently() -> None:
    x = _parameter("x")
    program = Program({"out": cast(cast(x, "float32"), "float64")})
    primal = np.array([1 / 3, 1 + 2**-25, -1 / 7, 1e-20], dtype=np.float64)
    direction = np.array([1 / 11, -1 / 13, 1 + 2**-24, 1e-30], dtype=np.float64)
    seed = np.array([-1 / 3, 1 / 9, 1 - 2**-25, 1e-27], dtype=np.float64)
    # This is the declared cast tangent/cotangent contract, independently
    # evaluated with NumPy conversions, not another new AD implementation.
    expected_jvp = direction.astype(np.float32).astype(np.float64)
    expected_vjp = seed.astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(
        jvp(program, {"x": primal}, {"x": direction}).output_tangents["out"],
        expected_jvp,
    )
    np.testing.assert_array_equal(
        vjp(program, {"x": primal}, {"out": seed}).input_cotangents["x"], expected_vjp
    )
    forward = linearize(program, ["x"]).program
    reverse = transpose_program(program, ["out"], inputs=["x"]).program
    np.testing.assert_array_equal(
        execute(forward, {"x": primal, "d_x": direction}).outputs["d_out"], expected_jvp
    )
    np.testing.assert_array_equal(
        execute(reverse, {"x": primal, "bar_out": seed}).outputs["bar_x"], expected_vjp
    )

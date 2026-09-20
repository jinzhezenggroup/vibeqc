"""First-class TensorIR precision/cast contracts for #528."""

import numpy as np
import pytest
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    cast,
    describe_precision,
    execute,
    input_tensor,
    jvp,
    linearize,
    multiply,
    transpose_program,
    vjp,
)
from vibeqc_compiler.tensor.cuda_emit import emit_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda
from vibeqc_compiler.tensor.cuda_search import estimate_schedule


def _parameter(name: str, *, dtype: str = "float64"):
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

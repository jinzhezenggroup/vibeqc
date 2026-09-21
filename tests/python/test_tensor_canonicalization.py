"""IEEE-safe view canonicalization for the shared compiler optimizer."""

import numpy as np
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    broadcast,
    cast,
    constant,
    execute,
    input_tensor,
    multiply,
    reshape,
    rewrite,
    slice_tensor,
    transpose,
)
from vibeqc_compiler.tensor.ir import Node


def _matrix_input() -> Node:
    rows = Index("i", IndexSpace("rows", "batch", 2))
    cols = Index("j", IndexSpace("cols", "batch", 3))
    return input_tensor(
        "x", TensorSpec((rows, cols), role="parameter", differentiable=True)
    )


def test_view_canonicalization_collapses_lossless_chains_bitwise() -> None:
    x = _matrix_input()
    restored = transpose(transpose(x, (1, 0)), (1, 0))
    flat = Index("flat", IndexSpace("flat", "batch", 6))
    reshaped = reshape(reshape(restored, (flat,)), x.spec.indices)
    same_dtype = cast(reshaped, "float64")
    full_slice = slice_tensor(same_dtype, ((0, 2), (0, 3)))
    identity_broadcast = broadcast(full_slice, full_slice.spec.indices, (0, 1))
    program = Program({"out": identity_broadcast})

    rewritten = rewrite(program, "view_canonicalization")
    values = np.array([[0.0, -0.0, 1.25], [-3.5, 9.0, 7.25]], dtype=np.float64)
    before = execute(program, {"x": values}).outputs["out"]
    after = execute(rewritten, {"x": values}).outputs["out"]

    assert rewritten.outputs["out"] is x
    np.testing.assert_array_equal(before.view(np.uint64), after.view(np.uint64))


def test_view_canonicalization_composes_nontrivial_transposes() -> None:
    a = Index("a", IndexSpace("a", "batch", 2))
    b = Index("b", IndexSpace("b", "batch", 3))
    c = Index("c", IndexSpace("c", "batch", 4))
    x = input_tensor("x", TensorSpec((a, b, c), role="parameter"))
    first = transpose(x, (2, 0, 1))
    second = transpose(first, (2, 1, 0))

    rewritten = rewrite(Program({"out": second}), "view_canonicalization")
    out = rewritten.outputs["out"]

    assert out.op == "transpose"
    assert out.inputs == (x,)
    assert out.attrs["axes"] == (1, 0, 2)
    values = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)
    np.testing.assert_array_equal(
        execute(rewritten, {"x": values}).outputs["out"],
        values.transpose(1, 0, 2),
    )


def test_view_canonicalization_collapses_nested_reshape_to_one_view() -> None:
    x = _matrix_input()
    flat = Index("flat", IndexSpace("flat2", "batch", 6))
    rows = Index("r", IndexSpace("reshaped_rows", "batch", 3))
    cols = Index("c", IndexSpace("reshaped_cols", "batch", 2))
    nested = reshape(reshape(x, (flat,)), (rows, cols))

    rewritten = rewrite(Program({"out": nested}), "view_canonicalization")
    out = rewritten.outputs["out"]

    assert out.op == "reshape"
    assert out.inputs == (x,)
    values = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    np.testing.assert_array_equal(
        execute(rewritten, {"x": values}).outputs["out"], values.reshape(3, 2)
    )


def test_view_canonicalization_keeps_precision_roundtrip() -> None:
    x = _matrix_input()
    roundtrip = cast(cast(x, "float32"), "float64")
    rewritten = rewrite(Program({"out": roundtrip}), "view_canonicalization")
    casts = [node for node in rewritten.nodes if node.op == "cast"]

    assert len(casts) == 2
    assert rewritten.outputs["out"].op == "cast"


def test_view_canonicalization_does_not_apply_ieee_sensitive_arithmetic() -> None:
    x = input_tensor("x", TensorSpec(role="parameter"))
    product = multiply(x, constant(0))
    rewritten = rewrite(Program({"out": product}), "view_canonicalization")

    assert rewritten.outputs["out"] is product
    result = execute(rewritten, {"x": np.array(-0.0)}).outputs["out"]
    assert np.signbit(result)

"""Cross-axis ragged contracts against explicit-loop and adjoint references."""

from itertools import pairwise

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    execute,
    indexed_gather,
    input_tensor,
    jvp,
    linearize,
    scatter_add,
    segment_sum,
    transpose_program,
    vjp,
)
from vibeqc_compiler.tensor.cuda_emit import emit_cuda
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

CASES = [
    (rank, axis, kind, dtype, empty)
    for rank in (1, 2, 3)
    for axis in range(rank)
    for kind in ("indexed_gather", "scatter_add", "segment_sum")
    for dtype in ("float32", "float64")
    for empty in (False, True)
]


@pytest.mark.parametrize("rank,axis,kind,dtype,empty", CASES)
def test_ragged_cross_axis(
    rank: int, axis: int, kind: str, dtype: str, empty: bool
) -> None:
    shape = [2] * rank
    shape[axis] = 0 if empty else 5
    indices = tuple(
        Index(f"i{k}", IndexSpace(f"s{k}", "orbital", n)) for k, n in enumerate(shape)
    )
    value = input_tensor(
        "x", TensorSpec(indices, dtype=dtype, role="input", differentiable=True)
    )
    extent = (0 if empty else 6) if kind == "indexed_gather" else 4
    target = Index("target", IndexSpace("target_space", "atom", extent + 3), start=3)
    if kind == "indexed_gather":
        mapping = () if empty else (4, 1, 4, 0, 2, 1)
        node = indexed_gather(value, axis, mapping, target)
    elif kind == "scatter_add":
        mapping = () if empty else (3, 0, 3, 1, 0)
        node = scatter_add(value, axis, mapping, target)
    else:
        mapping = (0, 0, 0, 0, 0) if empty else (0, 2, 2, 3, 5)
        node = segment_sum(value, axis, mapping, target)
    program = Program({"y": node})
    array = np.asfortranarray(
        (np.arange(np.prod(shape), dtype=dtype).reshape(shape) + 1) / 8
    )
    tangent = np.full(shape, 0.25, dtype=dtype)
    output_shape = list(shape)
    output_shape[axis] = extent

    def oracle(data: np.ndarray) -> np.ndarray:
        result = np.zeros(output_shape, dtype=dtype)
        source = np.moveaxis(data, axis, 0)
        destination = np.moveaxis(result, axis, 0)
        if kind == "indexed_gather":
            for out, index in enumerate(mapping):
                destination[out] = source[index]
        elif kind == "scatter_add":
            for index, out in enumerate(mapping):
                destination[out] += source[index]
        else:
            for out, (start, stop) in enumerate(pairwise(mapping)):
                for index in range(start, stop):
                    destination[out] += source[index]
        return result

    feeds = {"x": array}
    expected = oracle(array)
    np.testing.assert_array_equal(execute(program, feeds).outputs["y"], expected)
    restored = Program.loads(program.dumps())
    assert restored.logical_hash == program.logical_hash
    np.testing.assert_array_equal(execute(restored, feeds).outputs["y"], expected)
    expected_jvp = oracle(tangent)
    np.testing.assert_array_equal(
        jvp(program, feeds, {"x": tangent}).output_tangents["y"], expected_jvp
    )
    forward = linearize(program, ["x"])
    np.testing.assert_array_equal(
        execute(forward.program, {"x": array, "d_x": tangent}).outputs["d_y"],
        expected_jvp,
    )
    cotangent = np.ones(output_shape, dtype=dtype) / 2
    reverse = transpose_program(program, ["y"], inputs=["x"])
    expected_vjp = vjp(program, feeds, {"y": cotangent}).input_cotangents["x"]
    np.testing.assert_array_equal(
        execute(reverse.program, {"x": array, "bar_y": cotangent}).outputs["bar_x"],
        expected_vjp,
    )
    assert float(np.sum(expected_jvp * cotangent)) == float(
        np.sum(tangent * expected_vjp)
    )
    for candidate in (program, forward.program, reverse.program):
        plan = plan_cuda(
            candidate, cuda_target_info("sm_120"), schedule=TensorSchedule()
        )
        assert emit_cuda(plan)

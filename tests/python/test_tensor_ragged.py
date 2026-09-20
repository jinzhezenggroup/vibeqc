"""Ragged/indexed TensorIR primitives for #501."""

import os
from pathlib import Path

import numpy as np
import pytest
from vibeqc.profiles import find_nvcc
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    dot_test,
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
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.interpreter import execute

TARGET = cuda_target_info("sm_120")


def _index(name: str, kind: str, size: int) -> Index:
    return Index(name, IndexSpace(name, kind, size))


def _program() -> Program:
    shell = _index("shell", "shell", 3)
    orbital = _index("orbital", "orbital", 5)
    atom = _index("atom", "atom", 3)
    shell_values = input_tensor(
        "shell_values", TensorSpec((shell,), role="input", differentiable=True)
    )
    orbital_values = input_tensor(
        "orbital_values", TensorSpec((orbital,), role="input", differentiable=True)
    )
    mapping = (0, 0, 1, 2, 2)
    return Program(
        {
            "gathered": indexed_gather(shell_values, 0, mapping, orbital),
            "scattered": scatter_add(orbital_values, 0, mapping, shell),
            "segmented": segment_sum(orbital_values, 0, (0, 2, 2, 5), atom),
        }
    )


def _feeds() -> dict[str, np.ndarray]:
    return {
        "shell_values": np.array([2.0, 3.0, 5.0]),
        "orbital_values": np.array([1.0, 2.0, 4.0, 8.0, 16.0]),
    }


def test_ragged_interpreter_roundtrip_and_empty_segment() -> None:
    program = _program()
    result = execute(program, _feeds()).outputs
    np.testing.assert_array_equal(result["gathered"], [2.0, 2.0, 3.0, 5.0, 5.0])
    np.testing.assert_array_equal(result["scattered"], [3.0, 4.0, 24.0])
    np.testing.assert_array_equal(result["segmented"], [3.0, 0.0, 28.0])

    replay = Program.loads(program.dumps())
    assert replay.logical_hash == program.logical_hash
    for name, value in execute(replay, _feeds()).outputs.items():
        np.testing.assert_array_equal(value, result[name])


def test_ragged_maps_fail_closed() -> None:
    shell = _index("shell", "shell", 3)
    orbital = _index("orbital", "orbital", 5)
    x = input_tensor("x", TensorSpec((shell,), role="input"))

    with pytest.raises(ValueError, match="map length"):
        indexed_gather(x, 0, (0, 1), orbital)
    with pytest.raises(ValueError, match="source axis"):
        indexed_gather(x, 0, (0, 1, 2, 3, 0), orbital)

    y = input_tensor("y", TensorSpec((orbital,), role="input"))
    with pytest.raises(ValueError, match="target axis"):
        scatter_add(y, 0, (0, 0, 1, 2, 3), shell)
    with pytest.raises(ValueError, match="offsets"):
        segment_sum(y, 0, (0, 2, 1, 5), shell)


def test_ragged_reference_and_generated_adjoint_agree() -> None:
    source = _index("source", "orbital", 5)
    segment = _index("segment", "atom", 3)
    x = input_tensor("x", TensorSpec((source,), role="input", differentiable=True))
    reduced = segment_sum(x, 0, (0, 2, 2, 5), segment)
    out = indexed_gather(reduced, 0, (2, 0, 2, 1, 0), source)
    program = Program({"out": out})
    feeds = {"x": np.array([0.5, -1.0, 2.0, 3.0, -0.25])}
    tangent = {"x": np.array([1.0, 0.5, -2.0, 3.0, 0.25])}
    cotangent = {"out": np.array([2.0, -1.0, 0.5, 4.0, -2.0])}

    assert dot_test(program, feeds, tangent, cotangent).passed

    reference_jvp = jvp(program, feeds, tangent).output_tangents["out"]
    forward = linearize(program, ["x"])
    generated_jvp = execute(forward.program, {**feeds, "d_x": tangent["x"]}).outputs[
        "d_out"
    ]
    np.testing.assert_allclose(generated_jvp, reference_jvp, atol=1e-14, rtol=0)

    reference_vjp = vjp(program, feeds, cotangent).input_cotangents["x"]
    reverse = transpose_program(program, ["out"], inputs=["x"])
    generated_vjp = execute(
        reverse.program, {**feeds, "bar_out": cotangent["out"]}
    ).outputs["bar_x"]
    np.testing.assert_allclose(generated_vjp, reference_vjp, atol=1e-14, rtol=0)
    assert any(node.op == "scatter_add" for node in reverse.program.live_nodes)


def test_ragged_cuda_plan_emits_device_side_maps_and_reductions() -> None:
    plan = plan_cuda(_program(), TARGET, schedule=TensorSchedule())
    source = emit_cuda(plan)
    assert len(plan.index_tables) == 3
    assert "index_data_" in source
    assert "reinterpret_cast<const I*>" in source
    assert "for (I r =" in source


@pytest.mark.skipif(
    os.environ.get("VIBEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
def test_ragged_cuda_matches_interpreter(tmp_path: Path) -> None:
    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_TENSOR_CUDA_TEST requires a CUDA compiler")
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )
    program = _program()
    plan = plan_cuda(program, compiler.target, schedule=TensorSchedule())
    expected = execute(program, _feeds()).outputs
    with PreparedCuda(plan, compile_cuda(plan, compiler, Path(tmp_path))) as prepared:
        actual = prepared.execute(_feeds()).outputs
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-14, rtol=0)

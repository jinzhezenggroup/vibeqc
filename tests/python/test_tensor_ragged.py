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


def _heterogeneous_batch_program() -> tuple[Program, dict[str, np.ndarray]]:
    # Two systems are concatenated into one program: (2 shells, 3 orbitals,
    # 2 atoms) + (3 shells, 5 orbitals, 3 atoms). No host per-system loop is
    # required by the generated CUDA executor.
    shell = _index("batch_shell", "shell", 5)
    orbital = _index("batch_orbital", "orbital", 8)
    atom = _index("batch_atom", "atom", 5)
    shell_values = input_tensor(
        "shell_values", TensorSpec((shell,), role="input", differentiable=True)
    )
    orbital_values = input_tensor(
        "orbital_values", TensorSpec((orbital,), role="input", differentiable=True)
    )
    orbital_to_shell = (0, 0, 1, 2, 3, 3, 4, 4)
    atom_offsets = (0, 2, 3, 5, 6, 8)
    program = Program(
        {
            "gathered": indexed_gather(shell_values, 0, orbital_to_shell, orbital),
            "scattered": scatter_add(orbital_values, 0, orbital_to_shell, shell),
            "segmented": segment_sum(orbital_values, 0, atom_offsets, atom),
        }
    )
    feeds = {
        "shell_values": np.array([2.0, 3.0, 5.0, 7.0, 11.0]),
        "orbital_values": np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
    }
    return program, feeds


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
    assert plan.index_table_bytes == 3 * 256
    assert plan.accumulation_workspace_bytes == 0
    assert plan.ragged_resources == {
        "index_table_bytes": 3 * 256,
        "accumulation_workspace_bytes": 0,
        "included_in_arena_bytes": True,
    }
    schedule = plan.batch_schedule
    by_op = {step.op: step for step in schedule.ragged_steps}
    scatter = by_op["scatter_add"]
    assert scatter.lowering == "inverted-segments"
    assert scatter.scan_work == 15
    assert scatter.scheduled_work == 5
    assert scatter.avoided_scan_work == 10
    assert scatter.max_degree == 2
    assert scatter.degree_histogram == ((1, 1), (2, 2))
    assert schedule.scan_work == 25
    assert schedule.scheduled_work == 15
    assert schedule.avoided_scan_work == 10
    first_materialized = min(step.offset for step in plan.steps if step.offset >= 0)
    assert first_materialized >= plan.index_table_bytes
    assert "index_data_" in source
    assert "reinterpret_cast<const I*>" in source
    assert "const I begin = index[" in source
    assert "for (I q = begin; q < end; ++q)" in source
    assert "for (I r =" in source  # segment_sum keeps its direct segment traversal
    assert "{0LL, 2LL, 3LL, 5LL, 0LL, 1LL, 2LL, 3LL, 4LL}" in source
    assert "if (reinterpret_cast<const I*>" not in source


def test_batch_schedule_counts_outer_ragged_work() -> None:
    shell = _index("shell_outer", "shell", 3)
    orbital = _index("orbital_outer", "orbital", 5)
    component = _index("component_outer", "component", 2)
    values = input_tensor("values", TensorSpec((orbital, component), role="input"))
    program = Program({"out": scatter_add(values, 0, (0, 0, 1, 2, 2), shell)})
    schedule = plan_cuda(program, TARGET).batch_schedule
    scatter = schedule.ragged_steps[0]
    assert scatter.scan_work == 30
    assert scatter.scheduled_work == 10
    assert scatter.avoided_scan_work == 20


def test_batch_schedule_tracks_homogeneous_batch_domains() -> None:
    batch = _index("systems", "batch", 4)
    x = input_tensor("x", TensorSpec((batch,), role="input"))
    schedule = plan_cuda(Program({"out": x}), TARGET).batch_schedule
    assert schedule.batch_domains == (("systems", 4),)
    assert schedule.ragged_steps == ()


def test_changed_ragged_topology_changes_program_and_plan_identity() -> None:
    shell = _index("shell", "shell", 3)
    orbital = _index("orbital", "orbital", 5)
    values = input_tensor(
        "shell_values", TensorSpec((shell,), role="input", differentiable=True)
    )
    baseline = Program(
        {
            "gathered": indexed_gather(values, 0, (0, 0, 1, 2, 2), orbital),
        }
    )
    changed = Program(
        {
            "gathered": indexed_gather(values, 0, (0, 1, 1, 2, 2), orbital),
        }
    )
    assert baseline.logical_hash != changed.logical_hash
    assert plan_cuda(baseline, TARGET).identity != plan_cuda(changed, TARGET).identity


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


@pytest.mark.skipif(
    os.environ.get("VIBEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
def test_heterogeneous_ragged_batch_cuda_matches_interpreter(tmp_path: Path) -> None:
    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_TENSOR_CUDA_TEST requires a CUDA compiler")
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )
    program, feeds = _heterogeneous_batch_program()
    plan = plan_cuda(program, compiler.target, schedule=TensorSchedule())
    expected = execute(program, feeds).outputs
    with PreparedCuda(plan, compile_cuda(plan, compiler, Path(tmp_path))) as prepared:
        actual = prepared.execute(feeds).outputs
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-14, rtol=0)

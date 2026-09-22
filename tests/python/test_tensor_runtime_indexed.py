"""Runtime-indexed TensorIR domain primitives for #783."""

import os
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    dot_test,
    execute,
    input_tensor,
    linearize,
    runtime_indexed_select,
    transpose_program,
)
from vibeqc_compiler.tensor.cuda_emit import emit_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda

TARGET = cuda_target_info("sm_120")


def _program() -> Program:
    virtual = IndexSpace("virtual_runtime", "virtual", 4)
    occupied = IndexSpace("occupied_runtime", "occupied", 2)
    domain = IndexSpace("runtime_domain", "batch", 3)
    a = Index("a", virtual)
    b = Index("b", virtual)
    i = Index("i", occupied)
    q = Index("q", domain)
    source = input_tensor(
        "source",
        TensorSpec((a, b, i), role="input", differentiable=True),
    )
    a_map = input_tensor(
        "a_map",
        TensorSpec((q,), dtype="int64", role="input"),
    )
    b_map = input_tensor(
        "b_map",
        TensorSpec((q,), dtype="int64", role="input"),
    )
    selected = runtime_indexed_select(source, ((0, a_map), (1, b_map)), q)
    return Program({"selected": selected})


def _feeds(
    a: tuple[int, int, int] = (3, 2, 1),
    b: tuple[int, int, int] = (1, 0, 1),
) -> dict[str, np.ndarray]:
    source = np.arange(4 * 4 * 2, dtype=np.float64).reshape(4, 4, 2)
    return {
        "source": source,
        "a_map": np.asarray(a, dtype=np.int64),
        "b_map": np.asarray(b, dtype=np.int64),
    }


def test_runtime_indexed_interpreter_and_replay() -> None:
    program = _program()
    feeds = _feeds()
    actual = execute(program, feeds).outputs["selected"]
    expected = np.stack(
        [
            feeds["source"][a, b]
            for a, b in zip(feeds["a_map"], feeds["b_map"], strict=True)
        ]
    )
    np.testing.assert_array_equal(actual, expected)

    changed = _feeds(a=(0, 0, 3), b=(3, 2, 0))
    changed_actual = execute(program, changed).outputs["selected"]
    changed_expected = np.stack(
        [
            changed["source"][a, b]
            for a, b in zip(changed["a_map"], changed["b_map"], strict=True)
        ]
    )
    np.testing.assert_array_equal(changed_actual, changed_expected)

    replay = Program.loads(program.dumps())
    assert replay.logical_hash == program.logical_hash
    np.testing.assert_array_equal(
        execute(replay, changed).outputs["selected"],
        changed_expected,
    )


def test_runtime_indexed_bounds_and_control_contract_fail_closed() -> None:
    program = _program()
    bad = _feeds(a=(0, 4, 1))
    with pytest.raises(ValueError, match="outside its source axis"):
        execute(program, bad)

    q = Index("q_bad", IndexSpace("runtime_bad", "batch", 3))
    with pytest.raises(ValueError, match="int64"):
        TensorSpec((q,), dtype="int64", role="input", differentiable=True)
    with pytest.raises(ValueError, match="immutable runtime control"):
        TensorSpec((q,), dtype="int64")


def test_runtime_indexed_cuda_plan_uses_runtime_maps_not_static_tables() -> None:
    program = _program()
    plan = plan_cuda(program, TARGET)
    source = emit_cuda(plan)
    assert plan.index_tables == ()
    assert plan.precision == "fp64"
    assert "runtime_index_0" in source
    assert "runtime_index_1" in source
    assert "tensor runtime index out of bounds at step" in source
    assert "static const I" not in source
    # The mathematical graph is independent of map values: changing runtime
    # coordinates requires no re-lowering or new artifact identity.
    assert program.logical_hash == _program().logical_hash


def test_runtime_indexed_reference_adjoint_matches_runtime_maps() -> None:
    program = _program()
    feeds = _feeds(a=(3, 1, 3), b=(0, 2, 0))
    tangent = {
        "source": np.linspace(-1.0, 1.0, feeds["source"].size).reshape(
            feeds["source"].shape
        )
    }
    cotangent = {"selected": np.arange(6, dtype=np.float64).reshape(3, 2)}
    result = dot_test(program, feeds, tangent, cotangent)
    assert result.passed, result


def test_runtime_indexed_generated_ad_matches_runtime_maps() -> None:
    program = _program()
    feeds = _feeds(a=(3, 1, 3), b=(0, 2, 0))
    tangent = np.linspace(-1.0, 1.0, feeds["source"].size).reshape(
        feeds["source"].shape
    )
    cotangent = np.arange(6, dtype=np.float64).reshape(3, 2)

    forward = linearize(program, ["source"])
    forward_result = execute(
        forward.program,
        {**feeds, "d_source": tangent},
    ).outputs["d_selected"]
    expected_forward = np.stack(
        [tangent[a, b] for a, b in zip(feeds["a_map"], feeds["b_map"], strict=True)]
    )
    np.testing.assert_array_equal(forward_result, expected_forward)

    reverse = transpose_program(program, ["selected"], inputs=["source"])
    reverse_result = execute(
        reverse.program,
        {**feeds, "bar_selected": cotangent},
    ).outputs["bar_source"]
    expected_reverse = np.zeros_like(feeds["source"])
    for lane, (a, b) in enumerate(zip(feeds["a_map"], feeds["b_map"], strict=True)):
        expected_reverse[a, b] += cotangent[lane]
    np.testing.assert_array_equal(reverse_result, expected_reverse)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
def test_runtime_indexed_cuda_replays_changed_maps_and_recovers_bounds(
    tmp_path: Path,
) -> None:
    from vibeqc.profiles import find_nvcc
    from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from vibeqc_compiler.tensor.cuda_resident import PreparedResident, compile_resident

    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_TENSOR_CUDA_TEST requires a CUDA compiler")
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )
    program = _program()
    plan = plan_cuda(program, compiler.target)
    initial = _feeds()
    changed = _feeds(a=(0, 0, 3), b=(3, 2, 0))
    bad = _feeds(a=(0, 4, 1))

    expected_initial = execute(program, initial).outputs["selected"]
    expected_changed = execute(program, changed).outputs["selected"]

    artifact = compile_cuda(plan, compiler, tmp_path)
    with PreparedCuda(plan, artifact) as prepared:
        np.testing.assert_array_equal(
            prepared.execute(initial).outputs["selected"], expected_initial
        )
        np.testing.assert_array_equal(
            prepared.execute(changed).outputs["selected"], expected_changed
        )
        with pytest.raises(RuntimeError, match="runtime index out of bounds"):
            prepared.execute(bad)
        np.testing.assert_array_equal(
            prepared.execute(initial).outputs["selected"], expected_initial
        )

    resident_artifact = compile_resident(plan, compiler, tmp_path)
    with PreparedResident(plan, resident_artifact) as resident:
        resident.upload(initial)
        leases, _ = resident.run()
        np.testing.assert_array_equal(
            resident.download(leases["selected"]), expected_initial
        )
        resident.upload(changed)
        leases, _ = resident.run()
        np.testing.assert_array_equal(
            resident.download(leases["selected"]), expected_changed
        )
        resident.upload(bad)
        with pytest.raises(RuntimeError, match="runtime index out of bounds"):
            resident.run()
        resident.upload(initial)
        leases, _ = resident.run()
        np.testing.assert_array_equal(
            resident.download(leases["selected"]), expected_initial
        )

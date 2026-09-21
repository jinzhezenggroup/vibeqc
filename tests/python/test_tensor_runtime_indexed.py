"""Runtime-indexed TensorIR domain primitives for #783."""

import numpy as np
import pytest
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    execute,
    input_tensor,
    linearize,
    runtime_indexed_select,
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


def _feeds(a=(3, 2, 1), b=(1, 0, 1)) -> dict[str, np.ndarray]:
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


def test_runtime_indexed_generated_ad_fails_closed_until_transpose_rule_lands() -> None:
    program = _program()
    with pytest.raises(
        ValueError, match="no demand-driven JVP rule.*runtime_indexed_select"
    ):
        linearize(program, ["source"])

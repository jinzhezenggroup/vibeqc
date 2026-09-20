"""Issue #633 explicit scientific-index lowering and compiler-integration tests."""

import numpy as np
import pytest
from vibeqc_compiler.array_api import capabilities, trace
from vibeqc_compiler.array_api import namespace as xp
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    broadcast,
    execute,
    gather,
    input_tensor,
    reshape,
    slice_tensor,
    transpose_program,
)
from vibeqc_compiler.tensor.cuda_plan import plan_cuda

TARGET = cuda_target_info("sm_80")


def test_explicit_reshape_matches_tensorir_and_roundtrips() -> None:
    row = IndexSpace("row", "batch", 2)
    column = IndexSpace("column", "history", 3)
    flat = IndexSpace("flat", "batch", 6)
    r, c, f = Index("r", row), Index("c", column), Index("f", flat)
    spec = TensorSpec((r, c), role="input")

    captured = trace(
        lambda x: xp.reshape(x, (6,), indices=(f,)),
        {"x": spec},
    )
    manual = Program(
        {"output": reshape(input_tensor("x", spec), (f,))},
    )
    assert captured.logical_hash == manual.logical_hash

    values = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(
        execute(captured, {"x": values}).outputs["output"],
        values.reshape(6),
    )
    replay = Program.loads(captured.dumps())
    assert replay.logical_hash == captured.logical_hash


def test_reshape_and_broadcast_reject_shape_only_semantics() -> None:
    ao = IndexSpace("ao", "ao", 3)
    batch = IndexSpace("batch", "batch", 2)
    spin = IndexSpace("spin", "spin", 2)
    b, s, p = Index("b", batch), Index("s", spin), Index("p", ao)
    vector = TensorSpec((b, p), role="input")

    with pytest.raises(ValueError, match="explicit TensorIR indices"):
        trace(lambda x: xp.reshape(x, (6,)), {"x": vector})
    with pytest.raises(ValueError, match="explicit TensorIR indices"):
        trace(lambda x: xp.broadcast_to(x, (2, 2, 3), axes=(0, 2)), {"x": vector})
    with pytest.raises(ValueError, match="explicit TensorIR indices and axes"):
        trace(
            lambda x: xp.broadcast_to(x, (2, 2, 3), indices=(b, s, p)),
            {"x": vector},
        )

    wrong = Index("i", IndexSpace("occupied", "occupied", 3))
    with pytest.raises(ValueError, match="preserve existing domains"):
        trace(
            lambda x: xp.broadcast_to(
                x,
                (2, 2, 3),
                indices=(b, s, wrong),
                axes=(0, 2),
            ),
            {"x": vector},
        )


def test_explicit_broadcast_matches_manual_tensorir() -> None:
    batch = IndexSpace("batch", "batch", 2)
    spin = IndexSpace("spin", "spin", 2)
    ao = IndexSpace("ao", "ao", 3)
    b, s, p, q = Index("b", batch), Index("s", spin), Index("p", ao), Index("q", ao)
    spec = TensorSpec((b, p, q), role="input")
    indices = (b, s, p, q)

    captured = trace(
        lambda x: xp.broadcast_to(
            x,
            (2, 2, 3, 3),
            indices=indices,
            axes=(0, 2, 3),
        ),
        {"x": spec},
    )
    manual = Program(
        {"output": broadcast(input_tensor("x", spec), indices, (0, 2, 3))},
    )
    assert captured.logical_hash == manual.logical_hash

    values = np.arange(18.0).reshape(2, 3, 3)
    expected = np.broadcast_to(values[:, None, :, :], (2, 2, 3, 3))
    np.testing.assert_array_equal(
        execute(captured, {"x": values}).outputs["output"],
        expected,
    )


def test_static_slice_and_take_preserve_tensorir_domains() -> None:
    ao = IndexSpace("ao", "ao", 5)
    p = Index("p", ao)
    spec = TensorSpec((p,), role="input")

    sliced = trace(lambda x: x[1:4], {"x": spec})
    manual_slice = Program(
        {"output": slice_tensor(input_tensor("x", spec), ((1, 4),))},
    )
    assert sliced.logical_hash == manual_slice.logical_hash

    taken = trace(lambda x: xp.take(x, (4, 1, 1), axis=-1), {"x": spec})
    manual_take = Program(
        {"output": gather(input_tensor("x", spec), 0, (4, 1, 1))},
    )
    assert taken.logical_hash == manual_take.logical_hash

    values = np.arange(5.0)
    np.testing.assert_array_equal(
        execute(sliced, {"x": values}).outputs["output"],
        values[1:4],
    )
    np.testing.assert_array_equal(
        execute(taken, {"x": values}).outputs["output"],
        values[[4, 1, 1]],
    )

    with pytest.raises(TypeError, match="rank-preserving"):
        trace(lambda x: x[1], {"x": spec})
    with pytest.raises(ValueError, match="nonnegative"):
        trace(lambda x: x[-2:], {"x": spec})
    with pytest.raises(ValueError, match="unit step"):
        trace(lambda x: x[::2], {"x": spec})


def test_captured_take_vjp_replay_and_cuda_plan_use_plain_tensorir() -> None:
    ao = IndexSpace("ao", "ao", 4)
    p = Index("p", ao)
    spec = TensorSpec((p,), role="parameter", differentiable=True)
    captured = trace(
        lambda x: xp.sum(xp.pow(xp.take(x, (3, 1, 1), axis=0), 2)),
        {"x": spec},
    )

    reverse = transpose_program(captured, ["output"], inputs=["x"]).program
    replay = Program.loads(reverse.dumps())
    values = np.array([1.0, 2.0, 3.0, 4.0])
    outputs = execute(
        replay,
        {"x": values, "bar_output": np.asarray(1.0)},
    ).outputs
    np.testing.assert_array_equal(outputs["bar_x"], np.array([0.0, 8.0, 0.0, 8.0]))

    direct_plan = plan_cuda(reverse, TARGET)
    replay_plan = plan_cuda(replay, TARGET)
    assert direct_plan.identity == replay_plan.identity
    assert all(step.node.op != "array_frontend" for step in direct_plan.steps)


def test_scf_density_builders_reuse_single_source_equations_through_frontend() -> None:
    from vibeqc_compiler.array_api.scf import (
        density_program as frontend_density_program,
    )
    from vibeqc_compiler.array_api.scf import (
        weighted_density_program as frontend_weighted_density_program,
    )
    from vibeqc_compiler.tensor import density_program as tensor_density_program
    from vibeqc_compiler.tensor import (
        weighted_density_program as tensor_weighted_density_program,
    )

    pairs = (
        (frontend_density_program(2, 3), tensor_density_program(2, 3)),
        (
            frontend_weighted_density_program(2, 3),
            tensor_weighted_density_program(2, 3),
        ),
    )
    for frontend, direct in pairs:
        assert frontend.logical_hash == direct.logical_hash
        assert frontend.provenance["array_frontend_version"] == 1
        assert frontend.provenance["construction"] == "array_frontend"
        assert all(node.op != "array_frontend" for node in frontend.live_nodes)
        assert Program.loads(frontend.dumps()).logical_hash == frontend.logical_hash


def test_capabilities_declare_explicit_index_surfaces() -> None:
    functions = set(capabilities()["functions"])
    assert {
        "reshape_explicit_indices",
        "broadcast_to_explicit_indices",
        "slice_static",
        "take_static",
    } <= functions

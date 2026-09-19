"""Dense symmetry adjoints use group projections, not incidence matrices."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    PackedLayout,
    Program,
    Symmetry,
    TensorSpec,
    execute,
    input_tensor,
    linearize,
    multiply,
    reduce_sum,
    transpose_program,
)


@pytest.mark.parametrize(
    "rank,generators",
    [
        (2, (Symmetry((1, 0)),)),
        (2, (Symmetry((1, 0), -1),)),
        (4, (Symmetry((1, 0, 3, 2)),)),
        (3, (Symmetry((1, 0, 2), -1), Symmetry((0, 2, 1), -1))),
        (4, (Symmetry((1, 0, 2, 3)), Symmetry((0, 1, 3, 2)), Symmetry((2, 3, 0, 1)))),
        (2, (Symmetry((0, 1), -1),)),
    ],
)
def test_dense_projection_matches_independent_orbits_and_directional_derivative(
    rank, generators
):
    rng = np.random.default_rng(152)
    space = IndexSpace("o", "occupied", 3)
    spec = TensorSpec(
        tuple(Index(f"i{k}", space) for k in range(rank)),
        symmetries=generators,
        role="parameter",
        differentiable=True,
    )
    layout = PackedLayout(spec)
    x = input_tensor("x", spec)
    # Duplicate SSA inputs must contribute before the final projection.
    second = input_tensor("x", spec)
    weights = input_tensor("w", replace(spec, symmetries=(), differentiable=False))
    primal = Program(
        {"energy": reduce_sum(multiply(multiply(x, second), weights), range(rank))}
    )
    value = layout.unpack(rng.normal(size=layout.size))
    direction = layout.unpack(rng.normal(size=layout.size))
    w = rng.normal(size=spec.shape)
    feeds = {"x": value, "w": w}
    forward = linearize(primal, ["x"])
    reverse = transpose_program(primal, ["energy"], inputs=["x"])
    tangent = execute(forward.program, {**feeds, "d_x": direction}).outputs["d_energy"]
    seed = np.asarray(1.25)
    actual = execute(reverse.program, {**feeds, "bar_energy": seed}).outputs["bar_x"]
    expected = layout.unpack(layout.unpack_transpose(2 * seed * value * w))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        np.sum(actual * direction), seed * tangent, atol=1e-11, rtol=1e-12
    )
    for step in (1e-4, 1e-5):
        fd = (
            np.sum((value + step * direction) ** 2 * w)
            - np.sum((value - step * direction) ** 2 * w)
        ) / (2 * step)
        np.testing.assert_allclose(tangent, fd, atol=1e-8, rtol=1e-8)
    replay = Program.loads(reverse.program.dumps())
    np.testing.assert_array_equal(
        execute(replay, {**feeds, "bar_energy": seed}).outputs["bar_x"], actual
    )
    assert not any(n.op in ("constant", "gather") for n in reverse.program.live_nodes)
    assert max(n.spec.size for n in reverse.program.live_nodes) <= spec.size


def test_asymmetric_forward_seed_is_rejected():
    space = IndexSpace("o", "occupied", 2)
    spec = TensorSpec(
        (Index("i", space), Index("j", space)),
        symmetries=(Symmetry((1, 0)),),
        role="parameter",
        differentiable=True,
    )
    x = input_tensor("x", spec)
    forward = linearize(Program({"out": multiply(x, x)}), ["x"])
    with pytest.raises(ValueError, match="symmetry"):
        execute(
            forward.program, {"x": np.eye(2), "d_x": np.array([[0.0, 1.0], [0.0, 0.0]])}
        )


def test_symbolic_symmetry_group_expansion_is_bounded():
    space = IndexSpace("o", "occupied", 1)
    generators = []
    for k in range(6):
        axes = list(range(7))
        axes[k], axes[k + 1] = axes[k + 1], axes[k]
        generators.append(Symmetry(tuple(axes)))
    spec = TensorSpec(
        tuple(Index(f"i{k}", space) for k in range(7)),
        symmetries=tuple(generators),
        role="parameter",
        differentiable=True,
    )
    x = input_tensor("x", spec)
    with pytest.raises(ValueError, match="4096 signed permutations"):
        transpose_program(Program({"out": x}), ["out"], inputs=["x"])

"""Shared small matrix-free residual for CPU and allocated-CUDA implicit tests."""

import numpy as np
from vibeqc_compiler.method import ImplicitSolveSpec
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    input_tensor,
    multiply,
    reduce_sum,
)


def rank_one_problem(size=17):
    i = Index("i", IndexSpace("coordinate", "batch", size))

    def value(name, active=False):
        return input_tensor(
            name, TensorSpec((i,), role="parameter", differentiable=active)
        )

    x, q = value("x", True), value("q", True)
    diagonal, u, v = value("diagonal"), value("u"), value("v")
    rank_one = multiply(u, broadcast(reduce_sum(multiply(v, x), (0,)), (i,), ()))
    residual = add(
        multiply(diagonal, x),
        rank_one,
        multiply(x, x),
        q,
        coefficients=(1, 1, "1/5", -1),
    )
    spec = ImplicitSolveSpec(
        Program({"residual": residual}),
        "x",
        ("q",),
        "diagonal-plus-rank-one-nonlinear",
        state_metric=tuple(float(1 + k % 3) for k in range(size)),
        residual_metric=tuple(float(2 + k % 4) for k in range(size)),
    )
    feeds = {
        "x": np.linspace(-0.2, 0.3, size),
        "diagonal": np.linspace(2.0, 3.0, size),
        "u": np.linspace(0.01, 0.1, size),
        "v": np.linspace(0.08, -0.05, size),
    }
    feeds["q"] = (
        feeds["diagonal"] * feeds["x"]
        + feeds["u"] * (feeds["v"] @ feeds["x"])
        + 0.2 * feeds["x"] ** 2
    )
    return spec, feeds

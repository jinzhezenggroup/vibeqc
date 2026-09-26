"""Explicit output pruning transports only the surviving precision bindings."""

import numpy as np
import pytest
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    execute,
    input_tensor,
    optimize,
    reduce_sum,
    transpose,
)
from vibeqc_compiler.tensor.precision import (
    PrecisionDirective,
    describe_precision,
    lower_precision,
)


@pytest.mark.parametrize("keep", ("mixed", "fp32"))
@pytest.mark.parametrize("with_view", (False, True))
def test_output_projection_preserves_bound_arithmetic(
    keep: str, with_view: bool
) -> None:
    x = input_tensor(
        "x",
        TensorSpec((Index("i", IndexSpace("prune", "batch", 4)),), role="parameter"),
    )
    source = transpose(x, (0,)) if with_view else x
    mixed, fp32 = reduce_sum(source, (0,)), reduce_sum(source, (0,))
    program = Program({"mixed": mixed, "fp32": fp32})
    lowered = lower_precision(
        program,
        {
            program.debug_names[mixed]: PrecisionDirective(
                "float32", "float32", "float64", qualification="review-mixed"
            ),
            program.debug_names[fp32]: PrecisionDirective(
                "float32", "float32", "float32", qualification="review-fp32"
            ),
        },
    )
    before = lowered.dumps()
    feeds = {"x": np.array([1e8, 1.0, -1e8, 1.0])}
    expected = 2.0 if keep == "mixed" else 1.0
    result = optimize(lowered, requested_outputs=(keep,))
    assert tuple(result.outputs) == (keep,)
    assert execute(result, feeds).outputs[keep] == expected
    assert execute(Program.loads(result.dumps()), feeds).outputs[keep] == expected
    assert lowered.dumps() == before
    reductions = [
        value for value in describe_precision(result).values if value.op == "reduce"
    ]
    assert len(reductions) == 1
    assert reductions[0].accumulation_dtype == (
        "float64" if keep == "mixed" else "float32"
    )

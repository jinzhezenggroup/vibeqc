"""Independent admission, qualification identity and cast-AD regressions."""

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_target import CUDA_TARGETS
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    PrecisionDirective,
    Program,
    TensorSpec,
    cast,
    describe_precision,
    execute,
    exp,
    input_tensor,
    linearize,
    lower_precision,
    multiply,
    reduce_sum,
    transpose_program,
)
from vibeqc_compiler.tensor.cuda_plan import plan_cuda
from vibeqc_compiler.tensor.cuda_search import execution_key


def _input(dtype: str = "float64") -> Node:
    return input_tensor(
        "x",
        TensorSpec(
            (Index("i", IndexSpace("ao", "ao", 3)),),
            dtype=dtype,
            role="parameter",
            differentiable=True,
        ),
    )


def test_exp_requires_explicit_qualification() -> None:
    node = exp(_input())
    program = Program({"out": node})
    values = {"x": np.full(3, 100.0)}
    assert np.isfinite(execute(program, values).outputs["out"]).all()
    precision = next(v for v in describe_precision(program).values if v.op == "exp")
    assert precision.sensitivity == "sensitive"
    with pytest.raises(ValueError, match="qualification"):
        lower_precision(
            program,
            {
                program.debug_names[node]: PrecisionDirective(
                    "float32", "float32", "float32"
                )
            },
        )


def _qualified(qualification: str) -> Program:
    x = _input()
    node = reduce_sum(multiply(x, x), (0,))
    program = Program({"out": node})
    return lower_precision(
        program,
        {
            program.debug_names[node]: PrecisionDirective(
                "float32", "float32", "float32", qualification=qualification
            )
        },
    )


@pytest.mark.parametrize("mode", ["primal", "jvp", "vjp", "vjp_of_jvp", "relowered"])
def test_distinct_evidence_partitions_precision_plan_and_dedup(mode: str) -> None:
    programs = [
        _qualified(q) for q in ("water-sm120-evidence-A", "iron-sm80-evidence-B")
    ]
    if mode == "jvp":
        programs = [linearize(p, ["x"]).program for p in programs]
    elif mode == "vjp":
        programs = [
            transpose_program(p, ["out"], inputs=["x"]).program for p in programs
        ]
    elif mode == "vjp_of_jvp":
        programs = [
            transpose_program(
                linearize(p, ["x"]).program, ["d_out"], inputs=["x"]
            ).program
            for p in programs
        ]
    elif mode == "relowered":
        programs = [lower_precision(p, {}) for p in programs]
    left, right = programs
    assert left.logical_hash == right.logical_hash
    assert describe_precision(left).identity != describe_precision(right).identity
    a, b = [plan_cuda(p, CUDA_TARGETS["sm_80"]) for p in programs]
    assert a.identity != b.identity
    assert execution_key(a) != execution_key(b)
    for p in programs:
        replay = Program.loads(p.dumps())
        assert describe_precision(replay).identity == describe_precision(p).identity


@pytest.mark.parametrize(
    ("source", "target"), [("float64", "float32"), ("float32", "float64")]
)
def test_cast_ad_matches_independent_dtype_conversion_contract(
    source: str, target: str
) -> None:
    program = Program({"out": cast(_input(source), target)})
    values = np.array([1.25, -2.75, 3.5], dtype=source)
    tangent = np.array([1.00000006, -2.00000013, 16777217.0], dtype=source)
    cotangent = np.array([1.00000006, -2.00000013, 16777217.0], dtype=target)
    forward = linearize(program, ["x"]).program
    reverse = transpose_program(program, ["out"], inputs=["x"]).program
    jvp = execute(forward, {"x": values, "d_x": tangent}).outputs["d_out"]
    vjp = execute(reverse, {"x": values, "bar_out": cotangent}).outputs["bar_x"]
    np.testing.assert_array_equal(jvp, tangent.astype(target))
    np.testing.assert_array_equal(vjp, cotangent.astype(source))
    assert jvp.dtype == np.dtype(target)
    assert vjp.dtype == np.dtype(source)


def test_precision_request_digest_is_checked_after_replay() -> None:
    program = _qualified("evidence-A")
    provenance = program.provenance
    for directive in provenance["precision_request"]["directives"].values():
        directive["qualification"] = "evidence-B"
    forged = Program(program.outputs, program.definitions, provenance=provenance)
    with pytest.raises(ValueError, match="precision request.*identity"):
        describe_precision(forged)


def test_strict_ad_does_not_acquire_precision_qualification() -> None:
    x = _input()
    program = Program({"out": reduce_sum(multiply(x, x), (0,))})
    for derivative in (
        linearize(program, ["x"]).program,
        transpose_program(program, ["out"], inputs=["x"]).program,
    ):
        assert "precision_parent_schedule_identity" not in derivative.provenance
        assert describe_precision(derivative).request_identity is None
        assert describe_precision(derivative).parent_schedule_identity is None


@pytest.mark.parametrize("op", ["scatter_add", "segment_sum"])
def test_ragged_accumulations_keep_reduction_sensitivity(op: str) -> None:
    from vibeqc_compiler.tensor.precision import _sensitivity

    assert _sensitivity(op) == "reduction"

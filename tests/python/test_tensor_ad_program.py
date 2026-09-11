"""Demand-driven TensorIR derivative DAG generation for #151 slice B."""

import numpy as np
import pytest
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    PackedLayout,
    Program,
    Symmetry,
    TensorSpec,
    add,
    broadcast,
    divide,
    dot_test,
    einsum,
    execute,
    gather,
    input_tensor,
    jvp,
    linearize,
    multiply,
    optimize,
    reduce_sum,
    reshape,
    slice_tensor,
    transpose,
    transpose_program,
    vjp,
)
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

RNG = np.random.default_rng(1151)
TARGET = cuda_target_info("sm_80")


def _axis(name, size, kind="occupied"):
    return Index(name, IndexSpace(name, kind, size))


def _parameter(name, indices, *, representation="general"):
    return input_tensor(
        name,
        TensorSpec(
            tuple(indices),
            role="parameter",
            differentiable=True,
            representation=representation,
        ),
    )


def _run_generated_jvp(program, feeds, tangents, outputs=None):
    generated = linearize(program, list(tangents), outputs=outputs)
    generated_feeds = dict(feeds)
    generated_feeds.update({f"d_{name}": value for name, value in tangents.items()})
    actual = execute(generated.program, generated_feeds).outputs
    reference = jvp(program, feeds, tangents, outputs=outputs)
    for name in reference.output_tangents:
        np.testing.assert_allclose(
            actual[f"d_{name}"], reference.output_tangents[name], rtol=1e-12, atol=1e-12
        )
    return generated, actual


def _run_generated_vjp(program, feeds, cotangents, inputs=None):
    generated = transpose_program(program, list(cotangents), inputs=inputs)
    generated_feeds = dict(feeds)
    generated_feeds.update({f"bar_{name}": value for name, value in cotangents.items()})
    actual = execute(generated.program, generated_feeds).outputs
    reference = vjp(program, feeds, cotangents, inputs=inputs)
    for name in reference.input_cotangents:
        np.testing.assert_allclose(
            actual[f"bar_{name}"],
            reference.input_cotangents[name],
            rtol=1e-12,
            atol=1e-12,
        )
    return generated, actual


def _supported_dense_program():
    i, j, k = _axis("i", 3), _axis("j", 4), _axis("k", 2)
    x, y, z = (
        _parameter("x", (i, j)),
        _parameter("y", (i, j)),
        _parameter("z", (j,)),
    )
    product = multiply(x, y)
    row_sum = reduce_sum(product, (1,))
    contracted = einsum("i,ij->j", row_sum, x)
    combined = add(contracted, z, coefficients=("1/2", "3/2"))
    output = divide(multiply(combined, combined), add(z, z, coefficients=(1, 1)))
    permuted = transpose(reshape(x, (i, j)), (1, 0))
    expanded = broadcast(row_sum, (i, k), (0,))
    program = Program(
        {
            "output": output,
            "permuted": permuted,
            "expanded": expanded,
        }
    )
    feeds = {
        "x": RNG.normal(size=(3, 4)),
        "y": RNG.normal(size=(3, 4)),
        "z": RNG.normal(size=4) + 3.0,
    }
    tangents = {
        "x": RNG.normal(size=(3, 4)),
        "y": RNG.normal(size=(3, 4)),
        "z": RNG.normal(size=4),
    }
    cotangents = {
        "output": RNG.normal(size=4),
        "permuted": RNG.normal(size=(4, 3)),
        "expanded": RNG.normal(size=(3, 2)),
    }
    return program, feeds, tangents, cotangents


def test_generated_jvp_and_vjp_match_the_interpreter_reference():
    program, feeds, tangents, cotangents = _supported_dense_program()
    forward, forward_outputs = _run_generated_jvp(program, feeds, tangents)
    reverse, reverse_outputs = _run_generated_vjp(program, feeds, cotangents)
    assert set(forward_outputs) == {"d_output", "d_permuted", "d_expanded"}
    assert set(reverse_outputs) == {"bar_x", "bar_y", "bar_z"}
    # The generated programs are themselves valid TensorIR equations.
    replay = Program.loads(forward.program.dumps())
    generated_feeds = {
        **feeds,
        **{f"d_{name}": value for name, value in tangents.items()},
    }
    np.testing.assert_allclose(
        execute(replay, generated_feeds).outputs["d_output"],
        forward_outputs["d_output"],
        rtol=1e-12,
        atol=1e-12,
    )
    replay = Program.loads(reverse.program.dumps())
    generated_feeds = {
        **feeds,
        **{f"bar_{name}": value for name, value in cotangents.items()},
    }
    np.testing.assert_allclose(
        execute(replay, generated_feeds).outputs["bar_x"],
        reverse_outputs["bar_x"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_generation_is_demand_driven_and_prunes_unrelated_branches():
    i = _axis("i", 3)
    x, y = _parameter("x", (i,)), _parameter("y", (i,))
    left, right = multiply(x, x), multiply(y, y)
    program = Program({"left": left, "right": right})
    forward = linearize(program, ["x"], outputs=["left"])
    live_inputs = {
        node.attrs["name"] for node in forward.program.live_nodes if node.op == "input"
    }
    assert live_inputs == {"x", "d_x"}
    assert "right" not in {name for name, _ in forward.program.outputs.items()}
    reverse = transpose_program(program, ["left"], inputs=["x"])
    live_inputs = {
        node.attrs["name"] for node in reverse.program.live_nodes if node.op == "input"
    }
    assert live_inputs == {"x", "bar_left"}


def test_generated_programs_link_primal_and_derivative_identity():
    program, _feeds, tangents, cotangents = _supported_dense_program()
    forward = linearize(program, list(tangents))
    reverse = transpose_program(program, list(cotangents))
    assert forward.primal_logical_hash == program.logical_hash
    assert reverse.primal_logical_hash == program.logical_hash
    assert forward.derivative_hash == forward.program.logical_hash
    assert reverse.derivative_hash == reverse.program.logical_hash
    assert forward.program.provenance["generation"] == "demand-driven"
    assert reverse.program.provenance["generation"] == "demand-driven"
    assert forward.provenance()["primal_logical_hash"] == program.logical_hash
    assert reverse.provenance()["primal_logical_hash"] == program.logical_hash


def test_optimize_before_and_after_generation_agree_numerically():
    program, feeds, tangents, cotangents = _supported_dense_program()
    forward = linearize(program, list(tangents))
    optimized_forward = optimize(forward.program)
    generated_feeds = {
        **feeds,
        **{f"d_{name}": value for name, value in tangents.items()},
    }
    expected = execute(forward.program, generated_feeds).outputs
    actual = execute(optimized_forward, generated_feeds).outputs
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12)

    optimized_primal = optimize(program)
    second = linearize(optimized_primal, list(tangents))
    actual = execute(second.program, generated_feeds).outputs
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12)

    reference = dot_test(program, feeds, tangents, cotangents)
    assert reference.passed


def test_generated_programs_are_cpu_plannable_for_cuda_lowering():
    program, _feeds, tangents, cotangents = _supported_dense_program()
    forward = linearize(program, list(tangents))
    reverse = transpose_program(program, list(cotangents))
    for generated in (forward.program, reverse.program):
        plan = plan_cuda(generated, TARGET)
        assert plan.program is generated
        assert plan.peak_bytes > 0


def test_generated_reverse_plan_supports_bounded_recomputation():
    program, _feeds, _tangents, cotangents = _supported_dense_program()
    reverse = transpose_program(program, list(cotangents))
    retained = plan_cuda(reverse.program, TARGET)
    recomputed = plan_cuda(
        reverse.program, TARGET, schedule=TensorSchedule(recompute=True)
    )
    assert recomputed.estimated_flops >= retained.estimated_flops
    assert recomputed.arena_bytes <= retained.arena_bytes
    bounded = plan_cuda(
        reverse.program,
        TARGET,
        schedule=TensorSchedule(recompute=True),
        max_bytes=recomputed.peak_bytes,
    )
    assert bounded.peak_bytes <= recomputed.peak_bytes


def test_generated_vjp_does_not_materialize_a_dense_jacobian():
    i = _axis("i", 2000)
    x = _parameter("x", (i,))
    program = Program({"out": multiply(x, x)})
    reverse = transpose_program(program, ["out"], inputs=["x"])
    # The reverse DAG is linear in the primal node count, not n^2.
    assert len(reverse.program.live_nodes) < 20
    plan = plan_cuda(reverse.program, TARGET, library_bytes=0, provider_bytes=0)
    assert plan.allocation_bytes < 1_000_000


def test_generated_reverse_slice_gather_and_repeated_einsum_match_reference():
    i = _axis("i", 4)
    x = _parameter("x", (i,))
    sliced = slice_tensor(x, ((1, 3),))
    gathered = gather(x, 0, (2, 0, 2))
    program = Program({"sliced": sliced, "gathered": gathered})
    feeds = {"x": RNG.normal(size=4)}
    cotangents = {
        "sliced": RNG.normal(size=2),
        "gathered": RNG.normal(size=3),
    }
    _run_generated_vjp(program, feeds, cotangents, inputs=["x"])

    virtual = IndexSpace("v", "virtual", 3)
    a, b = Index("a", virtual), Index("b", virtual)
    trace_input = _parameter("trace_input", (i, a, b))
    trace = einsum("ijj->i", trace_input)
    _run_generated_vjp(
        Program({"trace": trace}),
        {"trace_input": RNG.normal(size=(4, 3, 3))},
        {"trace": RNG.normal(size=4)},
        inputs=["trace_input"],
    )


@pytest.mark.parametrize(
    "equation,ranks",
    [
        ("ii->i", (2,)),
        ("iii->i", (3,)),
        ("ii,i->i", (2, 1)),
        ("ijj->i", (3,)),
        ("iij,j->ij", (3, 1)),
    ],
)
def test_diagonal_vjp_maps_labels_retained_in_the_output(equation, ranks):
    space = IndexSpace("o", "occupied", 3)
    operands = [
        _parameter(f"x{k}", tuple(Index(f"i{axis}", space) for axis in range(rank)))
        for k, rank in enumerate(ranks)
    ]
    program = Program({"out": einsum(equation, *operands, coefficient="3/2")})
    rng = np.random.default_rng(216)
    feeds = {
        f"x{k}": rng.normal(size=node.spec.shape) for k, node in enumerate(operands)
    }
    cotangent = rng.normal(size=program.outputs["out"].spec.shape)
    generated, actual = _run_generated_vjp(program, feeds, {"out": cotangent})
    if equation == "ii->i":
        np.testing.assert_array_equal(actual["bar_x0"], 1.5 * np.diag(cotangent))
    replay = Program.loads(generated.program.dumps())
    replayed = execute(replay, {**feeds, "bar_out": cotangent}).outputs
    for name, value in actual.items():
        np.testing.assert_array_equal(replayed[name], value)


def test_named_input_occurrences_seed_all_requested_reverse_paths():
    first, second = _parameter("x", ()), _parameter("x", ())
    program = Program(
        {"out": add(multiply(first, first), second), "unused": multiply(second, second)}
    )
    feeds = {"x": np.asarray(3.0)}
    for primal in (program, Program.loads(program.dumps()), optimize(program)):
        generated = transpose_program(primal, ["out"], inputs=["x"])
        for candidate in (
            generated.program,
            Program.loads(generated.program.dumps()),
            optimize(generated.program),
        ):
            actual = execute(candidate, {**feeds, "bar_out": np.asarray(2.0)}).outputs
            # The unused output has zero cotangent; both uses in out contribute.
            np.testing.assert_array_equal(actual["bar_x"], np.asarray(14.0))


@pytest.mark.parametrize("differentiate_matrix", [False, True])
def test_inactive_einsum_operand_does_not_consume_projection_budget(
    differentiate_matrix,
):
    space = IndexSpace("o", "occupied", 1001)
    matrix = input_tensor(
        "A",
        TensorSpec(
            (Index("i", space), Index("j", space)),
            role="parameter",
            differentiable=differentiate_matrix,
        ),
    )
    scalar = _parameter("x", ())
    program = Program({"out": einsum("ii,->", matrix, scalar)})
    # A diagonal projection would exceed the default budget. The selected
    # scalar derivative is just trace(A), whether A is differentiable or fixed.
    generated = transpose_program(program, ["out"], inputs=["x"])
    assert all(node.op != "constant" for node in generated.program.nodes)
    result = execute(
        generated.program, {"A": np.eye(1001), "bar_out": np.asarray(2.0)}
    ).outputs
    np.testing.assert_array_equal(result["bar_x"], np.asarray(2002.0))
    if differentiate_matrix:
        with pytest.raises(ValueError, match="element budget"):
            transpose_program(program, ["out"], inputs=["A"])


def test_unrequested_division_adjoint_is_never_constructed():
    numerator, denominator = _parameter("x", ()), _parameter("y", ())
    program = Program({"out": divide(numerator, denominator)})
    generated = transpose_program(program, ["out"], inputs=["x"])
    # The denominator derivative would need a square and product; they must
    # be absent even from retained definitions, not merely dead at execution.
    assert all(node.op != "multiply" for node in generated.program.nodes)
    actual = execute(
        generated.program, {"y": np.asarray(4.0), "bar_out": np.asarray(2.0)}
    )
    np.testing.assert_array_equal(actual.outputs["bar_x"], np.asarray(0.5))


def test_reverse_generation_fails_closed_for_symmetric_boundaries_and_budgets():
    i = _axis("i", 4)
    x = _parameter("x", (i,))
    sliced = slice_tensor(x, ((0, 4),))
    with pytest.raises(ValueError, match="element budget"):
        transpose_program(
            Program({"sliced": sliced}),
            ["sliced"],
            inputs=["x"],
            max_elements=0,
        )

    space = IndexSpace("o", "occupied", 2)
    symmetric = input_tensor(
        "symmetric",
        TensorSpec(
            (Index("p", space), Index("q", space)),
            symmetries=(Symmetry((1, 0), -1),),
            role="parameter",
            differentiable=True,
        ),
    )
    with pytest.raises(NotImplementedError, match="boundary"):
        linearize(Program({"symmetric": symmetric}), ["symmetric"])
    with pytest.raises(NotImplementedError, match="boundary"):
        transpose_program(
            Program({"symmetric": symmetric}),
            ["symmetric"],
            inputs=["symmetric"],
        )


@pytest.mark.parametrize("duplicate_input", [False, True])
def test_packed_inputs_use_the_weighted_unpack_adjoint(duplicate_input):
    space = IndexSpace("o", "occupied", 2)
    indices = (Index("i", space), Index("j", space))
    symmetric_spec = TensorSpec(
        indices,
        symmetries=(Symmetry((1, 0), -1),),
        representation="spin_orbital",
        role="parameter",
        differentiable=True,
    )
    layout = PackedLayout(symmetric_spec)
    packed_x = RNG.normal(size=layout.size)
    dense_x = layout.unpack(packed_x)
    packed_tangent = RNG.normal(size=layout.size)
    cotangent = np.asarray(RNG.normal())

    symmetric_input = input_tensor("x", symmetric_spec)
    second = input_tensor("x", symmetric_spec) if duplicate_input else symmetric_input
    program = Program(
        {"energy": reduce_sum(multiply(symmetric_input, second), (0, 1))},
        # Dead definitions also need consistent packed input semantics.
        definitions=(input_tensor("x", symmetric_spec),),
    )
    forward = linearize(program, ["x"], packed={"x": layout})
    generated_forward = execute(
        forward.program, {"x": packed_x, "d_x": packed_tangent}
    ).outputs["d_energy"]
    for step in (1e-4, 1e-5):
        plus = layout.unpack(packed_x + step * packed_tangent)
        minus = layout.unpack(packed_x - step * packed_tangent)
        finite_difference = (
            execute(program, {"x": plus}).outputs["energy"]
            - execute(program, {"x": minus}).outputs["energy"]
        ) / (2 * step)
        np.testing.assert_allclose(
            generated_forward, finite_difference, rtol=1e-6, atol=1e-8
        )

    general_input = input_tensor(
        "x", TensorSpec(indices, role="parameter", differentiable=True)
    )
    general_program = Program(
        {"energy": reduce_sum(multiply(general_input, general_input), (0, 1))}
    )
    dense_reference = vjp(general_program, {"x": dense_x}, {"energy": cotangent})
    expected_bar = layout.unpack_transpose(dense_reference.input_cotangents["x"])
    reverse = transpose_program(program, ["energy"], inputs=["x"], packed={"x": layout})
    generated_bar = execute(
        reverse.program, {"x": packed_x, "bar_energy": cotangent}
    ).outputs["bar_x"]
    np.testing.assert_allclose(generated_bar, expected_bar, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        float(cotangent * generated_forward),
        layout.inner_product(generated_bar, packed_tangent),
        rtol=1e-11,
        atol=1e-11,
    )


def test_cc_like_scalar_fixture_builds_cpu_derivative_references():
    from tools.tensor_ad_examples import cpu_references, fixture

    program, feeds, tangents = fixture(2, 3, seed=151)
    (
        forward,
        reverse,
        _forward_feeds,
        _reverse_feeds,
        forward_outputs,
        reverse_outputs,
        finite_differences,
        dot_error,
    ) = cpu_references(program, feeds, tangents, np.asarray(1.0))
    assert dot_error <= 1e-10
    assert forward_outputs["d_energy"].shape == ()
    assert set(reverse_outputs) == {"bar_t", "bar_f", "bar_g"}
    assert len(forward.program.live_nodes) > 0
    assert len(reverse.program.live_nodes) > 0
    assert all(sample["error"]["passed"] for sample in finite_differences)

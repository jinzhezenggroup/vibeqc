"""Primitive JVP/VJP rules and adjoint dot-product checks for #151 slice A."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc_compiler.tensor import (
    AD_PRIMITIVES,
    AD_RULE_VERSION,
    AD_RULES,
    DotTestResult,
    Index,
    IndexSpace,
    PackedLayout,
    Program,
    Symmetry,
    TensorSpec,
    add,
    broadcast,
    capabilities,
    divide,
    dot_test,
    einsum,
    execute,
    gather,
    input_tensor,
    jvp,
    multiply,
    optimize,
    reduce_sum,
    reshape,
    slice_tensor,
    transpose,
    vjp,
)
from vibeqc_compiler.tensor.autodiff import _JVP_RULES, _VJP_RULES
from vibeqc_compiler.tensor.ir import PRIMITIVES

RNG = np.random.default_rng(151)


def _axis(name, size, kind="occupied"):
    return Index(name, IndexSpace(name, kind, size))


def _parameter(name, indices, *, dtype="float64", representation="general"):
    return input_tensor(
        name,
        TensorSpec(
            tuple(indices),
            dtype=dtype,
            role="parameter",
            differentiable=True,
            representation=representation,
        ),
    )


def _directional_fd(program, feeds, tangents, output, step):
    """Central difference of one output along the supplied tangent direction."""
    plus, minus = dict(feeds), dict(feeds)
    for name, tangent in tangents.items():
        plus[name] = feeds[name] + step * tangent
        minus[name] = feeds[name] - step * tangent
    return (
        execute(program, plus).outputs[output] - execute(program, minus).outputs[output]
    ) / (2 * step)


def _check_case(program, feeds, tangents, cotangents, *, rtol=1e-10):
    """Require the dot identity and a finite-difference convergence region."""
    forward = jvp(program, feeds, tangents)
    reverse = vjp(program, feeds, cotangents)
    result = dot_test(program, feeds, tangents, cotangents, rtol=rtol)
    assert isinstance(result, DotTestResult)
    assert result.passed, result
    for output in cotangents:
        errors = []
        for step in (1e-3, 1e-4, 1e-5):
            finite_difference = _directional_fd(program, feeds, tangents, output, step)
            error = np.max(np.abs(forward.output_tangents[output] - finite_difference))
            scale = max(
                1.0,
                float(np.max(np.abs(finite_difference)))
                if finite_difference.size
                else 1.0,
            )
            # Coarse steps expose the O(h^2) truncation term; the smallest
            # step must be in the asymptotic region.
            assert error <= 1e-4 * scale, (step, error, scale)
            errors.append(error)
        assert min(errors) <= 1e-6, errors
        assert errors[-1] <= 1e-5, errors
    return forward, reverse, result


def test_primitive_rules_and_dot_products():
    i = _axis("i", 4)
    x, y = _parameter("x", (i,)), _parameter("y", (i,))
    xv, yv = RNG.normal(size=4), RNG.normal(size=4)
    tx, ty = RNG.normal(size=4), RNG.normal(size=4)
    w = RNG.normal(size=4)

    program = Program({"out": add(x, y, coefficients=("1/2", "3/4"))})
    forward, reverse, _ = _check_case(
        program,
        {"x": xv, "y": yv},
        {"x": tx, "y": ty},
        {"out": w},
    )
    np.testing.assert_allclose(forward.output_tangents["out"], 0.5 * tx + 0.75 * ty)
    np.testing.assert_allclose(reverse.input_cotangents["x"], 0.5 * w)
    np.testing.assert_allclose(reverse.input_cotangents["y"], 0.75 * w)

    program = Program({"out": multiply(x, y)})
    forward, reverse, _ = _check_case(
        program,
        {"x": xv, "y": yv},
        {"x": tx, "y": ty},
        {"out": w},
    )
    np.testing.assert_allclose(forward.output_tangents["out"], tx * yv + xv * ty)
    np.testing.assert_allclose(reverse.input_cotangents["x"], w * yv)
    np.testing.assert_allclose(reverse.input_cotangents["y"], w * xv)

    program = Program({"out": divide(x, y)})
    forward, reverse, _ = _check_case(
        program,
        {"x": xv, "y": yv},
        {"x": tx, "y": ty},
        {"out": w},
    )
    np.testing.assert_allclose(
        forward.output_tangents["out"], (tx * yv - xv * ty) / yv**2
    )
    np.testing.assert_allclose(reverse.input_cotangents["x"], w / yv)
    np.testing.assert_allclose(reverse.input_cotangents["y"], -w * xv / yv**2)

    j, k = _axis("j", 3), _axis("k", 4)
    matrix = _parameter("matrix", (j, k))
    value, tangent, cotangent = (
        RNG.normal(size=(3, 4)),
        RNG.normal(size=(3, 4)),
        RNG.normal(size=(4, 3)),
    )
    forward, reverse, _ = _check_case(
        Program({"out": transpose(matrix, (1, 0))}),
        {"matrix": value},
        {"matrix": tangent},
        {"out": cotangent},
    )
    np.testing.assert_allclose(forward.output_tangents["out"], tangent.T)
    np.testing.assert_allclose(reverse.input_cotangents["matrix"], cotangent.T)

    flat = _axis("flat", 12)
    forward, reverse, _ = _check_case(
        Program({"out": reshape(matrix, (flat,))}),
        {"matrix": value},
        {"matrix": tangent},
        {"out": cotangent.T.reshape(-1)},
    )
    np.testing.assert_allclose(forward.output_tangents["out"], tangent.reshape(-1))
    np.testing.assert_allclose(
        reverse.input_cotangents["matrix"], cotangent.T.reshape(-1).reshape(3, 4)
    )

    tile = slice_tensor(matrix, ((1, 3), (2, 4)))
    tile_cotangent = RNG.normal(size=(2, 2))
    forward, reverse, _ = _check_case(
        Program({"out": tile}),
        {"matrix": value},
        {"matrix": tangent},
        {"out": tile_cotangent},
    )
    expected = np.zeros((3, 4))
    expected[1:3, 2:4] = tile_cotangent
    np.testing.assert_allclose(reverse.input_cotangents["matrix"], expected)
    np.testing.assert_allclose(forward.output_tangents["out"], tangent[1:3, 2:4])

    vector = _parameter("vector", (j,))
    gathered = gather(vector, 0, (2, 0, 2, 1))
    vector_value = RNG.normal(size=3)
    vector_tangent = RNG.normal(size=3)
    gather_cotangent = RNG.normal(size=4)
    forward, reverse, _ = _check_case(
        Program({"out": gathered}),
        {"vector": vector_value},
        {"vector": vector_tangent},
        {"out": gather_cotangent},
    )
    expected = np.zeros(3)
    for source, target in enumerate((2, 0, 2, 1)):
        expected[target] += gather_cotangent[source]
    np.testing.assert_allclose(reverse.input_cotangents["vector"], expected)
    np.testing.assert_allclose(
        forward.output_tangents["out"],
        vector_tangent[[2, 0, 2, 1]],
    )

    block = _parameter("block", (i, j, k))
    reduced = reduce_sum(block, (1,))
    block_value = RNG.normal(size=(4, 3, 4))
    block_tangent = RNG.normal(size=(4, 3, 4))
    reduced_cotangent = RNG.normal(size=(4, 4))
    forward, reverse, _ = _check_case(
        Program({"out": reduced}),
        {"block": block_value},
        {"block": block_tangent},
        {"out": reduced_cotangent},
    )
    np.testing.assert_allclose(
        forward.output_tangents["out"], block_tangent.sum(axis=1)
    )
    np.testing.assert_allclose(
        reverse.input_cotangents["block"],
        np.broadcast_to(reduced_cotangent[:, None, :], block_value.shape),
    )

    batch = _axis("batch", 4, kind="batch")
    source_i, source_j = _axis("source_i", 2), _axis("source_j", 3)
    source = _parameter("source", (source_i, source_j))
    expanded = broadcast(source, (source_j, batch, source_i), (2, 0))
    expanded_cotangent = RNG.normal(size=(3, 4, 2))
    source_value = RNG.normal(size=(2, 3))
    source_tangent = RNG.normal(size=(2, 3))
    forward, reverse, _ = _check_case(
        Program({"out": expanded}),
        {"source": source_value},
        {"source": source_tangent},
        {"out": expanded_cotangent},
    )
    expected = np.zeros((2, 3))
    for iv in range(2):
        for jv in range(3):
            expected[iv, jv] = sum(expanded_cotangent[jv, bv, iv] for bv in range(4))
    np.testing.assert_allclose(reverse.input_cotangents["source"], expected)
    for jv in range(3):
        for bv in range(4):
            for iv in range(2):
                assert (
                    forward.output_tangents["out"][jv, bv, iv] == source_tangent[iv, jv]
                )


def test_einsum_matrix_product_matches_analytic_adjoints():
    p, auxiliary, q = _axis("p", 3), _axis("P", 4, kind="auxiliary"), _axis("q", 2)
    left = _parameter("left", (p, auxiliary))
    right = _parameter("right", (auxiliary, q))
    left_value, right_value = RNG.normal(size=(3, 4)), RNG.normal(size=(4, 2))
    left_tangent, right_tangent = RNG.normal(size=(3, 4)), RNG.normal(size=(4, 2))
    cotangent = RNG.normal(size=(3, 2))
    program = Program({"out": einsum("pP,Pq->pq", left, right)})
    forward, reverse, _ = _check_case(
        program,
        {"left": left_value, "right": right_value},
        {"left": left_tangent, "right": right_tangent},
        {"out": cotangent},
    )
    np.testing.assert_allclose(
        forward.output_tangents["out"],
        left_tangent @ right_value + left_value @ right_tangent,
    )
    np.testing.assert_allclose(
        reverse.input_cotangents["left"], cotangent @ right_value.T
    )
    np.testing.assert_allclose(
        reverse.input_cotangents["right"], left_value.T @ cotangent
    )


def test_repeated_einsum_labels_project_onto_the_diagonal():
    i = _axis("i", 2)
    virtual = IndexSpace("v", "virtual", 3)
    a, b = Index("a", virtual), Index("b", virtual)
    auxiliary = _axis("P", 4, kind="auxiliary")
    x = _parameter("x", (i, a, b, auxiliary))
    trace = einsum("ijjk->ik", x, coefficient="-3/4")
    value, tangent = RNG.normal(size=(2, 3, 3, 4)), RNG.normal(size=(2, 3, 3, 4))
    cotangent = RNG.normal(size=(2, 4))
    forward, reverse, _ = _check_case(
        Program({"out": trace}),
        {"x": value},
        {"x": tangent},
        {"out": cotangent},
    )
    expected = np.zeros_like(value)
    for iv in range(2):
        for jv in range(3):
            for kv in range(4):
                expected[iv, jv, jv, kv] = -0.75 * cotangent[iv, kv]
    np.testing.assert_allclose(reverse.input_cotangents["x"], expected)
    expected_forward = np.zeros((2, 4))
    for iv in range(2):
        for kv in range(4):
            expected_forward[iv, kv] = -0.75 * sum(
                tangent[iv, jv, jv, kv] for jv in range(3)
            )
    np.testing.assert_allclose(forward.output_tangents["out"], expected_forward)


def test_random_dag_with_multiple_consumers_and_views():
    i, j = _axis("i", 2), _axis("j", 3)
    x, y, z = (
        _parameter("x", (i, j)),
        _parameter("y", (i, j)),
        _parameter("z", (j,)),
    )
    product = multiply(x, y)
    row_sum = reduce_sum(product, (1,))
    contracted = einsum("i,ij->j", row_sum, x)
    combined = add(contracted, z, coefficients=("1/2", "3/2"))
    output = multiply(combined, combined)
    scalar = einsum("ji,ij->", transpose(x, (1, 0)), y)
    tile = slice_tensor(x, ((0, 2), (1, 3)))
    selected = gather(tile, 1, (1, 0, 1))
    reduced = reduce_sum(selected, (1,))
    batch = _axis("batch", 4, kind="batch")
    expanded = broadcast(reduced, (i, batch), (0,))
    program = Program({"out": output, "scalar": scalar, "expanded": expanded})
    feeds = {
        "x": RNG.normal(size=(2, 3)),
        "y": RNG.normal(size=(2, 3)),
        "z": RNG.normal(size=3),
    }
    tangents = {
        "x": RNG.normal(size=(2, 3)),
        "y": RNG.normal(size=(2, 3)),
        "z": RNG.normal(size=3),
    }
    cotangents = {
        "out": RNG.normal(size=3),
        "scalar": RNG.normal(size=()),
        "expanded": RNG.normal(size=(2, 4)),
    }
    _check_case(program, feeds, tangents, cotangents)


def test_vjp_accumulates_cotangents_from_the_same_output_node():
    i = _axis("i", 5)
    x = _parameter("x", (i,))
    value = RNG.normal(size=5)
    output = multiply(x, x)
    program = Program({"a": output, "b": output})
    cotangent = np.ones(5)
    result = vjp(program, {"x": value}, {"a": cotangent, "b": 2 * cotangent})
    np.testing.assert_allclose(result.input_cotangents["x"], 6 * value)


@pytest.mark.parametrize("scalar", [False, True])
def test_distinct_inputs_with_one_name_accumulate_before_and_after_cse(scalar):
    indices = () if scalar else (_axis("i", 3),)
    first = _parameter("x", indices)
    second = _parameter("x", indices)
    program = Program({"out": add(multiply(first, first), second)})
    value = np.asarray(3.0) if scalar else np.array([1.0, 2.0, 3.0])
    tangent = np.ones_like(value)
    cotangent = np.full_like(value, 2.0)
    # Both input nodes read x: d(x*x + x)/dx = 2*x + 1. CSE must not
    # change this result, even though it can merge the input definitions.
    for candidate in (program, Program.loads(program.dumps()), optimize(program)):
        forward, reverse, _ = _check_case(
            candidate, {"x": value}, {"x": tangent}, {"out": cotangent}
        )
        np.testing.assert_array_equal(forward.output_tangents["out"], 2 * value + 1)
        np.testing.assert_array_equal(
            reverse.input_cotangents["x"], (2 * value + 1) * cotangent
        )


def test_matrix_free_vjp_fits_a_budget_far_below_the_dense_jacobian():
    i = _axis("i", 2000)
    x = _parameter("x", (i,))
    value = RNG.normal(size=2000)
    program = Program({"out": multiply(x, x)})
    # A dense Jacobian would require 2000**2 * 8 bytes (about 32 MB).
    budget = 1_000_000
    result = vjp(
        program,
        {"x": value},
        {"out": np.ones(2000)},
        max_bytes=budget,
    )
    np.testing.assert_allclose(result.input_cotangents["x"], 2 * value)
    with pytest.raises(ValueError, match="budget"):
        jvp(program, {"x": value}, {"x": np.ones(2000)}, max_bytes=100)


def test_packed_symmetric_tangent_spaces_fail_closed():
    space = IndexSpace("o", "occupied", 2)
    spec = TensorSpec(
        (Index("i", space), Index("j", space)),
        symmetries=(Symmetry((1, 0), -1),),
        representation="spin_orbital",
        role="parameter",
        differentiable=True,
    )
    x = input_tensor("x", spec)
    program = Program({"out": x})
    value = np.zeros((2, 2))
    with pytest.raises(ValueError, match="packed/symmetric tangent"):
        jvp(program, {"x": value}, {"x": np.ones((2, 2))})
    with pytest.raises(ValueError, match="packed/symmetric cotangent"):
        vjp(program, {"x": value}, {"out": np.ones((2, 2))})


def test_packed_layout_transposes_follow_the_weighted_metric():
    occupied = IndexSpace("o", "occupied", 2)
    virtual = IndexSpace("v", "virtual", 3)
    indices = (
        Index("i", occupied),
        Index("j", occupied),
        Index("a", virtual),
        Index("b", virtual),
    )
    spatial = TensorSpec(
        indices,
        representation="restricted_spatial",
        role="input",
        symmetries=(Symmetry((1, 0, 3, 2)),),
    )
    spin = replace(
        spatial,
        representation="spin_orbital",
        symmetries=(
            Symmetry((1, 0, 2, 3), -1),
            Symmetry((0, 1, 3, 2), -1),
        ),
    )
    for layout in (PackedLayout(spatial), PackedLayout(spin)):
        packed_x = RNG.normal(size=layout.size)
        dense_w = RNG.normal(size=layout.spec.shape)
        lhs = float(np.sum(dense_w * layout.unpack(packed_x)))
        rhs = layout.inner_product(layout.unpack_transpose(dense_w), packed_x)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-11, atol=1e-11)

        dense_y = layout.unpack(RNG.normal(size=layout.size))
        packed_x2 = RNG.normal(size=layout.size)
        lhs = layout.inner_product(layout.pack(dense_y), packed_x2)
        rhs = float(np.sum(dense_y * layout.pack_transpose(packed_x2)))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-11, atol=1e-11)

        # On the symmetric subspace unpack^T coincides with pack, but a general
        # dense cotangent is not projected by ordinary representative selection.
        raw_pack = dense_w.reshape(-1)[list(layout.representatives)]
        assert not np.allclose(layout.unpack_transpose(dense_w), raw_pack)
        replay = PackedLayout.from_payload(layout.to_payload())
        np.testing.assert_allclose(
            replay.unpack_transpose(dense_w), layout.unpack_transpose(dense_w)
        )
        np.testing.assert_allclose(
            replay.pack_transpose(packed_x2), layout.pack_transpose(packed_x2)
        )


def test_nondifferentiable_inputs_and_outputs_are_rejected():
    i = _axis("i", 3)
    x = input_tensor("x", TensorSpec((i,), role="input"))
    program = Program({"out": x})
    with pytest.raises(ValueError, match="not declared differentiable"):
        jvp(program, {"x": np.ones(3)}, {"x": np.ones(3)})
    with pytest.raises(ValueError, match="not differentiable"):
        vjp(program, {"x": np.ones(3)}, {"out": np.ones(3)})
    with pytest.raises(ValueError, match="unknown tangent"):
        jvp(program, {"x": np.ones(3)}, {"missing": np.ones(3)})
    with pytest.raises(ValueError, match="unknown cotangent"):
        vjp(program, {"x": np.ones(3)}, {"missing": np.ones(3)})


def test_provenance_links_the_derivative_to_the_primal_equation():
    i = _axis("i", 3)
    x = _parameter("x", (i,))
    program = Program({"out": multiply(x, x)})
    value, tangent, cotangent = (
        RNG.normal(size=3),
        RNG.normal(size=3),
        RNG.normal(size=3),
    )
    forward = jvp(program, {"x": value}, {"x": tangent})
    reverse = vjp(program, {"x": value}, {"out": cotangent})
    assert forward.primal_logical_hash == program.logical_hash
    assert reverse.primal_logical_hash == program.logical_hash
    assert (
        forward.derivative_hash
        == jvp(program, {"x": value}, {"x": tangent}).derivative_hash
    )
    assert forward.provenance()["primal_logical_hash"] == program.logical_hash
    assert reverse.provenance()["mode"] == "vjp"
    different = Program({"out": add(x, x)})
    assert (
        jvp(different, {"x": value}, {"x": tangent}).derivative_hash
        != forward.derivative_hash
    )


def test_requested_outputs_and_inputs_are_isolated_and_hashed():
    i = _axis("i", 3)
    x, y = _parameter("x", (i,)), _parameter("y", (i,))
    program = Program({"sum": add(x, y), "product": multiply(x, y)})
    feeds = {"x": RNG.normal(size=3), "y": RNG.normal(size=3)}
    tangents = {"x": RNG.normal(size=3)}
    cotangents = {"sum": RNG.normal(size=3)}
    forward = jvp(program, feeds, tangents, outputs=["sum"])
    assert set(forward.output_tangents) == {"sum"}
    assert forward.output_names == ("sum",)
    reverse = vjp(program, feeds, cotangents, inputs=["y"])
    assert set(reverse.input_cotangents) == {"y"}
    assert reverse.input_names == ("y",)
    other = jvp(program, feeds, tangents, outputs=["product"])
    assert other.derivative_hash != forward.derivative_hash
    with pytest.raises(ValueError, match="unknown output"):
        jvp(program, feeds, tangents, outputs=["missing"])
    with pytest.raises(ValueError, match="unknown input"):
        vjp(program, feeds, cotangents, inputs=["missing"])
    with pytest.raises(TypeError, match="not a string"):
        jvp(program, feeds, tangents, outputs="sum")


def test_capabilities_cover_every_primitive_and_claim_only_slice_a():
    report = capabilities()
    assert set(report["primitives"]) == set(PRIMITIVES) == set(AD_PRIMITIVES)
    assert set(AD_RULES) == set(_JVP_RULES) == set(_VJP_RULES) == set(PRIMITIVES)
    assert report["rule_version"] == AD_RULE_VERSION
    assert report["packed_symmetry"] is False
    assert report["derivative_dag_generation"] is False
    assert report["bounded_recomputation"] is False
    assert report["cuda"] is False


def test_float32_dot_test_uses_a_scale_aware_absolute_guard():
    i = _axis("i", 4)
    x = _parameter("x", (i,), dtype="float32")
    program = Program({"out": multiply(x, x)})
    feeds = {"x": np.array([0.5, -1.5, 2.0, 0.25], dtype=np.float32)}
    tangents = {"x": np.array([1.0, 0.5, -2.0, 3.0], dtype=np.float32)}
    cotangents = {"out": np.array([2.0, -1.0, 0.5, 4.0], dtype=np.float32)}
    result = dot_test(program, feeds, tangents, cotangents, rtol=1e-5)
    assert result.passed, result
    assert result.atol == pytest.approx(1e-6)


def test_zero_and_singleton_shapes_keep_the_adjoint_identity():
    empty = _axis("empty", 0)
    single = _axis("single", 1)
    x0 = _parameter("x0", (empty,))
    x1 = _parameter("x1", (single,))
    program = Program(
        {
            "zero": einsum("i,i->", x0, x0),
            "one": multiply(x1, x1),
        }
    )
    feeds = {"x0": np.empty(0), "x1": np.array([2.0])}
    tangents = {"x0": np.empty(0), "x1": np.array([3.0])}
    cotangents = {"zero": np.array(1.0), "one": np.array([4.0])}
    forward, reverse, result = _check_case(program, feeds, tangents, cotangents)
    assert result.lhs == pytest.approx(result.rhs)
    assert forward.output_tangents["zero"].shape == ()
    assert reverse.input_cotangents["x0"].shape == (0,)
    np.testing.assert_allclose(reverse.input_cotangents["x1"], [16.0])

"""Independent scalar-function and generated derivative qualification (#500)."""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    Symmetry,
    TensorSpec,
    add,
    dot_test,
    execute,
    exp,
    input_tensor,
    jvp,
    linearize,
    log,
    optimize,
    power,
    reduce_sum,
    sqrt,
    transpose_program,
    vjp,
)

CASES = [("exp", None), ("log", None), ("sqrt", None)] + [
    ("power", Fraction(p)) for p in (0, 1, 2, -2, "3/2", "1/3")
]


def _input(shape=(), dtype="float64"):
    axes = tuple(
        Index(f"i{k}", IndexSpace(f"axis{k}", "batch", n)) for k, n in enumerate(shape)
    )
    return input_tensor(
        "x", TensorSpec(axes, dtype=dtype, role="parameter", differentiable=True)
    )


def _operation(op, x, exponent=None):
    return (
        power(x, exponent)
        if op == "power"
        else {"exp": exp, "log": log, "sqrt": sqrt}[op](x)
    )


def _oracle(op, value, exponent=None, order=0, weight=1.0):
    # Decimal transcendental arithmetic is independent of NumPy and CUDA libm.
    with localcontext() as ctx:
        ctx.prec = 80
        x = Decimal.from_float(float(value))
        if op == "exp":
            result = x.exp()
        elif op == "log":
            result = x.ln() if order == 0 else 1 / x
        elif op == "sqrt":
            result = x.sqrt() if order == 0 else 1 / (2 * x.sqrt())
        else:
            p = Decimal(exponent.numerator) / Decimal(exponent.denominator)
            result = x**p if order == 0 else p * x ** (p - 1)
        return float(result * Decimal.from_float(weight))


@pytest.mark.parametrize("op,exponent", CASES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (0,)])
def test_values_and_both_ad_modes_against_decimal(op, exponent, dtype, shape):
    x = _input(shape, dtype)
    program = Program({"out": _operation(op, x, exponent)})
    n = x.spec.size
    # The nonempty array cases deliberately have noncontiguous storage.
    values = np.linspace(0.25, 1.75, max(1, 2 * n), dtype=dtype)[::2][:n].reshape(shape)
    seed = np.full(shape, 0.75, dtype=dtype)
    expected = np.array(
        [_oracle(op, v, exponent) for v in values.flat], dtype=dtype
    ).reshape(shape)
    slope = np.array(
        [_oracle(op, v, exponent, 1) for v in values.flat], dtype=dtype
    ).reshape(shape)
    tolerance = 32 * np.finfo(dtype).eps
    feeds = {"x": values}
    for candidate in (program, Program.loads(program.dumps()), optimize(program)):
        assert candidate.outputs["out"].spec.dtype == dtype
        np.testing.assert_allclose(
            execute(candidate, feeds).outputs["out"],
            expected,
            rtol=tolerance,
            atol=np.finfo(dtype).smallest_subnormal,
        )
    np.testing.assert_allclose(
        jvp(program, feeds, {"x": seed}).output_tangents["out"],
        seed * slope,
        rtol=tolerance,
        atol=0,
    )
    np.testing.assert_allclose(
        vjp(program, feeds, {"out": seed}).input_cotangents["x"],
        seed * slope,
        rtol=tolerance,
        atol=0,
    )
    assert dot_test(program, feeds, {"x": seed}, {"out": seed}, rtol=tolerance).passed
    for derivative, seed_name, output_name in (
        (linearize(program, ["x"]).program, "d_x", "d_out"),
        (transpose_program(program, ["out"]).program, "bar_out", "bar_x"),
    ):
        for candidate in (derivative, optimize(Program.loads(derivative.dumps()))):
            actual = execute(candidate, {**feeds, seed_name: seed}).outputs[output_name]
            assert actual.dtype == np.dtype(dtype)
            np.testing.assert_allclose(actual, seed * slope, rtol=tolerance, atol=0)


@pytest.mark.parametrize("op,exponent", CASES)
def test_multistep_finite_differences(op, exponent):
    program = Program({"out": _operation(op, _input((4,)), exponent)})
    x = np.array([0.05, 0.5, 1.0, 2.0])
    direction = x * np.array([0.2, -0.3, 0.4, -0.1])
    derivative = jvp(program, {"x": x}, {"x": direction}).output_tangents["out"]
    for h in (1e-4, 3e-5, 1e-5):
        plus = execute(program, {"x": x + h * direction}).outputs["out"]
        minus = execute(program, {"x": x - h * direction}).outputs["out"]
        np.testing.assert_allclose(
            (plus - minus) / (2 * h), derivative, rtol=3e-8, atol=1e-9
        )


@pytest.mark.parametrize(
    "op,exponent,bad",
    [
        ("log", None, -1.0),
        ("log", None, 0.0),
        ("sqrt", None, -1.0),
        ("exp", None, 1000.0),
        ("power", Fraction(2), 1e200),
    ]
    + [("power", Fraction(p), bad) for p in (0, 1, 2, "1/2") for bad in (-1.0, 0.0)],
)
def test_invalid_primal_not_erased_from_generated_derivatives(op, exponent, bad):
    program = Program({"out": _operation(op, _input(), exponent)})
    feeds = {"x": np.array(bad)}
    for seed in (np.array(0.0), np.array(1.0)):
        with pytest.raises(ValueError):
            jvp(program, feeds, {"x": seed})
        with pytest.raises(ValueError):
            vjp(program, feeds, {"out": seed})
        for candidate, extra in (
            (program, {}),
            (linearize(program, ["x"]).program, {"d_x": seed}),
            (transpose_program(program, ["out"]).program, {"bar_out": seed}),
        ):
            for replay in (candidate, optimize(Program.loads(candidate.dumps()))):
                with pytest.raises(ValueError, match="domain|non-finite|division"):
                    execute(replay, {**feeds, **extra})


@pytest.mark.parametrize("zero", [0.0, -0.0])
def test_sqrt_zero_value_is_legal_but_derivative_is_singular(zero):
    program = Program({"out": sqrt(_input())})
    feeds = {"x": np.array(zero)}
    assert execute(program, feeds).outputs["out"] == 0
    for seed in (np.array(0.0), np.array(1.0)):
        with pytest.raises(ValueError, match="strictly positive"):
            jvp(program, feeds, {"x": seed})
        with pytest.raises(ValueError, match="strictly positive"):
            vjp(program, feeds, {"out": seed})
        for p, key in (
            (linearize(program, ["x"]).program, "d_x"),
            (transpose_program(program, ["out"]).program, "bar_out"),
        ):
            with pytest.raises(ValueError, match="division"):
                execute(optimize(p), {**feeds, key: seed})


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_seeded_derivatives_near_floating_point_boundaries(dtype):
    typ = np.dtype(dtype).type
    tiny = np.finfo(dtype).smallest_subnormal
    maximum = np.finfo(dtype).max
    small = typ(1e-200 if dtype == "float64" else 1e-25)
    cases = [
        ("log", None, tiny, tiny),
        ("log", None, maximum, maximum),
        ("sqrt", None, tiny, typ(1)),
        ("sqrt", None, maximum, maximum),
        ("power", Fraction(2), small, typ(1)),
        ("power", Fraction(0), tiny, typ(1)),
        ("power", Fraction(1), tiny, typ(1)),
    ]
    for op, exponent, value, seed in cases:
        program = Program({"out": _operation(op, _input(dtype=dtype), exponent)})
        x, w = np.asarray(value), np.asarray(seed)
        expected = typ(_oracle(op, value, exponent, 1, float(seed)))
        feeds = {"x": x}
        for actual in (
            jvp(program, feeds, {"x": w}).output_tangents["out"],
            vjp(program, feeds, {"out": w}).input_cotangents["x"],
            execute(linearize(program, ["x"]).program, {**feeds, "d_x": w}).outputs[
                "d_out"
            ],
            execute(
                transpose_program(program, ["out"]).program, {**feeds, "bar_out": w}
            ).outputs["bar_x"],
        ):
            np.testing.assert_allclose(
                actual, expected, rtol=32 * np.finfo(dtype).eps, atol=tiny
            )
            if expected != 0:
                assert actual != 0, (
                    "lost a representable derivative after primal underflow"
                )


def test_composed_reused_inputs_and_second_derivative():
    x = _input((3,))
    terms = add(exp(x), log(x), sqrt(x), power(x, "3/2"))
    program = Program({"energy": reduce_sum(add(terms, terms), (0,))})
    values = np.array([0.25, 1.0, 2.0])
    grad = transpose_program(program, ["energy"]).program
    hessian_vector = linearize(grad, ["x"], outputs=["bar_x"]).program
    result = execute(
        hessian_vector,
        {"x": values, "bar_energy": np.array(1.0), "d_x": np.ones_like(values)},
    ).outputs["d_bar_x"]
    expected = 2 * (
        np.exp(values) - 1 / values**2 - 0.25 / values**1.5 + 0.75 / np.sqrt(values)
    )
    np.testing.assert_allclose(result, expected, rtol=5e-13, atol=1e-13)


def test_power_exact_exponent_identity_and_rejection():
    x = _input()
    graphs = [Program({"out": power(x, p)}) for p in ("1.5", "6/4", Fraction(3, 2))]
    assert len({p.logical_hash for p in graphs}) == 1
    assert Program({"out": power(x, 2)}).logical_hash != graphs[0].logical_hash
    for bad in (1.5, True, None, float("nan"), x, 10**500, Fraction(1, 10**500)):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            power(x, bad)
    with pytest.raises(ValueError, match="representable"):
        power(_input(dtype="float32"), "1e100")
    node = graphs[0].outputs["out"]
    for attrs in ((), (("exponent", (2, 4)),), (("exponent", (1, 0)),)):
        with pytest.raises(ValueError):
            Node("power", (x,), node.spec, attrs)
    with pytest.raises(ValueError):
        Node("exp", (x, x), exp(x).spec)


@pytest.mark.parametrize("op,exponent", CASES)
def test_nonlinear_symmetry_metadata_and_packed_ad(op, exponent):
    from vibeqc_compiler.tensor import PackedLayout

    axis = IndexSpace("a", "batch", 2)
    indices = (Index("i", axis), Index("j", axis))
    positive = TensorSpec(
        indices, symmetries=(Symmetry((1, 0)),), role="parameter", differentiable=True
    )
    negative = TensorSpec(
        indices,
        symmetries=(Symmetry((1, 0), -1),),
        role="parameter",
        differentiable=True,
    )
    x = input_tensor("x", positive)
    out = _operation(op, x, exponent)
    assert out.spec.symmetries == positive.symmetries
    assert _operation(op, input_tensor("x", negative), exponent).spec.symmetries == ()
    program = Program({"out": out})
    # Packed expansion exercises _rebuild_node for every new primitive.
    layout = PackedLayout(positive)
    derivative = linearize(program, ["x"], packed={"x": layout})
    assert derivative.program.outputs


@pytest.mark.parametrize("op,exponent", CASES)
def test_cuda_emission_and_dtype_gate_without_a_device(op, exponent):
    from vibeqc_compiler.common.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor.cuda_emit import emit_cuda
    from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
    from vibeqc_compiler.tensor.cuda_resident_emit import resident_source

    x = _input((3,))
    graph = Program({"out": add(_operation(op, x, exponent), x)})
    target = cuda_target_info("sm_80")
    for schedule in (
        TensorSchedule(),
        TensorSchedule(views=True, fuse=True, recompute=True),
    ):
        for program in (
            graph,
            linearize(graph, ["x"]).program,
            transpose_program(graph, ["out"]).program,
        ):
            plan = plan_cuda(program, target, schedule=schedule)
            source = emit_cuda(plan)
            assert ("::pow(" if op == "power" else f"::{op}(") in source
            assert "transcendental domain error" in resident_source(plan)
    with pytest.raises(ValueError, match="float64"):
        plan_cuda(
            Program({"out": _operation(op, _input(dtype="float32"), exponent)}), target
        )


@pytest.mark.parametrize(
    "exponent,value",
    [
        (Fraction(2**24 + 1), np.float32(1 + 2**-20)),
        (Fraction(1) + Fraction(1, 2**25), np.float32(1e-35)),
        (Fraction(1) - Fraction(1, 2**26), np.float32(1e-35)),
    ],
)
def test_power_ad_uses_dtype_rounded_execution_exponent(exponent, value):
    x = _input(dtype="float32")
    program = Program({"out": power(x, exponent)})
    feeds = {"x": np.asarray(value)}
    seed = np.asarray(0.75, dtype=np.float32)
    p_exec = Fraction(float(np.float32(float(exponent))))
    expected = np.float32(_oracle("power", value, p_exec, order=1, weight=0.75))
    candidates = [
        jvp(program, feeds, {"x": seed}).output_tangents["out"],
        vjp(program, feeds, {"out": seed}).input_cotangents["x"],
    ]
    for derivative, name, output in (
        (linearize(program, ["x"]).program, "d_x", "d_out"),
        (transpose_program(program, ["out"]).program, "bar_out", "bar_x"),
    ):
        for replay in (derivative, optimize(Program.loads(derivative.dumps()))):
            candidates.append(execute(replay, {**feeds, name: seed}).outputs[output])
    for actual in candidates:
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0)

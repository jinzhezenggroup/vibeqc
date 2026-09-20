"""Independent exact-binary-rational regressions for division AD (#477)."""

import typing
from fractions import Fraction

import numpy as np
import pytest
from vibeqc_compiler.tensor import (
    AD_RULE_VERSION,
    GENERATION_VERSION,
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    divide,
    dot_test,
    execute,
    input_tensor,
    jvp,
    linearize,
    optimize,
    scaled_bilinear,
    transpose_program,
    vjp,
)


def _program(dtype: typing.Any, shape: typing.Any = ()) -> typing.Any:
    indices = tuple(
        Index(f"i{i}", IndexSpace(f"s{i}", "batch", n)) for i, n in enumerate(shape)
    )
    spec = TensorSpec(
        indices, dtype=np.dtype(dtype).name, role="parameter", differentiable=True
    )
    x, y = (input_tensor(name, spec) for name in ("x", "y"))
    return Program({"out": divide(x, y)})


def _rational(value: typing.Any) -> typing.Any:
    return Fraction.from_float(float(value))


def _rounded(value: typing.Any, dtype: typing.Any) -> typing.Any:
    try:
        with np.errstate(over="ignore", under="ignore"):
            return dtype(float(value))
    except OverflowError:
        return dtype(np.inf if value > 0 else -np.inf)


def _close(actual: typing.Any, expected: typing.Any, dtype: typing.Any) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected, dtype=dtype)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(actual[expected == 0], expected[expected == 0])
    assert np.all(actual[expected != 0] != 0), "lost a representable nonzero result"
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=8 * np.finfo(dtype).eps,
        atol=np.finfo(dtype).smallest_subnormal,
    )


def _cases(dtype: typing.Any) -> typing.Any:
    power = 600 if dtype == np.float64 else 80
    large = 1e200 if dtype == np.float64 else 1e25
    small = 1 / large
    cases = []
    for scale in (
        large,
        small,
        np.finfo(dtype).max,
        np.finfo(dtype).smallest_subnormal,
    ):
        for sign in (1, -1):
            cases.append(
                (
                    scale,
                    sign * scale,
                    scale,
                    scale,
                    scale if scale == np.finfo(dtype).smallest_subnormal else 1,
                )
            )
    # A reassociation that divides the seed first silently loses bar_y.
    cases.append(
        (1e300, 1e100, 1.0, 1.0, 1e-300)
        if dtype == np.float64
        else (1e35, 1e15, 1.0, 1.0, 1e-35)
    )
    cases.extend(
        [
            (0, large, 1, 1, 2.5),
            (0, small, small, small, -2.5),
            (np.finfo(dtype).smallest_subnormal, 1, 0, 1, 1),
        ]
    )
    # Rounded products are equal, but the exact difference is -eps**2.
    eps = np.finfo(dtype).eps
    for exponent in (power, -power, 0):
        s = np.ldexp(dtype(1), exponent)
        cases.append((s, s * dtype(1 - eps), s * dtype(1 + eps), s, 1))
    # Both quotient-rule summands overflow; their difference does not.
    s, t = np.ldexp(dtype(1), -power), np.ldexp(dtype(1), power)
    cases.append((s, s, t, t, 1))
    return [tuple(np.asarray(value, dtype=dtype) for value in row) for row in cases]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_division_ad_extreme_scales_and_compensated_cancellation(
    dtype: typing.Any,
) -> None:
    program = _program(dtype)
    for x, y, dx, dy, w in _cases(dtype):
        feeds = {"x": x, "y": y}
        rx, ry, rdx, rdy, rw = map(_rational, (x, y, dx, dy, w))
        for names in (("x",), ("y",), ("x", "y")):
            tangents = {name: {"x": dx, "y": dy}[name] for name in names}
            expected = _rounded(
                ((rdx if "x" in names else 0) * ry - rx * (rdy if "y" in names else 0))
                / (ry * ry),
                dtype,
            )
            if np.isfinite(expected):
                _close(
                    jvp(program, feeds, tangents).output_tangents["out"],
                    expected,
                    dtype,
                )
                generated = linearize(program, names)
                args = {
                    **feeds,
                    **{f"d_{name}": value for name, value in tangents.items()},
                }
                for replay in (
                    generated.program,
                    optimize(Program.loads(generated.program.dumps())),
                ):
                    _close(execute(replay, args).outputs["d_out"], expected, dtype)
            expected_bars = {
                "x": _rounded(rw / ry, dtype),
                "y": _rounded(-rw * rx / (ry * ry), dtype),
            }
            if all(np.isfinite(expected_bars[name]) for name in names):
                reference = vjp(program, feeds, {"out": w}, inputs=names)
                generated = transpose_program(program, ["out"], inputs=names)
                actual = execute(generated.program, {**feeds, "bar_out": w}).outputs
                for name in names:
                    _close(reference.input_cotangents[name], expected_bars[name], dtype)
                    _close(actual[f"bar_{name}"], expected_bars[name], dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unrequested_overflowing_adjoint_does_not_poison_requested_one(
    dtype: typing.Any,
) -> None:
    x, y, w = (1e-300, 1e-100, 1e300) if dtype == np.float64 else (1e-35, 1e-10, 1e30)
    x, y, w = (np.asarray(a, dtype=dtype) for a in (x, y, w))
    program, feeds = _program(dtype), {"x": x, "y": y}
    expected = _rounded(-_rational(w) * _rational(x) / _rational(y) ** 2, dtype)
    _close(
        vjp(program, feeds, {"out": w}, inputs=["y"]).input_cotangents["y"],
        expected,
        dtype,
    )
    generated = transpose_program(program, ["out"], inputs=["y"])
    _close(
        execute(generated.program, {**feeds, "bar_out": w}).outputs["bar_y"],
        expected,
        dtype,
    )
    with pytest.raises(ValueError, match="non-finite"):
        vjp(program, feeds, {"out": w})


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_seeded_random_exponent_sweep_against_exact_rationals(
    dtype: typing.Any,
) -> None:
    rng = np.random.default_rng(477)
    limit = 1000 if dtype == np.float64 else 120
    program = _program(dtype)
    forward, reverse = (
        linearize(program, ["x", "y"]),
        transpose_program(program, ["out"]),
    )
    checked = 0
    for _ in range(300):
        values = np.ldexp(
            rng.uniform(0.5, 1, 5).astype(dtype) * rng.choice([-1, 1], 5).astype(dtype),
            rng.integers(-limit, limit, 5),
        )
        x, y, dx, dy, w = (np.asarray(a) for a in values)
        rx, ry, rdx, rdy, rw = map(_rational, values)
        expected = [
            _rounded(q, dtype)
            for q in (
                rx / ry,
                (rdx * ry - rx * rdy) / (ry * ry),
                rw / ry,
                -rw * rx / (ry * ry),
            )
        ]
        if not all(np.isfinite(expected)):
            continue
        checked += 1
        feeds, tangents, cotangents = {"x": x, "y": y}, {"x": dx, "y": dy}, {"out": w}
        _close(jvp(program, feeds, tangents).output_tangents["out"], expected[1], dtype)
        _close(
            execute(forward.program, {**feeds, "d_x": dx, "d_y": dy}).outputs["d_out"],
            expected[1],
            dtype,
        )
        bars = vjp(program, feeds, cotangents).input_cotangents
        generated = execute(reverse.program, {**feeds, "bar_out": w}).outputs
        for name, value in zip(("x", "y"), expected[2:]):
            _close(bars[name], value, dtype)
            _close(generated[f"bar_{name}"], value, dtype)
    assert checked >= 50


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_primitive_replay_adjoint_and_higher_derivatives(
    dtype: typing.Any,
) -> None:
    spec = TensorSpec(dtype=np.dtype(dtype).name, role="parameter", differentiable=True)
    nodes = [input_tensor(name, spec) for name in "abcdef"]
    program = Program({"out": scaled_bilinear(*nodes)})
    feeds = {
        name: np.asarray(value, dtype=dtype)
        for name, value in zip("abcdef", (2, 3, 4, 5, 2, -3))
    }
    tangents = {name: np.asarray(0.125, dtype=dtype) for name in "abcdef"}
    cotangents = {"out": np.asarray(0.75, dtype=dtype)}
    assert dot_test(
        program, feeds, tangents, cotangents, rtol=32 * np.finfo(dtype).eps
    ).passed
    forward = linearize(program, list(tangents))
    _close(
        execute(
            forward.program, {**feeds, **{f"d_{k}": v for k, v in tangents.items()}}
        ).outputs["d_out"],
        jvp(program, feeds, tangents).output_tangents["out"],
        dtype,
    )
    reverse = transpose_program(program, ["out"])
    bars = vjp(program, feeds, cotangents).input_cotangents
    actual = execute(reverse.program, {**feeds, "bar_out": cotangents["out"]}).outputs
    for name in feeds:
        _close(actual[f"bar_{name}"], bars[name], dtype)
    # Differentiating the generated quotient adjoint is still legal.
    quotient = _program(dtype)
    first = transpose_program(quotient, ["out"], inputs=["y"])
    second = linearize(first.program, ["y"])
    args = {
        "x": np.asarray(3, dtype=dtype),
        "y": np.asarray(2, dtype=dtype),
        "bar_out": np.asarray(1, dtype=dtype),
        "d_y": np.asarray(1, dtype=dtype),
    }
    _close(execute(second.program, args).outputs["d_bar_y"], dtype(0.75), dtype)
    assert AD_RULE_VERSION == GENERATION_VERSION == 2
    with pytest.raises(ValueError, match="six operands"):
        Node("scaled_bilinear", tuple(nodes[:5]), nodes[0].spec.result())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vector_empty_shape_and_explicit_errors(dtype: typing.Any) -> None:
    for shape in ((2, 3), (0,)):
        program = _program(dtype, shape)
        x = np.full(shape, 1e20, dtype=dtype)
        y = np.full(shape, 1e20, dtype=dtype)
        generated = linearize(program, ["x", "y"])
        result = execute(generated.program, {"x": x, "y": y, "d_x": x, "d_y": y})
        _close(result.outputs["d_out"], np.zeros(shape, dtype=dtype), dtype)
    program = _program(dtype)
    generated = linearize(program, ["x", "y"])
    args = {
        "x": np.asarray(0, dtype=dtype),
        "y": np.asarray(0, dtype=dtype),
        "d_x": np.asarray(0, dtype=dtype),
        "d_y": np.asarray(0, dtype=dtype),
    }
    with pytest.raises(ValueError, match="division by zero"):
        execute(generated.program, args)
    small = np.finfo(dtype).tiny
    args.update(
        x=np.asarray(1, dtype=dtype),
        y=np.asarray(small, dtype=dtype),
        d_x=np.asarray(0, dtype=dtype),
        d_y=np.asarray(1, dtype=dtype),
    )
    with pytest.raises(ValueError, match="non-finite"):
        execute(generated.program, args)
    with pytest.raises(ValueError, match="non-finite"):
        jvp(program, {k: args[k] for k in ("x", "y")}, {"y": args["d_y"]})


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("mode", ["jvp", "vjp", "vjp_y"])
def test_scaled_division_on_allocated_cuda(
    mode: typing.Any, fuse: typing.Any, dtype: typing.Any, tmp_path: typing.Any
) -> None:
    """No implicit GPU probing; execute only in an explicitly allocated job."""
    import os
    from pathlib import Path

    if os.environ.get("VIBEQC_TENSOR_CUDA_TEST") != "1":
        pytest.skip("requires explicit allocated-GPU opt-in")
    from vibeqc.profiles import find_nvcc
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

    nvcc = find_nvcc()
    assert nvcc is not None
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )
    rows, expected = [], []
    candidates = _cases(dtype)
    if mode == "vjp_y":
        values = (
            (1e-300, 1e-100, 1, 1, 1e300)
            if dtype == np.float64
            else (1e-35, 1e-15, 1, 1, 1e35)
        )
        candidates.append(tuple(np.asarray(v, dtype=dtype) for v in values))
    for row in candidates:
        x, y, dx, dy, w = map(_rational, row)
        if mode == "jvp":
            values = [(dx * y - x * dy) / (y * y)]
        else:
            values = [w / y, -w * x / (y * y)] if mode == "vjp" else [-w * x / (y * y)]
        values = [_rounded(v, dtype) for v in values]
        if all(np.isfinite(values)):
            rows.append(row)
            expected.append(values)
    inputs = {
        name: np.array([row[i] for row in rows], dtype=dtype)
        for i, name in enumerate(("x", "y", "d_x", "d_y", "bar_out"))
    }
    primal = _program(dtype, (len(rows),))
    if mode == "jvp":
        program = linearize(primal, ["x", "y"]).program
        names = ["d_out"]
    else:
        selected = ["x", "y"] if mode == "vjp" else ["y"]
        program = transpose_program(primal, ["out"], inputs=selected).program
        names = [f"bar_{name}" for name in selected]
    plan = plan_cuda(program, compiler.target, schedule=TensorSchedule(fuse=fuse))
    cache = Path(os.environ.get("VIBEQC_TENSOR_CACHE", str(tmp_path)))
    with PreparedCuda(plan, compile_cuda(plan, compiler, cache)) as prepared:
        feed_names = {
            step.node.attrs["name"] for step in plan.steps if step.node.op == "input"
        }
        feeds = {name: inputs[name] for name in feed_names}
        for _ in range(2):
            actual = prepared.execute(feeds).outputs
            for i, name in enumerate(names):
                _close(actual[name], np.asarray(expected, dtype=dtype)[:, i], dtype)
        # The fused primitive must keep the native zero-denominator diagnostic
        # and allow an independent subsequent execution on the same handle.
        with pytest.raises(RuntimeError, match="division by zero"):
            prepared.execute({**feeds, "y": np.zeros_like(feeds["y"])})
        overflow = {name: np.ones_like(value) for name, value in feeds.items()}
        overflow["y"].fill(np.finfo(dtype).tiny)
        if mode == "jvp":
            overflow["d_x"].fill(0)
        with pytest.raises(RuntimeError, match="non-finite"):
            prepared.execute(overflow)
        actual = prepared.execute(feeds).outputs
        for i, name in enumerate(names):
            _close(actual[name], np.asarray(expected, dtype=dtype)[:, i], dtype)


def test_scaled_cuda_source_contract_without_runtime_or_device() -> None:
    from vibeqc_compiler.common.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor.cuda_emit import emit_cuda
    from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

    program = linearize(_program(np.float64), ["x", "y"]).program
    for fuse in (False, True):
        plan = plan_cuda(
            program, cuda_target_info("sm_80"), schedule=TensorSchedule(fuse=fuse)
        )
        source = emit_cuda(plan, symbol_prefix="quotient_")
        assert source.count("__device__ inline double quotient_scaled_bilinear(") == 1
        assert "return quotient_scaled_bilinear(" in source
        assert "__fma_rn(ma, mb, -p)" in source
        assert "atomicCAS(error, 0, -(node + 1))" in source
        assert "finite(scalbn(ratio, exponent - ee - ef), error, node)" in source
    fp32 = plan_cuda(
        linearize(_program(np.float32), ["x", "y"]).program,
        cuda_target_info("sm_80"),
    )
    source = emit_cuda(fp32)
    assert "__device__ inline float scaled_bilinear(" in source
    assert "__fmaf_rn(ma, mb, -p)" in source
    assert "scalbnf(ratio, exponent - ee - ef)" in source
    assert "dp < -52" in source

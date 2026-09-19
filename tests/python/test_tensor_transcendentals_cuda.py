"""Real-device ordinary/resident TensorIR transcendental qualification (#500)."""

import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    execute,
    exp,
    input_tensor,
    linearize,
    log,
    multiply,
    power,
    sqrt,
    transpose_program,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)


@pytest.fixture(scope="module")
def compiler():
    from vibeqc.profiles import find_nvcc
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info

    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_TENSOR_CUDA_TEST requires a CUDA compiler")
    return CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return (
        Path(os.environ["VIBEQC_TENSOR_CACHE"])
        if "VIBEQC_TENSOR_CACHE" in os.environ
        else tmp_path_factory.mktemp("transcendentals-cuda")
    )


def _input():
    i = Index("i", IndexSpace("axis", "batch", 6))
    return input_tensor("x", TensorSpec((i,), role="parameter", differentiable=True))


@pytest.mark.parametrize("schedule_id", [0, 1, 2])
def test_cuda_primal_jvp_vjp_ordinary_and_resident(
    compiler, cache, schedule_id, monkeypatch
):
    from vibeqc_compiler.tensor import interpreter
    from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
    from vibeqc_compiler.tensor.cuda_resident import PreparedResident, compile_resident

    schedules = [
        TensorSchedule(),
        TensorSchedule(views=True, fuse=True),
        TensorSchedule(views=True, fuse=True, recompute=True),
    ]
    x = _input()
    y = add(exp(x), log(x), sqrt(x), power(x, "3/2"))
    primal = Program({"out": y, "square": multiply(y, y)})
    values = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    feeds = {"x": values}
    jobs = [
        (primal, feeds),
        (linearize(primal, ["x"]).program, {**feeds, "d_x": np.full(6, 0.75)}),
        (
            transpose_program(primal, ["out", "square"]).program,
            {**feeds, "bar_out": np.ones(6), "bar_square": np.full(6, 0.25)},
        ),
    ]
    expected = [execute(program, args).outputs for program, args in jobs]

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA delegated arithmetic to the CPU interpreter")

    monkeypatch.setattr(interpreter, "_evaluate", forbidden)
    for (program, args), reference in zip(jobs, expected):
        plan = plan_cuda(program, compiler.target, schedule=schedules[schedule_id])
        with PreparedCuda(plan, compile_cuda(plan, compiler, cache)) as prepared:
            for _ in range(2):
                result = prepared.execute(args).outputs
                for name, expected_value in reference.items():
                    np.testing.assert_allclose(
                        result[name], expected_value, rtol=5e-13, atol=1e-13
                    )
        with PreparedResident(
            plan, compile_resident(plan, compiler, cache)
        ) as resident:
            resident.upload(args)
            for _ in range(2):
                leases, metrics = resident.run()
                assert metrics["h2d_bytes"] == 0 and metrics["d2h_bytes"] == 4
                for name, expected_value in reference.items():
                    np.testing.assert_allclose(
                        resident.download(leases[name]),
                        expected_value,
                        rtol=5e-13,
                        atol=1e-13,
                    )


@pytest.mark.parametrize(
    "kind",
    [
        "log",
        "sqrt",
        "sqrt-zero-ad",
        "exp-overflow",
        "power-zero",
        "power-one",
        "power-overflow",
    ],
)
def test_cuda_error_propagation_recovery_and_no_stale_resident_outputs(
    compiler, cache, kind
):
    from vibeqc_compiler.tensor.cuda_execute import PreparedCuda
    from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
    from vibeqc_compiler.tensor.cuda_resident import PreparedResident, compile_resident

    x = _input()
    if kind.startswith("power"):
        exponent = {"power-zero": 0, "power-one": 1, "power-overflow": 2}[kind]
        node = power(x, Fraction(exponent))
    elif kind.startswith("sqrt"):
        node = sqrt(x)
    elif kind.startswith("exp"):
        node = exp(x)
    else:
        node = log(x)
    primal = Program({"out": node})
    # Check retained primal-domain/overflow guards even with an exactly zero seed.
    program = linearize(primal, ["x"]).program
    plan = plan_cuda(
        program, compiler.target, schedule=TensorSchedule(views=True, fuse=True)
    )
    artifact = compile_resident(plan, compiler, cache)
    good = {"x": np.ones(6), "d_x": np.zeros(6)}
    bad = {**good, "x": np.ones(6)}
    bad["x"][3] = {
        "sqrt-zero-ad": 0.0,
        "exp-overflow": 1000.0,
        "power-overflow": 1e200,
        "power-zero": 0.0,
    }.get(kind, -1.0)
    message = (
        "division"
        if kind == "sqrt-zero-ad"
        else "non-finite"
        if "overflow" in kind
        else "domain"
    )
    with PreparedCuda(plan, artifact) as ordinary:
        ordinary.execute(good)
        with pytest.raises(RuntimeError, match=message):
            ordinary.execute(bad)
        np.testing.assert_array_equal(ordinary.execute(good).outputs["d_out"], 0.0)
    with PreparedResident(plan, artifact) as resident:
        resident.upload(good)
        old, _ = resident.run()
        resident.upload(bad)
        with pytest.raises(RuntimeError, match=message):
            resident.run()
        with pytest.raises(RuntimeError):
            resident.download(old["d_out"])
        resident.upload(good)
        new, _ = resident.run()
        np.testing.assert_array_equal(resident.download(new["d_out"]), 0.0)

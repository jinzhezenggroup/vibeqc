"""Real-device GeometryIR/PairIR energy and generated-force qualification (#502)."""

import os
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.geometry import GeometryIR, PairTopology, inverse_power_program
from vibeqc_compiler.tensor import execute

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)


@pytest.fixture(scope="module")
def compiler() -> typing.Any:
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
def cache(tmp_path_factory: typing.Any) -> typing.Any:
    return (
        Path(os.environ["VIBEQC_TENSOR_CACHE"])
        if "VIBEQC_TENSOR_CACHE" in os.environ
        else tmp_path_factory.mktemp("geometry-pair-cuda")
    )


def test_cuda_energy_and_generated_coordinate_vjp_match_reference_and_fd(
    compiler: typing.Any, cache: typing.Any
) -> None:
    from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

    pair_program = inverse_power_program(
        GeometryIR((1, 6, 8), parameter_identity="elements-v1"),
        PairTopology.complete(3),
        ("3/2", "-2/3", "5/4"),
        exponent=-2,
        parameter_identity="qualification-v1",
    )
    coordinates = np.array(
        [
            [0.1, -0.2, 0.3],
            [1.4, 0.5, -0.1],
            [-0.7, 1.2, 0.9],
        ],
        dtype=np.float64,
    )
    reverse = pair_program.coordinate_vjp().program
    jobs = [
        (pair_program.program, {"coordinates": coordinates}),
        (
            reverse,
            {"coordinates": coordinates, "bar_energy": np.array(1.0)},
        ),
    ]
    references = [execute(program, feeds).outputs for program, feeds in jobs]
    schedule = TensorSchedule(views=True, fuse=True)
    plans = [
        plan_cuda(program, compiler.target, schedule=schedule) for program, _ in jobs
    ]
    artifacts = [compile_cuda(plan, compiler, cache) for plan in plans]
    with (
        PreparedCuda(plans[0], artifacts[0]) as energy_cuda,
        PreparedCuda(plans[1], artifacts[1]) as gradient_cuda,
    ):
        cuda_energy = energy_cuda.execute(jobs[0][1]).outputs["energy"]
        cuda_gradient = gradient_cuda.execute(jobs[1][1]).outputs["bar_coordinates"]
        np.testing.assert_allclose(
            cuda_energy, references[0]["energy"], rtol=5e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            cuda_gradient, references[1]["bar_coordinates"], rtol=8e-12, atol=2e-12
        )
        step = 2e-6
        finite_difference = np.empty_like(coordinates)
        for atom in range(3):
            for axis in range(3):
                plus, minus = coordinates.copy(), coordinates.copy()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                e_plus = (
                    energy_cuda.execute({"coordinates": plus}).outputs["energy"].item()
                )
                e_minus = (
                    energy_cuda.execute({"coordinates": minus}).outputs["energy"].item()
                )
                finite_difference[atom, axis] = (e_plus - e_minus) / (2 * step)
        np.testing.assert_allclose(
            cuda_gradient, finite_difference, rtol=3e-8, atol=3e-9
        )

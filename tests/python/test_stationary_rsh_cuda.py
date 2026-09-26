"""CUDA RSH stationary exchange binding over the generated weighted-ERI provider."""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc._stationary_rsh_cpu import RangeExchangeExecutor
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.spec import RangeSeparatedExchangePrimitive


def _fake_s_basis() -> SimpleNamespace:
    centers = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )
    primitives = np.array(
        [[0.57, 0.83], [0.71, -0.19], [0.89, 0.67], [1.13, 0.42]]
    )
    aos = np.zeros((4, 16))
    for index in range(4):
        aos[index, :8] = (index, index, 1, 1, 0, 0, 0, 1)
    return SimpleNamespace(
        natom=4,
        nao=4,
        nprimitive=4,
        shells=tuple(SimpleNamespace(angular_momentum=0) for _ in range(4)),
        packed=np.concatenate((centers.ravel(), primitives.ravel(), aos.ravel())),
    )


def test_range_exchange_executor_rejects_implicit_backend_and_cpu_device(
    tmp_path: Path,
) -> None:
    basis = _fake_s_basis()
    with pytest.raises(TypeError, match="explicit CPU C\\+\\+ or CUDA"):
        RangeExchangeExecutor(basis, tmp_path, 8, object())
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native C++ compiler unavailable")
    with pytest.raises(ValueError, match="device_id=0"):
        RangeExchangeExecutor(
            basis,
            tmp_path,
            8,
            CppCompilerAdapter(Path(compiler)),
            device_id=1,
        )


@pytest.mark.parametrize("operator", ("short-range", "long-range"))
def test_cuda_range_exchange_derivative_matches_cpu(
    tmp_path: Path, operator: str
) -> None:
    if os.environ.get("VIBEQC_TEST_RANGE_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_RANGE_CUDA=1 inside a Slurm GPU job")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("native CUDA validation requires a Slurm allocation")
    nvcc = shutil.which("nvcc")
    cxx = shutil.which("c++")
    if nvcc is None or cxx is None:
        pytest.skip("native CPU/CUDA compilers unavailable")

    basis = _fake_s_basis()
    method = resolve_method("CAM-B3LYP", spin="unpolarized")
    primitive = next(
        item
        for item in method.primitives
        if type(item) is RangeSeparatedExchangePrimitive and item.operator == operator
    )
    cpu = RangeExchangeExecutor(
        basis, tmp_path / "cpu", 8, CppCompilerAdapter(Path(cxx))
    )
    cuda = RangeExchangeExecutor(
        basis,
        tmp_path / "cuda",
        8,
        CudaCompilerAdapter(Path(nvcc), cuda_target_info("sm_120")),
        device_id=0,
    )
    try:
        expected_owners, expected = cpu.integral(
            primitive, (0, 1, 2, 3), 0.731
        )
        actual_owners, actual = cuda.integral(
            primitive, (0, 1, 2, 3), 0.731
        )
        assert cuda.backend == "cuda"
        assert cpu.backend == "cpu"
        assert actual_owners == expected_owners == [0, 1, 2, 3]
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-11)
        np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-12, rtol=0)
        assert cuda.records == cpu.records > 0
    finally:
        cuda.close()
        cpu.close()

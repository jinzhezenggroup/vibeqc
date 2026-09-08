"""Real-device CC DIIS and physical max-norm primitives, separate from equations."""

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_CC_CUDA_TEST") != "1",
    reason="requires explicitly allocated GPU validation window",
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def state_library(tmp_path_factory):
    from vibeqc.profiles import find_nvcc

    compiler = CudaCompilerAdapter(
        find_nvcc(), cuda_target_info(os.environ["VIBEQC_TENSOR_ARCH"])
    )
    directory = tmp_path_factory.mktemp("cc-state")
    library = directory / "state.so"
    result = compiler.compile_shared(
        ROOT / "tests/cc_cuda_state.cu",
        library,
        includes=(ROOT / "src", ROOT / "src/tensor"),
        libraries=("cublas",),
        options=("--fmad=false",),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lib = ctypes.CDLL(str(library))
    lib.test_cc_state.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        *([ctypes.c_void_p] * 6),
        ctypes.c_size_t,
    ]
    lib.test_cc_state.restype = ctypes.c_int
    return lib


@pytest.mark.parametrize("elements,history", [(7, 2), (257, 6), (513, 20)])
def test_gpu_diis_and_norm_against_dense_reference(state_library, elements, history):
    rng = np.random.default_rng(149)
    errors = rng.normal(size=(history, elements))
    vectors = rng.normal(size=errors.shape)
    output = np.empty(elements)
    norm, status, error = (
        ctypes.c_double(),
        ctypes.c_int(),
        ctypes.create_string_buffer(2048),
    )
    assert (
        state_library.test_cc_state(
            elements,
            history,
            errors.ctypes.data,
            vectors.ctypes.data,
            output.ctypes.data,
            ctypes.byref(norm),
            ctypes.byref(status),
            error,
            len(error),
        )
        == 0
    ), error.value
    gram = errors @ errors.T
    system = np.full((history + 1, history + 1), -1.0)
    system[:history, :history] = gram / np.max(np.abs(gram))
    system[-1, -1] = 0
    rhs = np.zeros(history + 1)
    rhs[-1] = -1
    coefficients = np.linalg.solve(system, rhs)[:history]
    assert status.value == 0
    assert norm.value == np.max(np.abs(errors[0]))
    np.testing.assert_allclose(output, coefficients @ vectors, atol=1e-11, rtol=1e-10)


def test_gpu_diis_singular_history_is_explicit_rejection(state_library):
    errors = np.ones((2, 7))
    vectors = errors.copy()
    output = np.empty(7)
    norm, status, error = (
        ctypes.c_double(),
        ctypes.c_int(),
        ctypes.create_string_buffer(2048),
    )
    assert (
        state_library.test_cc_state(
            7,
            2,
            errors.ctypes.data,
            vectors.ctypes.data,
            output.ctypes.data,
            ctypes.byref(norm),
            ctypes.byref(status),
            error,
            len(error),
        )
        == 0
    ), error.value
    assert status.value == 1
    np.testing.assert_array_equal(output, 0)

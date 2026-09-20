"""Executable CPU TensorIR subset: independent algebra and hostile ABI gates."""

import ctypes as ct
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    multiply,
    reduce_sum,
    reshape,
)
from vibeqc_compiler.tensor.cpu import NativeTensorProgram, emit_cpu


def tensor(
    name: typing.Any, shape: typing.Any, dtype: typing.Any = "float64"
) -> typing.Any:
    return input_tensor(
        name,
        TensorSpec(
            tuple(
                Index(chr(105 + k), IndexSpace("space" + str(n), "batch", n))
                for k, n in enumerate(shape)
            ),
            dtype=dtype,
            role="input",
        ),
    )


def native(
    program: typing.Any, tmp_path: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    return NativeTensorProgram(
        program, compiler=CppCompilerAdapter(Path("c++")), cache=tmp_path, **kwargs
    )


@pytest.mark.parametrize("size", [0, 3])
def test_reductions_einsums_and_detached_outputs(
    tmp_path: typing.Any, size: typing.Any
) -> None:
    a, b = tensor("a", (2, size)), tensor("b", (size, 4))
    contracted = einsum("ij,jk->ki", a, b, coefficient="-1/2")
    program = Program({"matrix": contracted, "total": reduce_sum(contracted, (0, 1))})
    executor = native(program, tmp_path)
    av = np.arange(2 * size, dtype=float).reshape(2, size)[:, ::-1]
    bv = np.arange(size * 4, dtype=float).reshape(size, 4)
    result = executor.execute({"a": av, "b": bv})
    np.testing.assert_array_equal(result["matrix"], -0.5 * (av @ bv).T)
    assert result["total"] == result["matrix"].sum()
    with pytest.raises(ValueError):
        result["matrix"].flags.writeable = True
    replay = native(Program.loads(program.dumps()), tmp_path)
    assert executor.identity == replay.identity
    assert executor.artifact.library == replay.artifact.library
    other = native(
        Program(dict(program.outputs), provenance={"different_plan": True}), tmp_path
    )
    assert executor.identity != other.identity


def test_ordinary_pointwise_arithmetic(tmp_path: typing.Any) -> None:
    a, b = tensor("a", (4,)), tensor("b", (4,))
    executor = native(
        Program({"value": add(multiply(a, b), a, coefficients=(2, -3))}), tmp_path
    )
    av, bv = np.arange(4, dtype=float), np.arange(4, dtype=float) + 2
    np.testing.assert_array_equal(
        executor.execute({"a": av, "b": bv})["value"], 2 * av * bv - 3 * av
    )


def test_preallocation_and_semantic_rejection(tmp_path: typing.Any) -> None:
    a = tensor("a", (3,))
    for program, message in (
        (Program({"a": tensor("a", (3,), "float32")}), "float64"),
        (
            Program({"a": reshape(a, a.spec.indices)}),
            "unsupported",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            emit_cpu(program)
    with pytest.raises(ValueError, match="budget"):
        emit_cpu(Program({"a": a}), max_bytes=1)
    with pytest.raises(ValueError, match="budget"):
        emit_cpu(Program({"a": tensor("a", (10**12,))}))
    with pytest.raises(ValueError, match="budget"):
        emit_cpu(Program({"a": reduce_sum(a, (0,))}), max_work=1)
    with pytest.raises(TypeError, match="CPU compiler"):
        NativeTensorProgram(Program({"a": a}), compiler=object(), cache=tmp_path)
    for budget in (-1, True, 2**64):
        with pytest.raises(ValueError):
            emit_cpu(Program({"a": a}), max_bytes=budget)


def test_invalid_feeds_and_late_native_overflow_are_transactional(
    tmp_path: typing.Any,
) -> None:
    a = tensor("a", (3,))
    executor = native(Program({"a": multiply(a, a)}), tmp_path)
    for bad in (
        {},
        {"a": np.ones(2)},
        {"a": np.ones(3, dtype=np.float32)},
        {"a": np.array([1.0, np.nan, 2.0])},
    ):
        with pytest.raises(ValueError):
            executor.execute(bad)
    ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
    values = np.array([1.0, 2.0, 1e200])
    output = np.full(3, 42.0)
    for ni, no, budget in (
        (3, 3, executor.max_bytes),
        (2, 3, executor.max_bytes),
        (2**64 - 1, 3, executor.max_bytes),
        (3, 3, 0),
        (3, 2, executor.max_bytes),
    ):
        assert executor.call(ptr(values), ni, ptr(output), no, budget) != 0
        np.testing.assert_array_equal(output, 42.0)
    values[2] = np.nan
    assert executor.call(ptr(values), 3, ptr(output), 3, executor.max_bytes) != 0
    np.testing.assert_array_equal(output, 42.0)
    np.testing.assert_array_equal(
        executor.execute({"a": np.arange(3, dtype=float)})["a"], [0, 1, 4]
    )


def test_native_source_generation_does_not_probe_runtime_or_compilers() -> None:
    import subprocess
    import sys

    source = r"""
import importlib.abc
import sys
import ctypes
import subprocess
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vibeqc', 'pyscf', 'cupy', 'torch'}:
            raise AssertionError('unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
def blocked(*args, **kwargs):
    raise AssertionError('generation probed a runtime or compiler')
ctypes.CDLL = blocked
subprocess.run = blocked
subprocess.Popen = blocked
subprocess.check_output = blocked
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.stationary_gradient import (
    StationaryGradientPlan, StationaryMeanField, SCF_POINT_MODEL,
)
from vibeqc_compiler.tensor.cpu import emit_cpu
from vibeqc_compiler.xc.grid_native import emit_grid_contraction
plan = StationaryGradientPlan(resolve_method('PBE'), StationaryMeanField(SCF_POINT_MODEL))
program = plan.integral_block('coulomb', terms=3).weights
first = emit_cpu(program)
assert first == emit_cpu(program)
assert 'tensor_cpu' in first[0]
first = emit_grid_contraction(3)
assert first == emit_grid_contraction(3)
assert 'grid_contract' in first
assert 'namespace vibeqc_grid_adjoint {' in first
assert 'grid_response_adjoint.hpp' not in first
"""
    subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

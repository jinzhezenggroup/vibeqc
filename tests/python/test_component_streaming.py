"""Native scheduler transactionality, immutable metadata and artifact identity."""

import ctypes as ct
import os
import shutil
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc._stationary_cpu_components import ComponentPrimitiveExecutor
from vibeqc._stationary_cpu_streaming import (
    CompiledComponentExecutor,
    Dispatch,
    DoublePointer,
    IndexPointer,
)
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter


def test_native_failure_is_transactional_and_header_is_revalidated(
    tmp_path: Path,
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import _stationary_cpu_streaming as module

    executable = shutil.which(os.environ.get("CXX", "c++"))
    if executable is None:
        pytest.skip("C++ compiler required")
    compiler = CppCompilerAdapter(Path(executable))
    centers = np.array([[0.1, 0.2, -0.3], [1.2, -0.4, 0.5]])
    primitives = np.array([[0.7, 0.3], [1.2, -0.2], [0.5, 0.8]])
    aos = np.zeros((2, 16))
    aos[:, :4] = [[0, 0, 2, 1], [1, 2, 1, 1]]
    aos[:, 7] = [0.6, 0.9]
    basis = SimpleNamespace(
        natom=2,
        nprimitive=3,
        shells=(SimpleNamespace(angular_momentum=0),),
        packed=np.concatenate((centers.ravel(), primitives.ravel(), aos.ravel())),
    )
    # Use a copied runtime header to verify its bytes participate in identity.
    header = tmp_path / "src/integrals/first_derivative_component_runtime.hpp"
    header.parent.mkdir(parents=True)
    header.write_bytes(
        module.asset_path(
            "src/integrals/first_derivative_component_runtime.hpp"
        ).read_bytes()
    )
    monkeypatch.setattr(module, "asset_path", lambda name: header)
    executor = CompiledComponentExecutor(basis, tmp_path, 1, compiler)
    baseline = ComponentPrimitiveExecutor(basis, tmp_path, 1, compiler)
    for array in (executor.labels, executor.bindings):
        assert array.dtype == np.int64
        with pytest.raises(ValueError):
            array.flags.writeable = True
    expected = baseline.integral("overlap", (0, 1), -0.4)[1]
    np.testing.assert_array_equal(
        executor.integral("overlap", (0, 1), -0.4)[1], expected
    )
    before = executor.records
    original = executor.dispatch[0]
    # Copy the function address: indexing the ctypes array may alias its slot.
    saved = ct.cast(ct.cast(original, ct.c_void_p).value, Dispatch)
    calls = []

    @Dispatch
    def fail_late(
        kind: int, records: typing.Any, count: int, output: typing.Any
    ) -> int:
        calls.append(count)
        return 1 if len(calls) == 2 else saved(kind, records, count, output)

    executor.dispatch[0] = fail_late
    out, work = np.full((4, 3), 123.0), ct.c_uint64(987)
    ids = np.array([0, 1], dtype=np.int64)
    status = executor.contract(
        *executor.arguments,
        0,
        ids.ctypes.data_as(IndexPointer),
        -0.4,
        -1,
        executor.buffer.ctypes.data_as(DoublePointer),
        1,
        out.ctypes.data_as(DoublePointer),
        ct.byref(work),
    )
    assert status != 0 and calls == [1, 1]
    np.testing.assert_array_equal(out, 123)
    assert work.value == 987 and executor.records == before
    executor.dispatch[0] = saved
    np.testing.assert_array_equal(
        executor.integral("overlap", (0, 1), -0.4)[1], expected
    )
    assert executor.records == before + 2
    for indices in ((-1, 1), (0,), (0, 2), (0.5, 1)):
        with pytest.raises(ValueError, match="indices"):
            executor.integral("overlap", indices, 1)
    with pytest.raises(ValueError, match="nucleus"):
        executor.integral("nuclear_attraction", (0, 1), 1, 2)
    # Corrupt a selected native binding, bypassing Python's input validation.
    bindings = executor.bindings.copy()
    bindings[0, 6:9] = 0
    arguments = list(executor.arguments)
    arguments[7] = bindings.ctypes.data_as(IndexPointer)
    assert (
        executor.contract(
            *arguments,
            0,
            ids.ctypes.data_as(IndexPointer),
            1,
            -1,
            executor.buffer.ctypes.data_as(DoublePointer),
            1,
            out.ctypes.data_as(DoublePointer),
            ct.byref(work),
        )
        != 0
    )
    np.testing.assert_array_equal(out, 123)
    assert work.value == 987
    header.write_bytes(header.read_bytes() + b"\n// identity mutation\n")
    rebuilt = CompiledComponentExecutor(basis, tmp_path, 1, compiler)
    assert rebuilt.runtime._name != executor.runtime._name
    np.testing.assert_array_equal(
        rebuilt.integral("overlap", (0, 1), -0.4)[1], expected
    )

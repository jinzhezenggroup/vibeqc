"""Resident ownership teardown is testable without allocating a CUDA device."""

import ctypes as ct
import typing
from types import SimpleNamespace
from weakref import WeakValueDictionary

import pytest

from tools.vibeqc_response.resident_cuda import CudaResidentRHFResponse


def _owner() -> typing.Any:
    destroyed = []
    owner = CudaResidentRHFResponse.__new__(CudaResidentRHFResponse)
    owner._lib = SimpleNamespace(
        vibeqc_rhf_response_resident_destroy=lambda handle: destroyed.append(
            handle.value
        )
    )
    owner._handle = ct.c_void_p(123)
    owner._closed = False
    owner._free = [0, 1]
    owner._live = set()
    owner._vectors = WeakValueDictionary()
    owner.vector_slots = 2
    return owner, destroyed


@pytest.mark.parametrize("error_type", [MemoryError, ValueError, RuntimeError])
def test_exception_teardown_preserves_error_and_destroys_native_owner(
    error_type: typing.Any,
) -> None:
    owner, destroyed = _owner()
    original = error_type("injected solver failure")
    with pytest.raises(error_type) as captured, owner:
        # A failed solver frame/traceback retains its vector leases while
        # __exit__ runs; garbage collection cannot make them disappear.
        vector = owner._allocate()
        raise original
    assert captured.value is original
    assert owner._closed
    assert not owner._handle
    assert destroyed == [123]
    vector.release()
    assert not owner._live
    owner.close()
    assert destroyed == [123]


def test_normal_close_still_rejects_live_vector_leases() -> None:
    owner, destroyed = _owner()
    vector = owner._allocate()
    with pytest.raises(RuntimeError, match="live vectors"):
        owner.close()
    assert not destroyed
    vector.release()
    owner.close()
    assert destroyed == [123]


@pytest.mark.parametrize("kind", ["foreign", "released", "reused"])
@pytest.mark.parametrize("operation", ["copy", "norm", "to_host"])
def test_native_access_rejects_invalid_vector_leases(
    kind: typing.Any, operation: typing.Any, monkeypatch: typing.Any
) -> None:
    owner, _ = _owner()
    owner.dimension = 1
    origin = _owner()[0] if kind == "foreign" else owner
    value = origin._allocate()
    replacement = None
    if kind != "foreign":
        value.release()
        if kind == "reused":
            replacement = owner._allocate()
            assert replacement.slot == value.slot

    def forbidden(*args: typing.Any) -> None:
        pytest.fail("invalid vector lease reached the native ABI")

    monkeypatch.setattr(owner, "_call", forbidden)
    before = set(owner._live)
    with pytest.raises((ValueError, RuntimeError), match="lease|owner|released"):
        getattr(owner, operation)(value)
    assert owner._live == before
    if replacement is not None:
        replacement.release()
    value.release()


def test_native_access_rejects_closed_borrowed_backend() -> None:
    import threading

    owner, _ = _owner()

    def closed() -> typing.Any:
        raise RuntimeError("CUDA direct response backend is closed")

    def forbidden(*args: typing.Any) -> None:
        pytest.fail("closed borrowed backend reached the native ABI")

    owner._backend = SimpleNamespace(_lock=threading.RLock(), _ensure_open=closed)
    owner._lib.vibeqc_rhf_response_resident_zero = forbidden
    with pytest.raises(RuntimeError, match="backend is closed"):
        owner._call("zero", 0)


@pytest.mark.parametrize("raises", [False, True])
def test_solver_releases_temporaries_even_when_a_profiler_retains_them(
    monkeypatch: typing.Any, raises: typing.Any
) -> None:
    import numpy as np

    from tools.vibeqc_response import krylov

    owner, _ = _owner()
    owner.dimension = 1
    retained = []

    def retaining_solver(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        retained.append(owner._allocate())
        if raises:
            raise RuntimeError("retained solver frame")
        return krylov._workspace_failure(1, 1, 1)[0]

    monkeypatch.setattr(krylov, "_solve_single", retaining_solver)
    operator = SimpleNamespace(dimension=1, _krylov_engine=owner)
    if raises:
        with pytest.raises(RuntimeError, match="retained solver frame"):
            krylov.solve(operator, np.ones(1))
    else:
        krylov.solve(operator, np.ones(1))
    assert retained
    assert not owner._live
    assert retained[0]._released
    owner.close()

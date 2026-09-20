"""Device-resident RHF response vectors over the borrowed direct-CUDA J/K stream.

This is a storage/execution adapter for the existing #179 GMRES controller.
It owns no Krylov algorithm and no alternative response equation. Vector
copies/AXPY/dot/norm and RHF operator actions execute on the device; the host
receives scalar reductions/status during iteration and explicit final results.
"""

from __future__ import annotations

import ctypes as ct
import typing
from contextlib import contextmanager
from weakref import WeakValueDictionary

import numpy as np
from vibeqc import _native
from vibeqc.profiles import canonical_hash

from .direct_cuda import CudaDirectJKBackend
from .problem import ResponseProblem

_DOUBLE = ct.POINTER(ct.c_double)


class _Diagnostic(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("nbf", ct.c_uint64),
        ("nocc", ct.c_uint64),
        ("nvirt", ct.c_uint64),
        ("dimension", ct.c_uint64),
        ("vector_slots", ct.c_uint64),
        ("owned_device_bytes", ct.c_uint64),
        ("h2d_bytes", ct.c_uint64),
        ("d2h_bytes", ct.c_uint64),
        ("synchronizations", ct.c_uint64),
        ("operator_actions", ct.c_uint64),
        ("blas_calls", ct.c_uint64),
        ("device_id", ct.c_int32),
    ]


def _bind(lib: typing.Any) -> None:
    handle = ct.c_void_p
    lib.vibeqc_rhf_response_resident_create.argtypes = [
        ct.c_void_p,
        _DOUBLE,
        ct.c_uint64,
        _DOUBLE,
        ct.c_uint64,
        ct.c_uint32,
        ct.c_uint32,
        ct.c_uint64,
        ct.POINTER(handle),
    ]
    lib.vibeqc_rhf_response_resident_create.restype = ct.c_int32
    lib.vibeqc_rhf_response_resident_destroy.argtypes = [handle]
    lib.vibeqc_rhf_response_resident_destroy.restype = None
    lib.vibeqc_rhf_response_resident_last_error.argtypes = [handle]
    lib.vibeqc_rhf_response_resident_last_error.restype = ct.c_char_p
    lib.vibeqc_rhf_response_resident_get_diagnostic.argtypes = [
        handle,
        ct.POINTER(_Diagnostic),
    ]
    lib.vibeqc_rhf_response_resident_get_diagnostic.restype = ct.c_int32
    signatures = {
        "upload": [handle, ct.c_uint32, _DOUBLE, ct.c_uint64],
        "download": [handle, ct.c_uint32, _DOUBLE, ct.c_uint64],
        "zero": [handle, ct.c_uint32],
        "copy": [handle, ct.c_uint32, ct.c_uint32],
        "scale": [handle, ct.c_uint32, ct.c_double],
        "axpy": [handle, ct.c_uint32, ct.c_double, ct.c_uint32],
        "dot": [handle, ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_double)],
        "norm": [handle, ct.c_uint32, ct.POINTER(ct.c_double)],
        "apply": [handle, ct.c_uint32, ct.c_uint32],
    }
    for name, args in signatures.items():
        function = getattr(lib, f"vibeqc_rhf_response_resident_{name}")
        function.argtypes = args
        function.restype = ct.c_int32


class _ResidentVector:
    __slots__ = ("__weakref__", "_released", "owner", "slot")

    def __init__(self, owner: typing.Any, slot: typing.Any) -> None:
        self.owner, self.slot, self._released = owner, slot, False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self.owner._release_slot(self.slot)

    def __del__(self) -> None:
        if hasattr(self, "_released"):
            self.release()


class CudaResidentRHFResponse:
    """Bound device vector/operator owner consumed by the shared GMRES loop."""

    resident = True

    def __init__(
        self,
        backend: typing.Any,
        problem: typing.Any,
        *,
        vector_slots: typing.Any,
        device_budget_bytes: typing.Any,
    ) -> None:
        if not isinstance(backend, CudaDirectJKBackend):
            raise TypeError("resident RHF response requires CudaDirectJKBackend")
        if not isinstance(problem, ResponseProblem) or problem.method != "rhf":
            raise TypeError("resident RHF response requires an RHF ResponseProblem")
        if type(vector_slots) is not int or not 8 <= vector_slots <= 4096:
            raise ValueError("resident RHF vector_slots must be in [8,4096]")
        if type(device_budget_bytes) is not int or not 0 < device_budget_bytes < 2**63:
            raise ValueError("resident RHF device budget must be a positive int64")
        backend.validate_reference(problem.reference)
        backend._ensure_open()
        plan = backend._plan
        lib = plan._library
        if not hasattr(lib, "vibeqc_rhf_response_resident_create"):
            raise NotImplementedError(
                "native library lacks resident RHF response support"
            )
        _bind(lib)
        self._lib, self._backend, self.problem = lib, backend, problem
        self.dimension = problem.dimension
        self._handle = ct.c_void_p()
        self._closed = False
        coefficients = np.ascontiguousarray(
            problem.reference.coefficients, dtype=np.float64
        )
        energies = np.ascontiguousarray(
            problem.reference.orbital_energies, dtype=np.float64
        )
        status = lib.vibeqc_rhf_response_resident_create(
            plan._handle,
            coefficients.ctypes.data_as(_DOUBLE),
            coefficients.size,
            energies.ctypes.data_as(_DOUBLE),
            energies.size,
            problem.layout.nocc,
            vector_slots,
            device_budget_bytes,
            ct.byref(self._handle),
        )
        if status:
            message = lib.vibeqc_fock_plan_last_error(plan._handle)
            raise RuntimeError(
                (message or b"resident RHF response creation failed").decode()
            )
        self.vector_slots = vector_slots
        self._free = list(reversed(range(vector_slots)))
        self._live = set()
        self._vectors = WeakValueDictionary()
        diagnostic = self.diagnostics
        if (
            diagnostic["nbf"] != problem.reference.nmo
            or diagnostic["nocc"] != problem.layout.nocc
            or diagnostic["nvirt"] != problem.layout.nvirt
            or diagnostic["dimension"] != problem.dimension
            or diagnostic["device_id"] != backend.device_id
        ):
            self.close()
            raise RuntimeError("resident RHF response native identity mismatch")
        self.identity = canonical_hash(
            {
                "schema": "vibeqc.rhf-response-resident/v1",
                "problem": problem.identity,
                "operator": problem.operator_identity,
                "backend": backend.identity,
                "device_id": backend.device_id,
                "vector_slots": vector_slots,
                "owned_device_bytes": diagnostic["owned_device_bytes"],
            }
        )

    def _call(self, name: typing.Any, *args: typing.Any) -> None:
        if self._closed or not self._handle:
            raise RuntimeError("resident RHF response owner is closed")
        # The native adapter borrows the parent's stream and direct-J/K plan.
        # Explicit close must not turn a retained Python reference into a
        # dangling native parent, including a concurrent backend teardown.
        with self._backend._lock:
            self._backend._ensure_open()
            status = getattr(self._lib, f"vibeqc_rhf_response_resident_{name}")(
                self._handle, *args
            )
            if status:
                message = self._lib.vibeqc_rhf_response_resident_last_error(
                    self._handle
                )
                raise RuntimeError(
                    (message or b"resident RHF response failure").decode()
                )

    @property
    def diagnostics(self) -> typing.Any:
        if self._closed or not self._handle:
            raise RuntimeError("resident RHF response owner is closed")
        value = _Diagnostic(
            struct_size=ct.sizeof(_Diagnostic), abi_version=_native.ABI_VERSION
        )
        status = self._lib.vibeqc_rhf_response_resident_get_diagnostic(
            self._handle, ct.byref(value)
        )
        if status:
            raise RuntimeError("resident RHF response diagnostics unavailable")
        return {
            name: int(getattr(value, name))
            for name, _ in value._fields_
            if name not in ("struct_size", "abi_version")
        }

    @property
    def workspace_bytes(self) -> typing.Any:
        return self.diagnostics["owned_device_bytes"]

    @contextmanager
    def solver_workspace(self) -> typing.Any:
        # A returned/failed solver frame can be retained by a profiler or a
        # traceback. Its temporary leases must not depend on garbage collection.
        self.reset()
        try:
            yield
        finally:
            for vector in list(self._vectors.values()):
                vector.release()

    def reset(self) -> None:
        if self._live:
            raise RuntimeError("resident Krylov reset with live vector leases")
        self._free = list(reversed(range(self.vector_slots)))

    def _validate_vector(self, value: typing.Any) -> None:
        if self._closed or not self._handle:
            raise RuntimeError("resident RHF response owner is closed")
        if not isinstance(value, _ResidentVector) or value.owner is not self:
            raise ValueError("resident vector lease belongs to a different owner")
        if value._released or value.slot not in self._live:
            raise ValueError("resident vector lease has been released")

    def _allocate(self) -> typing.Any:
        if self._closed or not self._handle:
            raise RuntimeError("resident RHF response owner is closed")
        if not self._free:
            raise MemoryError("resident RHF Krylov vector-slot capacity exhausted")
        slot = self._free.pop()
        self._live.add(slot)
        vector = _ResidentVector(self, slot)
        self._vectors[slot] = vector
        return vector

    def _release_slot(self, slot: typing.Any) -> None:
        if slot in self._live:
            self._live.remove(slot)
            self._vectors.pop(slot, None)
            self._free.append(slot)

    def from_host(self, values: typing.Any) -> typing.Any:
        values = np.asarray(values)
        if (
            values.shape != (self.dimension,)
            or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()
        ):
            raise ValueError("resident Krylov vectors must be finite real values")
        host = np.ascontiguousarray(values, dtype=np.float64)
        vector = self._allocate()
        try:
            self._call("upload", vector.slot, host.ctypes.data_as(_DOUBLE), host.size)
            return vector
        except BaseException:
            vector.release()
            raise

    def zeros(self) -> typing.Any:
        vector = self._allocate()
        try:
            self._call("zero", vector.slot)
            return vector
        except BaseException:
            vector.release()
            raise

    def copy(self, value: typing.Any) -> typing.Any:
        self._validate_vector(value)
        vector = self._allocate()
        try:
            self._call("copy", vector.slot, value.slot)
            return vector
        except BaseException:
            vector.release()
            raise

    def scale(self, value: typing.Any, alpha: typing.Any) -> typing.Any:
        vector = self.copy(value)
        self._call("scale", vector.slot, float(alpha))
        return vector

    def subtract(self, left: typing.Any, right: typing.Any) -> typing.Any:
        self._validate_vector(left)
        self._validate_vector(right)
        result = self.copy(left)
        self._call("axpy", result.slot, -1.0, right.slot)
        return result

    def dot(self, left: typing.Any, right: typing.Any) -> typing.Any:
        self._validate_vector(left)
        self._validate_vector(right)
        output = ct.c_double()
        self._call("dot", left.slot, right.slot, ct.byref(output))
        return float(output.value)

    def norm(self, value: typing.Any) -> typing.Any:
        self._validate_vector(value)
        output = ct.c_double()
        self._call("norm", value.slot, ct.byref(output))
        return float(output.value)

    def apply(self, operator: typing.Any, value: typing.Any) -> typing.Any:
        self._validate_vector(value)
        del operator
        result = self._allocate()
        try:
            self._call("apply", result.slot, value.slot)
            return result
        except BaseException:
            result.release()
            raise

    def precondition(self, preconditioner: typing.Any, value: typing.Any) -> typing.Any:
        if preconditioner is not None:
            raise NotImplementedError(
                "resident RHF Krylov does not permit a host preconditioner fallback"
            )
        return self.copy(value)

    def orthogonalize(
        self,
        basis: typing.Any,
        value: typing.Any,
        *,
        reorthogonalize: typing.Any,
        tolerance: typing.Any,
    ) -> typing.Any:
        self._validate_vector(value)
        for vector in basis:
            self._validate_vector(vector)
        work = value
        coefficients = np.zeros(len(basis), dtype=np.float64)
        for _ in range(reorthogonalize):
            for column, vector in enumerate(basis):
                projection = self.dot(vector, work)
                coefficients[column] += projection
                self._call("axpy", work.slot, -projection, vector.slot)
            norm = self.norm(work)
            if norm <= tolerance:
                break
        return work, coefficients, self.norm(work)

    def combination(
        self,
        base: typing.Any,
        basis: typing.Any,
        coefficients: typing.Any,
        preconditioner: typing.Any,
    ) -> typing.Any:
        self._validate_vector(base)
        basis = tuple(basis)
        for vector in basis:
            self._validate_vector(vector)
        if preconditioner is not None:
            raise NotImplementedError(
                "resident RHF Krylov does not permit a host preconditioner fallback"
            )
        result = self.copy(base)
        for coefficient, vector in zip(coefficients, basis, strict=True):
            self._call("axpy", result.slot, float(coefficient), vector.slot)
        return result

    def to_host(self, value: typing.Any) -> typing.Any:
        self._validate_vector(value)
        output = np.empty(self.dimension, dtype=np.float64)
        self._call("download", value.slot, output.ctypes.data_as(_DOUBLE), output.size)
        return output

    def stack_host(self, values: typing.Any) -> typing.Any:
        if not values:
            return np.empty((self.dimension, 0))
        return np.column_stack([self.to_host(value) for value in values])

    def close(self) -> None:
        if not self._closed:
            if self._live:
                raise RuntimeError(
                    "cannot close resident RHF response with live vectors"
                )
            self._lib.vibeqc_rhf_response_resident_destroy(self._handle)
            self._handle = ct.c_void_p()
            self._closed = True

    def __enter__(self) -> typing.Any:
        return self

    def __exit__(self, exc_type: object, _exc: object, _traceback: object) -> None:
        if exc_type is not None:
            # Solver traceback frames still own their vectors while unwinding.
            # Revoke these leases before destroying the native owner instead of
            # masking the original failure with a live-vector close error.
            # Any retained vector now refers to a closed owner; later release
            # is harmless because its slot is no longer in the live set.
            self._live.clear()
        elif self._live:
            import gc

            gc.collect()
        self.close()

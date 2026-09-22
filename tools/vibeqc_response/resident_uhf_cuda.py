"""Device-resident unrestricted-HF response vectors and coupled J/K action.

The owner is deliberately a narrow adapter for :class:`UHFResponseOperator`:
the shared GMRES/block/recycle implementation remains the only solver. Alpha
and beta occupied-virtual blocks share one vector lease, while all response
arithmetic stays on the parent's exact unrestricted CUDA stream.
"""

from __future__ import annotations

import ctypes as ct
import typing
from weakref import WeakValueDictionary

import numpy as np
from vibeqc import _native
from vibeqc.profiles import canonical_hash

from .spin_cuda import CudaSpinJKBackend
from .uhf import UHFResponseProblem, uhf_operator_identity

_DOUBLE = ct.POINTER(ct.c_double)


class _Diagnostic(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("abi_version", ct.c_uint32),
        ("nbf", ct.c_uint64),
        ("nocc_alpha", ct.c_uint64),
        ("nvirt_alpha", ct.c_uint64),
        ("nocc_beta", ct.c_uint64),
        ("nvirt_beta", ct.c_uint64),
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
    prefix = "vibeqc_uhf_response_resident_"
    lib.vibeqc_uhf_response_resident_create.argtypes = [
        ct.c_void_p,
        _DOUBLE,
        ct.c_uint64,
        _DOUBLE,
        ct.c_uint64,
        ct.c_uint32,
        _DOUBLE,
        ct.c_uint64,
        _DOUBLE,
        ct.c_uint64,
        ct.c_uint32,
        ct.c_uint32,
        ct.c_uint64,
        ct.POINTER(handle),
    ]
    lib.vibeqc_uhf_response_resident_create.restype = ct.c_int32
    lib.vibeqc_uhf_response_resident_destroy.argtypes = [handle]
    lib.vibeqc_uhf_response_resident_destroy.restype = None
    lib.vibeqc_uhf_response_resident_last_error.argtypes = [handle]
    lib.vibeqc_uhf_response_resident_last_error.restype = ct.c_char_p
    lib.vibeqc_uhf_response_resident_get_diagnostic.argtypes = [
        handle,
        ct.POINTER(_Diagnostic),
    ]
    lib.vibeqc_uhf_response_resident_get_diagnostic.restype = ct.c_int32
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
        function = getattr(lib, f"{prefix}{name}")
        function.argtypes = args
        function.restype = ct.c_int32


class _ResidentVector:
    __slots__ = ("__weakref__", "_released", "owner", "slot")

    def __init__(self, owner: typing.Any, slot: int) -> None:
        self.owner, self.slot, self._released = owner, slot, False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self.owner._release_slot(self.slot)

    def __del__(self) -> None:
        if hasattr(self, "_released"):
            self.release()


class CudaResidentUHFResponse:
    """Bound device vector/operator owner consumed by shared UHF GMRES."""

    resident = True

    def __init__(
        self,
        backend: typing.Any,
        problem: typing.Any,
        *,
        vector_slots: int,
        device_budget_bytes: int,
    ) -> None:
        if not isinstance(backend, CudaSpinJKBackend):
            raise TypeError("resident UHF response requires CudaSpinJKBackend")
        if backend.approximation != "exact":
            raise NotImplementedError(
                "resident UHF response is qualified for exact CUDA J/K only"
            )
        if (
            not isinstance(problem, UHFResponseProblem)
            or problem.reference.algorithm != "UHF"
        ):
            raise TypeError("resident UHF response requires a UHFResponseProblem")
        if problem.operator_identity != uhf_operator_identity(backend):
            raise ValueError("resident UHF problem/operator identity mismatch")
        # The native owner stores canonical occupied/virtual columns directly.
        # Arbitrary reorderings remain valid on the host but must not silently
        # acquire this different device packing.
        for spin in ("alpha", "beta"):
            occupied = problem.reference.nocc(spin)
            if problem.layout.spaces(spin) != (
                tuple(range(occupied)),
                tuple(range(occupied, problem.reference.nbf)),
            ):
                raise ValueError(
                    "resident UHF requires canonical spin rotation ordering"
                )
        if type(vector_slots) is not int or not 8 <= vector_slots <= 4096:
            raise ValueError("resident UHF vector_slots must be in [8,4096]")
        if type(device_budget_bytes) is not int or not 0 < device_budget_bytes < 2**63:
            raise ValueError("resident UHF device budget must be a positive int64")
        backend.validate_reference(problem.reference)
        backend._ensure_open()
        lib = backend._plan._library
        if not hasattr(lib, "vibeqc_uhf_response_resident_create"):
            raise NotImplementedError(
                "native library lacks resident UHF response support"
            )
        _bind(lib)
        self._lib, self._backend, self.problem = lib, backend, problem
        self.dimension = problem.dimension
        self._handle = ct.c_void_p()
        self._closed = False
        reference = problem.reference
        ca = np.ascontiguousarray(reference.coefficients_alpha, dtype=np.float64)
        cb = np.ascontiguousarray(reference.coefficients_beta, dtype=np.float64)
        ea = np.ascontiguousarray(reference.orbital_energies_alpha, dtype=np.float64)
        eb = np.ascontiguousarray(reference.orbital_energies_beta, dtype=np.float64)
        status = lib.vibeqc_uhf_response_resident_create(
            backend._plan._handle,
            ca.ctypes.data_as(_DOUBLE),
            ca.size,
            ea.ctypes.data_as(_DOUBLE),
            ea.size,
            reference.nocc("alpha"),
            cb.ctypes.data_as(_DOUBLE),
            cb.size,
            eb.ctypes.data_as(_DOUBLE),
            eb.size,
            reference.nocc("beta"),
            vector_slots,
            device_budget_bytes,
            ct.byref(self._handle),
        )
        if status:
            message = lib.vibeqc_fock_plan_last_error(backend._plan._handle)
            raise RuntimeError(
                (message or b"resident UHF response creation failed").decode()
            )
        self.vector_slots = vector_slots
        self._free = list(reversed(range(vector_slots)))
        self._live: set[int] = set()
        self._retained: set[int] = set()
        self._vectors = WeakValueDictionary()
        diagnostic = self.diagnostics
        if (
            diagnostic["nbf"] != reference.nbf
            or diagnostic["nocc_alpha"] != reference.nocc("alpha")
            or diagnostic["nocc_beta"] != reference.nocc("beta")
            or diagnostic["dimension"] != problem.dimension
            or diagnostic["device_id"] != backend.device_id
        ):
            self.close()
            raise RuntimeError("resident UHF response native identity mismatch")
        self.identity = canonical_hash(
            {
                "schema": "vibeqc.uhf-response-resident/v1",
                "problem": problem.identity,
                "operator": problem.operator_identity,
                "backend": backend.identity,
                "device_id": backend.device_id,
                "vector_slots": vector_slots,
                "owned_device_bytes": diagnostic["owned_device_bytes"],
            }
        )

    def _call(self, name: str, *args: typing.Any) -> None:
        if self._closed or not self._handle:
            raise RuntimeError("resident UHF response owner is closed")
        with self._backend._lock:
            self._backend._ensure_open()
            status = getattr(self._lib, f"vibeqc_uhf_response_resident_{name}")(
                self._handle, *args
            )
        if status:
            message = self._lib.vibeqc_uhf_response_resident_last_error(self._handle)
            raise RuntimeError((message or b"resident UHF response failure").decode())

    @property
    def diagnostics(self) -> dict[str, int]:
        if self._closed or not self._handle:
            raise RuntimeError("resident UHF response owner is closed")
        value = _Diagnostic(
            struct_size=ct.sizeof(_Diagnostic), abi_version=_native.ABI_VERSION
        )
        status = self._lib.vibeqc_uhf_response_resident_get_diagnostic(
            self._handle, ct.byref(value)
        )
        if status:
            raise RuntimeError("resident UHF response diagnostics unavailable")
        return {
            name: int(getattr(value, name))
            for name, _ in value._fields_
            if name not in ("struct_size", "abi_version")
        }

    @property
    def workspace_bytes(self) -> int:
        return self.diagnostics["owned_device_bytes"]

    def _validate_vector(self, value: typing.Any) -> None:
        if self._closed or not self._handle:
            raise RuntimeError("resident UHF response owner is closed")
        if not isinstance(value, _ResidentVector) or value.owner is not self:
            raise ValueError("resident vector lease belongs to a different owner")
        if value._released or value.slot not in self._live:
            raise ValueError("resident vector lease has been released")

    def _allocate(self) -> _ResidentVector:
        if self._closed or not self._handle:
            raise RuntimeError("resident UHF response owner is closed")
        if not self._free:
            raise MemoryError("resident UHF Krylov vector-slot capacity exhausted")
        slot = self._free.pop()
        self._live.add(slot)
        vector = _ResidentVector(self, slot)
        self._vectors[slot] = vector
        return vector

    def _release_slot(self, slot: int) -> None:
        if slot in self._live:
            self._live.remove(slot)
            self._retained.discard(slot)
            self._vectors.pop(slot, None)
            self._free.append(slot)

    def from_host(self, values: typing.Any) -> _ResidentVector:
        values = np.asarray(values)
        if (
            values.shape != (self.dimension,)
            or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()
        ):
            raise ValueError("resident UHF Krylov vectors must be finite real values")
        host = np.ascontiguousarray(values, dtype=np.float64)
        vector = self._allocate()
        try:
            self._call("upload", vector.slot, host.ctypes.data_as(_DOUBLE), host.size)
            return vector
        except BaseException:
            vector.release()
            raise

    def zeros(self) -> _ResidentVector:
        vector = self._allocate()
        try:
            self._call("zero", vector.slot)
            return vector
        except BaseException:
            vector.release()
            raise

    def copy(self, value: _ResidentVector) -> _ResidentVector:
        self._validate_vector(value)
        vector = self._allocate()
        try:
            self._call("copy", vector.slot, value.slot)
            return vector
        except BaseException:
            vector.release()
            raise

    def scale(self, value: _ResidentVector, alpha: float) -> _ResidentVector:
        vector = self.copy(value)
        self._call("scale", vector.slot, float(alpha))
        return vector

    def subtract(
        self, left: _ResidentVector, right: _ResidentVector
    ) -> _ResidentVector:
        self._validate_vector(left)
        self._validate_vector(right)
        result = self.copy(left)
        self._call("axpy", result.slot, -1.0, right.slot)
        return result

    def dot(self, left: _ResidentVector, right: _ResidentVector) -> float:
        self._validate_vector(left)
        self._validate_vector(right)
        output = ct.c_double()
        self._call("dot", left.slot, right.slot, ct.byref(output))
        return float(output.value)

    def norm(self, value: _ResidentVector) -> float:
        self._validate_vector(value)
        output = ct.c_double()
        self._call("norm", value.slot, ct.byref(output))
        return float(output.value)

    def apply(self, operator: typing.Any, value: _ResidentVector) -> _ResidentVector:
        if (
            operator.problem.compatibility_identity
            != self.problem.compatibility_identity
        ):
            raise ValueError("resident UHF vector owner/operator identity mismatch")
        operator.validate_current()
        self._validate_vector(value)
        result = self._allocate()
        try:
            self._call("apply", result.slot, value.slot)
            return result
        except BaseException:
            result.release()
            raise

    def precondition(
        self, preconditioner: typing.Any, value: _ResidentVector
    ) -> _ResidentVector:
        if preconditioner is not None:
            raise NotImplementedError(
                "resident UHF Krylov does not permit a host preconditioner fallback"
            )
        return self.copy(value)

    def orthogonalize(
        self,
        basis: typing.Any,
        value: _ResidentVector,
        *,
        reorthogonalize: int,
        tolerance: float,
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
        base: _ResidentVector,
        basis: typing.Any,
        coefficients: typing.Any,
        preconditioner: typing.Any,
    ) -> _ResidentVector:
        self._validate_vector(base)
        for vector in basis:
            self._validate_vector(vector)
        if preconditioner is not None:
            raise NotImplementedError(
                "resident UHF Krylov does not permit a host preconditioner fallback"
            )
        result = self.copy(base)
        for coefficient, vector in zip(coefficients, basis, strict=True):
            self._call("axpy", result.slot, float(coefficient), vector.slot)
        return result

    def to_host(self, value: _ResidentVector) -> np.ndarray:
        self._validate_vector(value)
        output = np.empty(self.dimension, dtype=np.float64)
        self._call("download", value.slot, output.ctypes.data_as(_DOUBLE), output.size)
        return output

    def stack_host(self, values: typing.Any) -> np.ndarray:
        if not values:
            return np.empty((self.dimension, 0))
        return np.column_stack([self.to_host(value) for value in values])

    def solver_workspace(self) -> typing.Any:
        from contextlib import contextmanager

        @contextmanager
        def workspace() -> typing.Iterator[None]:
            if self._live != self._retained:
                raise RuntimeError("resident UHF Krylov reset with live vector leases")
            try:
                yield
            finally:
                for vector in list(self._vectors.values()):
                    if vector.slot not in self._retained:
                        vector.release()

        return workspace()

    def reset(self) -> None:
        if self._live != self._retained:
            raise RuntimeError("resident UHF Krylov reset with live vector leases")
        self._free = [
            slot
            for slot in reversed(range(self.vector_slots))
            if slot not in self._live
        ]

    def _retain(self, vector: _ResidentVector) -> None:
        self._validate_vector(vector)
        self._retained.add(vector.slot)

    def close(self) -> None:
        if not self._closed:
            if self._live:
                raise RuntimeError(
                    "cannot close resident UHF response with live vectors"
                )
            self._lib.vibeqc_uhf_response_resident_destroy(self._handle)
            self._handle = ct.c_void_p()
            self._closed = True

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(self, exc_type: object, _exc: object, _traceback: object) -> None:
        if exc_type is not None:
            self._live.clear()
            self._retained.clear()
        elif self._live:
            import gc

            gc.collect()
        self.close()

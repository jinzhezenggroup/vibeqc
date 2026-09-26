"""Device-resident final molecular HVP accumulator for bounded RHF tools."""

from __future__ import annotations

import ctypes as ct
import threading
import typing
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_runtime import _PREPARATION_LOCK
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import file_hash

from tools.vibeqc_posthf.reference import immutable

_DOUBLE = ct.POINTER(ct.c_double)
_UINT32 = ct.POINTER(ct.c_uint32)


class _Diagnostics(ct.Structure):
    _fields_ = [
        ("owned_device_bytes", ct.c_uint64),
        ("h2d_bytes", ct.c_uint64),
        ("d2h_bytes", ct.c_uint64),
        ("synchronizations", ct.c_uint64),
        ("dense_adds", ct.c_uint64),
        ("scatters", ct.c_uint64),
        ("natoms", ct.c_uint64),
    ]


class _MatrixDiagnostics(ct.Structure):
    _fields_ = [
        ("owned_device_bytes", ct.c_uint64),
        ("d2h_bytes", ct.c_uint64),
        ("synchronizations", ct.c_uint64),
        ("column_copies", ct.c_uint64),
        ("rows", ct.c_uint64),
        ("columns", ct.c_uint64),
    ]


def _pointer(array: np.ndarray) -> typing.Any:
    return array.ctypes.data_as(_DOUBLE)


def compile_hvp_assembly(
    compiler: CudaCompilerAdapter, cache: str | Path
) -> typing.Any:
    """Compile one target-specific bounded HVP accumulator through the shared cache."""
    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError("CUDA HVP assembly requires an explicit CudaCompilerAdapter")
    source = asset_path("src/integrals/hvp_assembly_runtime.cu")
    headers = (
        asset_path("src/runtime/bounded_workspace.hpp"),
        asset_path("src/runtime/cuda_resources.cuh"),
        asset_path("src/runtime/resource_cuda.cuh"),
        asset_path("src/runtime/resource_ledger.hpp"),
    )
    return compile_runtime(
        compiler,
        Path(cache),
        source,
        headers=headers,
        options=("--fmad=false", f"-I{source.parents[2] / 'src'}"),
    )


class CudaHVPAccumulator:
    """Accumulate molecular HVP terms on device and publish only the final vector."""

    def __init__(
        self,
        artifact: typing.Any,
        *,
        natoms: int,
        device_id: int = 0,
        budget_bytes: int = 8 << 20,
    ) -> None:
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        if type(natoms) is not int or not 0 < natoms <= 4096:
            raise ValueError("natoms must be a positive bounded integer")
        if type(device_id) is not int or not 0 <= device_id < 2**31:
            raise ValueError("device_id must be a nonnegative int32")
        if type(budget_bytes) is not int or not 0 < budget_bytes < 2**63:
            raise ValueError("budget_bytes must be a positive int64")
        if file_hash(artifact.library) != artifact.metadata["binary_sha256"]:
            raise ValueError("CUDA HVP assembly binary hash mismatch")
        self.natoms = natoms
        self.device_id = device_id
        self.artifact = artifact
        target = artifact.metadata["identity"]["target"]
        lib = self._library = ct.CDLL(str(artifact.library))
        tail = [ct.c_char_p, ct.c_size_t]
        lib.vibeqc_hvp_assembly_create_v1.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
            *tail,
        ]
        lib.vibeqc_hvp_assembly_destroy_v1.argtypes = [ct.c_void_p]
        lib.vibeqc_hvp_assembly_destroy_v1.restype = None
        lib.vibeqc_hvp_assembly_output_device_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_void_p),
            ct.POINTER(ct.c_size_t),
            *tail,
        ]
        lib.vibeqc_hvp_assembly_reset_nuclear_v1.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            _DOUBLE,
            _DOUBLE,
            ct.c_size_t,
            *tail,
        ]
        lib.vibeqc_hvp_assembly_add_dense_v1.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            ct.c_size_t,
            ct.c_double,
            *tail,
        ]
        lib.vibeqc_hvp_assembly_scatter_v1.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            _UINT32,
            ct.c_size_t,
            _UINT32,
            ct.c_size_t,
            ct.c_double,
            *tail,
        ]
        lib.vibeqc_hvp_assembly_download_v1.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            ct.c_size_t,
            *tail,
        ]
        lib.vibeqc_hvp_assembly_diagnostics_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(_Diagnostics),
            *tail,
        ]
        with _PREPARATION_LOCK:
            self._call(
                "create",
                device_id,
                target["compute_capability_major"],
                target["compute_capability_minor"],
                natoms,
                budget_bytes,
                ct.byref(self._handle),
            )

    def _call(self, name: str, *args: typing.Any) -> None:
        detail = ct.create_string_buffer(2048)
        status = getattr(self._library, f"vibeqc_hvp_assembly_{name}_v1")(
            *args, detail, len(detail)
        )
        if status:
            raise {1: ValueError, 5: FloatingPointError, 7: MemoryError}.get(
                status, RuntimeError
            )(detail.value.decode() or f"CUDA HVP assembly status {status}")

    def _ensure_open(self) -> None:
        if not self._handle:
            raise RuntimeError("CUDA HVP accumulator is closed")

    def reset_nuclear(
        self, coordinates: typing.Any, charges: typing.Any, direction: typing.Any
    ) -> None:
        with self._lock:
            self._ensure_open()
            coords = np.ascontiguousarray(coordinates, dtype=np.float64)
            z = np.ascontiguousarray(charges, dtype=np.float64)
            vector = np.ascontiguousarray(direction, dtype=np.float64)
            if (
                coords.shape != (self.natoms, 3)
                or z.shape != (self.natoms,)
                or vector.shape != (self.natoms, 3)
                or not np.isfinite(coords).all()
                or not np.isfinite(z).all()
                or not np.isfinite(vector).all()
            ):
                raise ValueError("invalid finite nuclear HVP inputs")
            self._call(
                "reset_nuclear",
                self._handle,
                _pointer(coords),
                _pointer(z),
                _pointer(vector),
                self.natoms,
            )

    def add_device(self, device_pointer: int, *, coefficient: float = 1.0) -> None:
        with self._lock:
            self._ensure_open()
            if type(device_pointer) is not int or device_pointer <= 0:
                raise ValueError("device contribution pointer must be positive")
            self._call(
                "add_dense",
                self._handle,
                ct.cast(ct.c_void_p(device_pointer), _DOUBLE),
                3 * self.natoms,
                float(coefficient),
            )

    def scatter_device(
        self,
        device_pointer: int,
        output_indices: typing.Any,
        center_atoms: typing.Any,
        *,
        coefficient: float = 1.0,
    ) -> None:
        with self._lock:
            self._ensure_open()
            if type(device_pointer) is not int or device_pointer <= 0:
                raise ValueError("compact device contribution pointer must be positive")
            indices = np.ascontiguousarray(output_indices, dtype=np.uint32)
            atoms = np.ascontiguousarray(center_atoms, dtype=np.uint32)
            if (
                indices.ndim != 1
                or not 1 <= indices.size <= 12
                or atoms.ndim != 1
                or not 1 <= atoms.size <= 4
            ):
                raise ValueError("invalid compact HVP mapping")
            self._call(
                "scatter",
                self._handle,
                ct.cast(ct.c_void_p(device_pointer), _DOUBLE),
                indices.ctypes.data_as(_UINT32),
                indices.size,
                atoms.ctypes.data_as(_UINT32),
                atoms.size,
                float(coefficient),
            )

    def device_output(self) -> tuple[int, int]:
        """Borrow the validated final HVP until this owner is mutated or closed."""
        with self._lock:
            self._ensure_open()
            pointer = ct.c_void_p()
            count = ct.c_size_t()
            self._call(
                "output_device",
                self._handle,
                ct.byref(pointer),
                ct.byref(count),
            )
            if not pointer.value or count.value != 3 * self.natoms:
                raise RuntimeError("CUDA HVP device output contract mismatch")
            return int(pointer.value), int(count.value)

    def finish(self) -> np.ndarray:
        with self._lock:
            self._ensure_open()
            output = np.empty((self.natoms, 3), dtype=np.float64)
            self._call("download", self._handle, _pointer(output), output.size)
            return immutable(output)

    @property
    def diagnostics(self) -> dict[str, int]:
        with self._lock:
            self._ensure_open()
            record = _Diagnostics()
            self._call("diagnostics", self._handle, ct.byref(record))
            return {name: int(getattr(record, name)) for name, _ in record._fields_}

    def close(self) -> None:
        with self._lock, _PREPARATION_LOCK:
            if self._handle:
                self._library.vibeqc_hvp_assembly_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self) -> typing.Self:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()


class CudaHessianAccumulator:
    """Collect validated device HVP columns and publish one full Hessian."""

    def __init__(
        self,
        artifact: typing.Any,
        *,
        coordinates: int,
        device_id: int = 0,
        budget_bytes: int = 16 << 20,
    ) -> None:
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        if type(coordinates) is not int or not 0 < coordinates <= 12288:
            raise ValueError("coordinates must be a positive bounded integer")
        if type(device_id) is not int or not 0 <= device_id < 2**31:
            raise ValueError("device_id must be a nonnegative int32")
        if type(budget_bytes) is not int or not 0 < budget_bytes < 2**63:
            raise ValueError("budget_bytes must be a positive int64")
        if file_hash(artifact.library) != artifact.metadata["binary_sha256"]:
            raise ValueError("CUDA Hessian assembly binary hash mismatch")
        self.coordinates = coordinates
        self.device_id = device_id
        self.artifact = artifact
        target = artifact.metadata["identity"]["target"]
        lib = self._library = ct.CDLL(str(artifact.library))
        tail = [ct.c_char_p, ct.c_size_t]
        lib.vibeqc_hessian_assembly_create_v1.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
            *tail,
        ]
        lib.vibeqc_hessian_assembly_destroy_v1.argtypes = [ct.c_void_p]
        lib.vibeqc_hessian_assembly_destroy_v1.restype = None
        lib.vibeqc_hessian_assembly_reset_v1.argtypes = [ct.c_void_p, *tail]
        lib.vibeqc_hessian_assembly_copy_column_v1.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            ct.c_size_t,
            ct.c_size_t,
            *tail,
        ]
        lib.vibeqc_hessian_assembly_download_v1.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            ct.c_size_t,
            *tail,
        ]
        lib.vibeqc_hessian_assembly_diagnostics_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(_MatrixDiagnostics),
            *tail,
        ]
        with _PREPARATION_LOCK:
            self._call(
                "create",
                device_id,
                target["compute_capability_major"],
                target["compute_capability_minor"],
                coordinates,
                coordinates,
                budget_bytes,
                ct.byref(self._handle),
            )
        self._call("reset", self._handle)

    def _call(self, name: str, *args: typing.Any) -> None:
        detail = ct.create_string_buffer(2048)
        status = getattr(self._library, f"vibeqc_hessian_assembly_{name}_v1")(
            *args, detail, len(detail)
        )
        if status:
            raise {1: ValueError, 5: FloatingPointError, 7: MemoryError}.get(
                status, RuntimeError
            )(detail.value.decode() or f"CUDA Hessian assembly status {status}")

    def _ensure_open(self) -> None:
        if not self._handle:
            raise RuntimeError("CUDA Hessian accumulator is closed")

    def copy_column(self, device_pointer: int, count: int, column: int) -> None:
        with self._lock:
            self._ensure_open()
            if type(device_pointer) is not int or device_pointer <= 0:
                raise ValueError("Hessian source pointer must be positive")
            if count != self.coordinates:
                raise ValueError("Hessian source column size mismatch")
            if type(column) is not int or not 0 <= column < self.coordinates:
                raise ValueError("Hessian column index out of range")
            self._call(
                "copy_column",
                self._handle,
                ct.cast(ct.c_void_p(device_pointer), _DOUBLE),
                count,
                column,
            )

    def finish(self) -> np.ndarray:
        with self._lock:
            self._ensure_open()
            # Native storage is column-major so each HVP can be copied contiguously.
            output = np.empty(
                (self.coordinates, self.coordinates),
                dtype=np.float64,
                order="F",
            )
            self._call("download", self._handle, _pointer(output), output.size)
            return immutable(np.array(output, copy=True))

    @property
    def diagnostics(self) -> dict[str, int]:
        with self._lock:
            self._ensure_open()
            record = _MatrixDiagnostics()
            self._call("diagnostics", self._handle, ct.byref(record))
            return {name: int(getattr(record, name)) for name, _ in record._fields_}

    def close(self) -> None:
        with self._lock, _PREPARATION_LOCK:
            if self._handle:
                self._library.vibeqc_hessian_assembly_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self) -> typing.Self:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()

"""Explicit CUDA compilation and one resident weighted-gradient accumulator."""

from __future__ import annotations

import ctypes as ct
import math
import os
import tempfile
import threading
import typing
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_runtime import _PREPARATION_LOCK, CudaArtifact
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.common.resources import byte_product, checked_bytes

from .first_gradient import (
    FirstGradientTerm,
    emit_first_gradient,
    first_gradient_identity,
)

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike
    from typing_extensions import Self

    from .ir import IntegralIR

_HEADERS = (
    "src/integrals/first_gradient_runtime.cuh",
    "src/integrals/eri_geometry.hpp",
    "src/integrals/range_moments.hpp",
    "src/tensor/cuda_runtime.cuh",
    "src/runtime/bounded_workspace.hpp",
    "src/runtime/cuda_resources.cuh",
    "src/runtime/resource_cuda.cuh",
    "src/runtime/resource_ledger.hpp",
    "src/tensor/cuda_error.hpp",
    "src/tensor/metrics.hpp",
    "src/runtime/allocation_measurement.hpp",
)


@dataclass(frozen=True)
class CompiledFirstGradient:
    native: CudaArtifact
    integral: IntegralIR
    component_indices: tuple
    terms: tuple
    program_identity: str
    runtime_identity: str
    target: tuple

    def validate(self, *, check_binary: bool = True) -> None:
        if (
            first_gradient_identity(self.integral, self.component_indices, self.terms)
            != self.program_identity
        ):
            raise ValueError("first-gradient artifact mathematical identity mismatch")
        meta = self.native.metadata
        target = meta["identity"]["target"]
        if (
            canonical_hash(meta["identity"]) != meta["key"]
            or (
                check_binary and file_hash(self.native.library) != meta["binary_sha256"]
            )
            or self.target
            != (target["compute_capability_major"], target["compute_capability_minor"])
        ):
            raise ValueError("first-gradient artifact native identity mismatch")


def compile_first_gradient(
    integral: IntegralIR,
    compiler: CudaCompilerAdapter,
    cache: str | os.PathLike[str],
    *,
    component_indices: Sequence[int],
    terms: Sequence[FirstGradientTerm],
) -> CompiledFirstGradient:
    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError(
            "first-gradient device lowering requires an explicit CUDA compiler"
        )
    if os.environ.get("NVCC_PREPEND_FLAGS") or os.environ.get("NVCC_APPEND_FLAGS"):
        raise ValueError("first-gradient strict CUDA rejects NVCC flag overrides")
    indices, terms = tuple(component_indices), tuple(terms)
    identity = first_gradient_identity(integral, indices, terms)
    headers = tuple(asset_path(name) for name in _HEADERS)
    runtime = canonical_hash(
        {name: file_hash(path) for name, path in zip(_HEADERS, headers, strict=True)}
    )
    source = emit_first_gradient(integral, indices, terms, runtime_identity=runtime)
    directory = Path(cache).resolve() / "generated-sources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (canonical_hash({"source": source}) + ".cu")
    if not path.exists():
        with tempfile.NamedTemporaryFile(
            mode="w", dir=directory, delete=False
        ) as stream:
            stream.write(source)
            temporary = Path(stream.name)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    elif path.read_text() != source:
        raise ValueError("first-gradient source cache integrity failure")
    root = headers[0].parents[2]
    native = compile_runtime(
        compiler,
        Path(cache) / "native",
        path,
        headers=headers,
        libraries=("cublas",),
        options=("--fmad=false", f"-I{root / 'src'}"),
    )
    return CompiledFirstGradient(
        native,
        integral,
        indices,
        terms,
        identity,
        runtime,
        (
            compiler.target.compute_capability_major,
            compiler.target.compute_capability_minor,
        ),
    )


_DOUBLE = ct.POINTER(ct.c_double)


class _Mapping(ct.Structure):
    _fields_ = [("offsets", ct.c_size_t * 4), ("atoms", ct.c_size_t * 4)]


def _pointer(array: np.ndarray) -> typing.Any:
    return array.ctypes.data_as(_DOUBLE)


def _bind(artifact: CompiledFirstGradient) -> ct.CDLL:
    artifact.validate()
    lib = ct.CDLL(str(artifact.native.library))
    for name, expected in (
        ("identity", artifact.program_identity),
        ("abi", artifact.runtime_identity),
    ):
        function = getattr(lib, f"vibeqc_first_gradient_{name}_v1")
        function.restype = ct.c_char_p
        if function().decode() != expected:
            raise ValueError("first-gradient compiled identity mismatch")
    tail = [ct.c_char_p, ct.c_size_t]
    lib.vibeqc_first_gradient_create_v1.argtypes = (
        [ct.c_int] * 3 + [ct.c_size_t] * 5 + [ct.POINTER(ct.c_void_p)] + tail
    )
    lib.vibeqc_first_gradient_reset_v1.argtypes = [
        ct.c_void_p,
        _DOUBLE,
        ct.c_size_t,
    ] + tail
    lib.vibeqc_first_gradient_reset_mixed_v1.argtypes = [
        ct.c_void_p,
        _DOUBLE,
        ct.c_size_t,
        _DOUBLE,
        ct.c_size_t,
    ] + tail
    lib.vibeqc_first_gradient_append_v1.argtypes = [
        ct.c_void_p,
        _DOUBLE,
        ct.c_size_t,
        ct.POINTER(_Mapping),
    ] + tail
    lib.vibeqc_first_gradient_finish_v1.argtypes = [
        ct.c_void_p,
        _DOUBLE,
        ct.c_size_t,
    ] + tail
    lib.vibeqc_first_gradient_destroy_v1.argtypes = [ct.c_void_p]
    lib.vibeqc_first_gradient_destroy_v1.restype = None
    return lib


def first_gradient_storage(
    nbf: int, natoms: int, weight_slots: int, capacity: int
) -> dict[str, int]:
    for value in (nbf, natoms, weight_slots, capacity):
        if type(value) is not int or value < 1:
            raise ValueError("first-gradient dimensions must be positive integers")
    if weight_slots > 8 or capacity > 2**20:
        raise ValueError(
            "first-gradient weight/record capacity exceeds supported range"
        )
    matrix = byte_product(nbf, nbf)
    weights = byte_product(weight_slots, matrix, 8)
    output = byte_product(natoms, 3, 8)
    record = byte_product(capacity, 17, 8)
    device = checked_bytes(weights + output + record + 8)
    peak = checked_bytes(device + record + 2 * weights + 3 * output + 192)
    return {
        "device_bytes": device,
        "numeric_peak_bytes": peak,
        "weight_bytes": weights,
        "output_bytes": output,
        "record_bytes": record,
    }


def _array(value: ArrayLike, shape: tuple[int, ...], name: str) -> typing.Any:
    array = np.asarray(value)
    if (
        array.shape != shape
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    array = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} is not representable in FP64")
    return array


class FirstGradientAccumulator:
    """Shared bounded device gradient across shell programs and input chunks."""

    def __init__(
        self,
        artifact: CompiledFirstGradient,
        *,
        nbf: int,
        natoms: int,
        weight_slots: int,
        capacity: int = 128,
        device_id: int = 0,
        budget_bytes: int = 64 << 20,
    ) -> None:
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self._failed = True
        if not isinstance(artifact, CompiledFirstGradient):
            raise TypeError("expected CompiledFirstGradient")
        self.storage = first_gradient_storage(nbf, natoms, weight_slots, capacity)
        if type(device_id) is not int or not 0 <= device_id < 2**31:
            raise ValueError("device_id must fit nonnegative int32")
        checked_bytes(budget_bytes)
        if self.storage["numeric_peak_bytes"] > budget_bytes:
            raise MemoryError(
                "first-gradient numeric budget exceeded before allocation"
            )
        self.nbf, self.natoms, self.weight_slots, self.capacity = (
            nbf,
            natoms,
            weight_slots,
            capacity,
        )
        self.runtime_identity, self.target = artifact.runtime_identity, artifact.target
        meta = artifact.native.metadata["identity"]
        self._host_abi = (meta["host_compiler"], meta["host_version"])
        self._library = _bind(artifact)
        self._libraries = {
            artifact.native.metadata["key"]: (self._library, artifact.program_identity)
        }
        self.records = np.empty((capacity, 17), dtype=np.float64)
        with _PREPARATION_LOCK:
            self._call(
                self._library,
                "create",
                device_id,
                *self.target,
                nbf,
                natoms,
                weight_slots,
                capacity,
                budget_bytes,
                ct.byref(self._handle),
            )
        self.statistics = {
            "chunks": 0,
            "primitive_records": 0,
            "weight_uploads": 0,
            "resident_weight_imports": 0,
            "gradient_downloads": 0,
        }

    def _call(self, library: ct.CDLL, name: str, *args: typing.Any) -> None:
        detail = ct.create_string_buffer(2048)
        status = getattr(library, f"vibeqc_first_gradient_{name}_v1")(
            *args, detail, len(detail)
        )
        if status:
            raise {1: ValueError, 5: FloatingPointError, 7: MemoryError}.get(
                status, RuntimeError
            )(detail.value.decode())

    def _ensure_open(self) -> None:
        if not self._handle:
            raise RuntimeError("first-gradient accumulator is closed")

    def reset(self, weights: ArrayLike) -> None:
        with self._lock:
            self._ensure_open()
            self._failed = True
            array = _array(
                weights,
                (self.weight_slots, self.nbf, self.nbf),
                "first-gradient weights",
            )
            self._call(
                self._library,
                "reset",
                self._handle,
                _pointer(array),
                array.size,
            )
            self.statistics["weight_uploads"] += 1
            self._failed = False

    def reset_device_prefix(
        self,
        device_pointer: int,
        device_count: int,
        host_suffix: ArrayLike,
    ) -> None:
        """Import a validated device prefix and append host-owned weight matrices."""
        with self._lock:
            self._ensure_open()
            self._failed = True
            if type(device_pointer) is not int or device_pointer <= 0:
                raise ValueError(
                    "first-gradient device weight pointer must be positive"
                )
            if type(device_count) is not int or not 0 < device_count < (
                self.weight_slots * self.nbf * self.nbf
            ):
                raise ValueError("first-gradient device weight count is invalid")
            expected = self.weight_slots * self.nbf * self.nbf - device_count
            suffix = np.asarray(host_suffix)
            if (
                suffix.size != expected
                or suffix.dtype.kind not in "iuf"
                or not np.isfinite(suffix).all()
            ):
                raise ValueError(
                    "first-gradient host suffix has invalid shape or values"
                )
            suffix = np.ascontiguousarray(suffix, dtype=np.float64).reshape(-1)
            self._call(
                self._library,
                "reset_mixed",
                self._handle,
                ct.cast(ct.c_void_p(device_pointer), _DOUBLE),
                device_count,
                _pointer(suffix),
                suffix.size,
            )
            self.statistics["weight_uploads"] += 1
            self.statistics["resident_weight_imports"] += 1
            self._failed = False

    def append_shell(
        self,
        artifact: CompiledFirstGradient,
        primitives: Sequence[Sequence[tuple[float, float]]],
        centers: ArrayLike,
        *,
        offsets: Sequence[int],
        atoms: Sequence[int],
    ) -> None:
        with self._lock:
            self._ensure_open()
            if self._failed:
                raise RuntimeError(
                    "first-gradient accumulator requires successful reset"
                )
            self._failed = True
            if not isinstance(artifact, CompiledFirstGradient):
                raise TypeError("expected CompiledFirstGradient")
            if (artifact.runtime_identity, artifact.target) != (
                self.runtime_identity,
                self.target,
            ):
                raise ValueError("first-gradient runtime/target incompatibility")
            native_identity = artifact.native.metadata["identity"]
            if (
                native_identity["host_compiler"],
                native_identity["host_version"],
            ) != self._host_abi:
                raise ValueError("first-gradient host compiler ABI incompatibility")
            key = artifact.native.metadata["key"]
            artifact.validate(check_binary=False)
            bound = self._libraries.get(key)
            if bound is None:
                bound = (_bind(artifact), artifact.program_identity)
                self._libraries[key] = bound
            library, program_identity = bound
            if artifact.program_identity != program_identity:
                raise ValueError("first-gradient cached program identity mismatch")
            ns = len(artifact.integral.signature.shells)
            nc = len(artifact.integral.operator.centers)
            if len(primitives) != ns or any(not primitive for primitive in primitives):
                raise ValueError("primitive lists must match the compiled shell slots")
            if any(np.iscomplexobj(primitive) for primitive in primitives):
                raise ValueError("primitive exponents/coefficients must be real")
            coords = _array(centers, (nc, 3), "centers")
            offsets, atoms = tuple(offsets), tuple(atoms)
            if (
                len(offsets) != ns
                or len(atoms) != nc
                or any(
                    type(index) is not int or index < 0 for index in (*offsets, *atoms)
                )
            ):
                raise ValueError(
                    "first-gradient mapping must contain nonnegative integer indices"
                )
            if any(
                offset + extent > self.nbf
                for offset, extent in zip(
                    offsets, artifact.integral.signature.component_shape, strict=True
                )
            ) or any(atom >= self.natoms for atom in atoms):
                raise ValueError("first-gradient mapping out of bounds")
            mapping = _Mapping((ct.c_size_t * 4)(*offsets), (ct.c_size_t * 4)(*atoms))
            self.records.fill(0)
            self.records[:, 4 : 4 + nc * 3] = coords.ravel()
            count = 0

            def flush() -> None:
                self._call(
                    library,
                    "append",
                    self._handle,
                    _pointer(self.records),
                    count,
                    ct.byref(mapping),
                )
                self.statistics["chunks"] += 1
                self.statistics["primitive_records"] += count

            for combination in product(*primitives):
                exponents, coefficients = zip(*combination, strict=True)
                self.records[count, :ns] = exponents
                self.records[count, 16] = math.prod(coefficients)
                count += 1
                if count == self.capacity:
                    flush()
                    count = 0
            if count:
                flush()
            self._failed = False

    def finish(self) -> np.ndarray:
        with self._lock:
            self._ensure_open()
            if self._failed:
                raise RuntimeError(
                    "first-gradient accumulator requires successful reset"
                )
            self._failed = True
            output = np.empty((self.natoms, 3), dtype=np.float64)
            self._call(
                self._library,
                "finish",
                self._handle,
                _pointer(output),
                output.size,
            )
            self.statistics["gradient_downloads"] += 1
            self._failed = False
            return immutable(output)

    def close(self) -> None:
        with self._lock:
            if self._handle:
                with _PREPARATION_LOCK:
                    self._library.vibeqc_first_gradient_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()

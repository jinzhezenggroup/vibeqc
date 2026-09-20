"""Explicit CUDA compilation and one resident directional matrix accumulator.

Uses the shared compiler cache and TensorIR device owner. Host work validates
and stages primitive geometry/normalization only; all direction/external-weight
contractions and matrix reductions execute in the generated device program.
"""

import ctypes as ct
import math
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

from .first_directional import directional_identity, emit_directional_matrix
from .ir import IntegralIR

_HEADERS = (
    "src/integrals/first_directional_runtime.cuh",
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
class CompiledDirectionalFirst:
    native: CudaArtifact
    integral: IntegralIR
    component_indices: tuple
    terms: tuple
    program_identity: str
    runtime_identity: str
    target: tuple

    def validate(self, *, check_binary: typing.Any = True) -> None:
        if (
            directional_identity(self.integral, self.component_indices, self.terms)
            != self.program_identity
        ):
            raise ValueError("directional artifact mathematical identity mismatch")
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
            raise ValueError("directional artifact native identity mismatch")


def compile_directional_first(
    integral: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
    *,
    component_indices: typing.Any,
    terms: typing.Any,
) -> typing.Any:
    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError(
            "directional device lowering requires an explicit CUDA compiler"
        )
    # Cache flags must not advertise strict FP64 while an external override
    # injects fast math/FTZ. Match the compiler's explicit scientific policy.
    import os

    if os.environ.get("NVCC_PREPEND_FLAGS") or os.environ.get("NVCC_APPEND_FLAGS"):
        raise ValueError("directional strict CUDA rejects NVCC flag overrides")
    indices, terms = tuple(component_indices), tuple(terms)
    identity = directional_identity(integral, indices, terms)
    headers = tuple(asset_path(name) for name in _HEADERS)
    runtime = canonical_hash(
        {name: file_hash(path) for name, path in zip(_HEADERS, headers, strict=True)}
    )
    source = emit_directional_matrix(integral, indices, terms, runtime_identity=runtime)
    directory = Path(cache).resolve() / "generated-sources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (canonical_hash({"source": source}) + ".cu")
    # A content-addressed file is immutable; atomic replacement avoids readers
    # observing a partial file from a concurrent identical compiler request.
    if not path.exists():
        import tempfile

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
        raise ValueError("directional source cache integrity failure")
    root = headers[0].parents[2]
    native = compile_runtime(
        compiler,
        Path(cache) / "native",
        path,
        headers=headers,
        libraries=("cublas",),
        options=("--fmad=false", f"-I{root / 'src'}"),
    )
    return CompiledDirectionalFirst(
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


def _pointer(a: typing.Any) -> typing.Any:
    return a.ctypes.data_as(_DOUBLE)


def _bind(artifact: typing.Any) -> typing.Any:
    artifact.validate()
    lib = ct.CDLL(str(artifact.native.library))
    for name, expected in (
        ("identity", artifact.program_identity),
        ("abi", artifact.runtime_identity),
    ):
        function = getattr(lib, f"vibeqc_directional_{name}_v1")
        function.restype = ct.c_char_p
        if function().decode() != expected:
            raise ValueError("directional compiled identity mismatch")
    tail = [ct.c_char_p, ct.c_size_t]
    lib.vibeqc_directional_create_v1.argtypes = (
        [ct.c_int] * 3 + [ct.c_size_t] * 5 + [ct.POINTER(ct.c_void_p)] + tail
    )
    lib.vibeqc_directional_reset_v1.argtypes = [
        ct.c_void_p,
        _DOUBLE,
        ct.c_size_t,
        _DOUBLE,
        ct.c_size_t,
    ] + tail
    lib.vibeqc_directional_append_v1.argtypes = [
        ct.c_void_p,
        _DOUBLE,
        ct.c_size_t,
        ct.POINTER(_Mapping),
    ] + tail
    lib.vibeqc_directional_finish_v1.argtypes = [
        ct.c_void_p,
        _DOUBLE,
        ct.c_size_t,
    ] + tail
    lib.vibeqc_directional_destroy_v1.argtypes = [ct.c_void_p]
    lib.vibeqc_directional_destroy_v1.restype = None
    return lib


def directional_storage(
    nbf: typing.Any, natoms: typing.Any, outputs: typing.Any, capacity: typing.Any
) -> typing.Any:
    for x in (nbf, natoms, outputs, capacity):
        if type(x) is not int or x < 1:
            raise ValueError("directional dimensions must be positive integers")
    if outputs > 32 or capacity > 2**20:
        raise ValueError("directional output/record capacity exceeds supported range")
    matrix = byte_product(nbf, nbf)
    record = byte_product(capacity, 17, 8)
    output = byte_product(outputs, matrix, 8)
    inputs = checked_bytes(byte_product(matrix, 8) + byte_product(natoms, 3, 8))
    device = checked_bytes(record + output + inputs + 8)
    # Native publication candidate + Python candidate + immutable output copy;
    # retained host records + owned reset snapshots coexist with the arena.
    # Account for conversion plus owned input snapshots and bounded center
    # staging, including Python-list inputs and three-center attraction records.
    peak = checked_bytes(device + record + 3 * output + 2 * inputs + 192)
    return {
        "device_bytes": device,
        "numeric_peak_bytes": peak,
        "output_bytes": output,
        "record_bytes": record,
        "input_bytes": inputs,
    }


def _array(value: typing.Any, shape: typing.Any, name: typing.Any) -> typing.Any:
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "iuf" or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    a = np.array(a, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} is not representable in FP64")
    return a


class DirectionalFirstAccumulator:
    """Shared bounded device outputs across shell programs and input chunks.

    Only final compact matrices are downloaded. On any failure, finish is
    blocked until reset; reset permits a clean replay with new weights/direction.
    Caller metadata, compiled artifacts, native call stacks and CUDA context
    overhead are explicit exclusions from the numeric storage budget.
    """

    def __init__(
        self,
        artifact: typing.Any,
        *,
        nbf: typing.Any,
        natoms: typing.Any,
        outputs: typing.Any = 2,
        capacity: typing.Any = 128,
        device_id: typing.Any = 0,
        budget_bytes: typing.Any = 64 << 20,
    ) -> None:
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self._failed = True
        if not isinstance(artifact, CompiledDirectionalFirst):
            raise TypeError("expected CompiledDirectionalFirst")
        self.storage = directional_storage(nbf, natoms, outputs, capacity)
        if type(device_id) is not int or not 0 <= device_id < 2**31:
            raise ValueError("device_id must fit nonnegative int32")
        checked_bytes(budget_bytes)
        if self.storage["numeric_peak_bytes"] > budget_bytes:
            raise MemoryError("directional numeric budget exceeded before allocation")
        self.nbf, self.natoms, self.outputs, self.capacity = (
            nbf,
            natoms,
            outputs,
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
                outputs,
                capacity,
                budget_bytes,
                ct.byref(self._handle),
            )
        self.statistics = {"chunks": 0, "primitive_records": 0, "matrix_downloads": 0}

    def _call(self, library: typing.Any, name: typing.Any, *args: typing.Any) -> None:
        detail = ct.create_string_buffer(2048)
        status = getattr(library, f"vibeqc_directional_{name}_v1")(
            *args, detail, len(detail)
        )
        if status:
            raise {1: ValueError, 5: FloatingPointError, 7: MemoryError}.get(
                status, RuntimeError
            )(detail.value.decode())

    def _ensure_open(self) -> None:
        if not self._handle:
            raise RuntimeError("directional accumulator is closed")

    def reset(self, weights: typing.Any, direction: typing.Any) -> None:
        with self._lock:
            self._ensure_open()
            self._failed = True
            w = _array(weights, (self.nbf, self.nbf), "external weights")
            v = _array(direction, (self.natoms, 3), "direction")
            self._call(
                self._library,
                "reset",
                self._handle,
                _pointer(w),
                w.size,
                _pointer(v),
                v.size,
            )
            self._failed = False

    def append_shell(
        self,
        artifact: typing.Any,
        primitives: typing.Any,
        centers: typing.Any,
        *,
        offsets: typing.Any,
        atoms: typing.Any,
    ) -> None:
        with self._lock:
            self._ensure_open()
            if self._failed:
                raise RuntimeError("directional accumulator requires successful reset")
            self._failed = True
            if not isinstance(artifact, CompiledDirectionalFirst):
                raise TypeError("expected CompiledDirectionalFirst")
            if (artifact.runtime_identity, artifact.target) != (
                self.runtime_identity,
                self.target,
            ):
                raise ValueError("directional runtime/target incompatibility")
            native_identity = artifact.native.metadata["identity"]
            if (
                native_identity["host_compiler"],
                native_identity["host_version"],
            ) != self._host_abi:
                raise ValueError("directional host compiler ABI incompatibility")
            key = artifact.native.metadata["key"]
            artifact.validate(check_binary=False)
            bound = self._libraries.get(key)
            if bound is None:
                bound = (_bind(artifact), artifact.program_identity)
                self._libraries[key] = bound
            lib, program_identity = bound
            if artifact.program_identity != program_identity:
                raise ValueError("directional cached program identity mismatch")
            ns = len(artifact.integral.signature.shells)
            nc = len(artifact.integral.operator.centers)
            if len(primitives) != ns or any(not p for p in primitives):
                raise ValueError("primitive lists must match the compiled shell slots")
            if any(np.iscomplexobj(p) for p in primitives):
                raise ValueError("primitive exponents/coefficients must be real")
            coords = _array(centers, (nc, 3), "centers")
            offsets, atoms = tuple(offsets), tuple(atoms)
            if (
                len(offsets) != ns
                or len(atoms) != nc
                or any(type(i) is not int or i < 0 for i in (*offsets, *atoms))
            ):
                raise ValueError(
                    "directional mapping must contain nonnegative integer indices"
                )
            if any(
                o + extent > self.nbf
                for o, extent in zip(
                    offsets, artifact.integral.signature.component_shape, strict=True
                )
            ) or any(a >= self.natoms for a in atoms):
                raise ValueError("directional mapping out of bounds")
            mapping = _Mapping((ct.c_size_t * 4)(*offsets), (ct.c_size_t * 4)(*atoms))
            self.records.fill(0)
            self.records[:, 4 : 4 + nc * 3] = coords.ravel()
            count = 0

            def flush() -> None:
                self._call(
                    lib,
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

    def finish(self) -> typing.Any:
        with self._lock:
            self._ensure_open()
            if self._failed:
                raise RuntimeError("directional accumulator requires successful reset")
            self._failed = True
            output = np.empty((self.outputs, self.nbf, self.nbf))
            self._call(
                self._library, "finish", self._handle, _pointer(output), output.size
            )
            self.statistics["matrix_downloads"] += 1
            self._failed = False
            return immutable(output)

    def close(self) -> None:
        with self._lock:
            if self._handle:
                with _PREPARATION_LOCK:
                    self._library.vibeqc_directional_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self) -> typing.Any:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()

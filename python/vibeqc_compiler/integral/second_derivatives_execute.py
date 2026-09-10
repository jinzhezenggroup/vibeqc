"""Explicit compiled, bounded execution of fixed-weight integral Hessian tiles.

The primitive interface consumes normalized Cartesian weights and mathematical
centers. Public-basis pullbacks and molecular response assembly are separate
consumer operations. Compilation, resource planning and native storage reuse
their shared owners; importing this module requires no reference engine.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_runtime import (
    _PREPARATION_LOCK,
    CudaArtifact,
    _Metrics,
)
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    checked_bytes,
    plan_resources,
)

from .ir import IntegralIR
from .ir_serialization import integral_to_payload
from .second_derivatives import build_second_derivative_kernel, require_second_consumer
from .second_derivatives_native import (
    SECOND_RECORD_TAG,
    emit_second_derivative_runtime,
    second_program_identity,
)


@dataclass(frozen=True)
class CompiledSecondDerivative:
    """Verified native artifact and immutable shell/coordinate subset identity."""

    native: CudaArtifact
    integral: IntegralIR
    component_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    backend: str
    program_identity: str

    def validate(self):
        """Reject replaced metadata before loading an artifact or allocating buffers."""
        require_second_consumer(self.integral)
        if (
            second_program_identity(
                self.integral, self.component_indices, self.output_indices, self.backend
            )
            != self.program_identity
        ):
            raise ValueError(
                "second derivative mathematical metadata identity mismatch"
            )
        identity = self.native.metadata["identity"]
        if (
            canonical_hash(identity) != self.native.metadata["key"]
            or identity.get("backend", "cuda") != self.backend
        ):
            raise ValueError("second derivative native artifact identity mismatch")
        if (
            not 1 <= len(self.component_indices) <= 64
            or not 1 <= len(self.output_indices) <= 12
        ):
            raise ValueError(
                "second derivative artifact exceeds bounded subset dimensions"
            )

    @property
    def record_format(self):
        """Tagged 208-byte primitive prefix, packed weights and twelve directions."""
        return struct.Struct("<14I19d" + "d" * (len(self.component_indices) + 12))


def compile_second_derivative(
    integral, compiler, cache, *, component_indices=None, output_indices=None
):
    """Compile an explicit coordinate tile without executing or probing a GPU."""
    if isinstance(compiler, CppCompilerAdapter):
        backend = "cpu"
    elif isinstance(compiler, CudaCompilerAdapter):
        backend = "cuda"
    else:
        raise TypeError("select an explicit CPU C++ or CUDA compiler adapter")
    kernel = build_second_derivative_kernel(
        integral, component_indices, output_indices=output_indices
    )
    source = emit_second_derivative_runtime(kernel, backend=backend)
    directory = Path(cache).resolve() / "generated-sources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        canonical_hash({"source": source}) + (".cu" if backend == "cuda" else ".cpp")
    )
    if not path.exists():
        with tempfile.NamedTemporaryFile(
            mode="w", dir=directory, delete=False
        ) as temporary:
            temporary.write(source)
            name = temporary.name
        try:
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    elif path.read_text() != source:
        raise ValueError("second derivative generated source integrity failure")
    names = [
        "src/integrals/range_moments.hpp",
        "src/integrals/eri_geometry.hpp",
        "src/scf/cuda_weighted_eri.hpp",
        "src/scf/weighted_eri_runtime.hpp",
        "include/vibeqc/vibeqc.h",
    ]
    if backend == "cuda":
        names.append("src/tensor/cuda_runtime.cuh")
    headers = tuple(asset_path(name) for name in names)
    root = asset_path("src/scf/weighted_eri_runtime.hpp").parents[2]
    options = ("--fmad=false",) if backend == "cuda" else ("-ffp-contract=off",)
    native = compile_runtime(
        compiler,
        Path(cache) / "native",
        path,
        headers=headers,
        libraries=("cublas",) if backend == "cuda" else (),
        options=(*options, f"-I{root / 'src'}", f"-I{root / 'include'}"),
    )
    return CompiledSecondDerivative(
        native,
        integral,
        kernel.component_indices,
        kernel.output_indices,
        backend,
        second_program_identity(
            integral, kernel.component_indices, kernel.output_indices, backend
        ),
    )


@dataclass(frozen=True)
class SecondPrimitive:
    """One primitive geometry with fixed packed weights and an optional direction.

    Exponents and centers follow the compiled shell/operator order. Weights
    follow its selected Cartesian component indices. ``scale`` contains any
    fixed contraction/normalization factor; operator and weight-descriptor
    factors are already generated in the callable and must not be repeated.
    A raw consumer accepts omitted/unit component weights only. Direction rows
    follow requested mathematical centers, before physical-atom accumulation.
    Caller input ownership is excluded from the prepared numeric-buffer budget.
    """

    exponents: tuple
    centers: tuple
    weights: tuple | None = None
    direction: tuple | None = None
    scale: float = 1.0
    output_tile: int = 0


def _finite(values, size, name):
    """Normalize bounded scalar inputs without silently dropping complex parts."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    values = tuple(float(value) for value in values)
    if len(values) != size or not all(map(math.isfinite, values)):
        raise ValueError(f"{name} must contain {size} finite values")
    return values


def pack_second_primitive(artifact, primitive):
    """Validate and pack one explicitly tagged record for an immutable program."""
    if not isinstance(primitive, SecondPrimitive):
        raise TypeError("expected a SecondPrimitive record")
    integral = artifact.integral
    consumer = require_second_consumer(integral)
    count, shells = len(integral.operator.centers), len(integral.signature.shells)
    exponents = _finite(primitive.exponents, shells, "primitive exponents")
    if any(value <= 0 for value in exponents):
        raise ValueError("primitive exponents must be positive")
    if len(primitive.centers) != count:
        raise ValueError(
            "primitive centers must follow the complete operator inventory"
        )
    centers = tuple(
        value
        for center in primitive.centers
        for value in _finite(center, 3, "primitive center")
    )
    raw = consumer.weights is None
    weights = primitive.weights
    if weights is None and raw:
        weights = (1.0,) * len(artifact.component_indices)
    if weights is None:
        raise ValueError("weighted second derivatives require explicit packed weights")
    weights = _finite(
        weights, len(artifact.component_indices), "packed component weights"
    )
    if raw and any(weight != 1 for weight in weights):
        raise ValueError("raw second derivatives require unit component weights")
    hvp = consumer.output == "weighted_hvp"
    directions = 3 * len(integral.requested_derivative_centers) if hvp else 0
    direction = primitive.direction
    if hvp:
        if direction is None or len(direction) * 3 != directions:
            raise ValueError("HVP direction must follow requested center xyz order")
        direction = tuple(
            value
            for center in direction
            for value in _finite(center, 3, "HVP direction")
        )
    elif direction is not None:
        raise ValueError("a Hessian record does not consume a direction")
    else:
        direction = ()
    (scale,) = _finite((primitive.scale,), 1, "fixed normalization scale")
    checked_bytes(primitive.output_tile, "output tile")
    if primitive.output_tile >= 2**32:
        raise ValueError("output tile exceeds the record ABI")
    return artifact.record_format.pack(
        SECOND_RECORD_TAG,
        primitive.output_tile,
        *((0,) * 12),
        *exponents,
        *((1.0,) * (4 - shells)),
        *centers,
        *((0.0,) * (12 - 3 * count)),
        scale,
        0.0,
        0.0,
        *weights,
        *direction,
        *((0.0,) * (12 - directions)),
    )


@dataclass(frozen=True)
class SecondDerivativeExecution:
    """Detached coordinate tiles and provenance for later response assembly."""

    values: np.ndarray
    diagnostics: dict


class PreparedSecondDerivative:
    """Retain shared native storage for a bounded immutable coordinate tile.

    Primitive iterables are streamed once, including partial/empty chunks.
    Calls serialize; results are published only after every chunk succeeds.
    The numeric budget includes record staging, native results/arena, chunk
    results and the detached output. Compiler metadata, caller inputs, native
    call stacks and CUDA context remain explicit exclusions. A failed call can
    be replayed on the same owner. Destruction requires caller serialization.
    """

    def __init__(
        self,
        artifact,
        *,
        record_capacity=256,
        tile_capacity=1,
        budget=None,
        device_id=0,
    ):
        self._lock, self._handle, self._library = threading.RLock(), ct.c_void_p(), None
        if not isinstance(artifact, CompiledSecondDerivative):
            raise TypeError("expected a compiled second derivative artifact")
        artifact.validate()
        if sys.byteorder != "little":
            raise NotImplementedError(
                "second derivative record ABI requires a little-endian host"
            )
        for value, name in (
            (record_capacity, "record capacity"),
            (tile_capacity, "tile capacity"),
            (device_id, "device ordinal"),
        ):
            checked_bytes(value, name)
        if not record_capacity or not 0 < tile_capacity < 2**32 or device_id >= 2**31:
            raise ValueError(
                "second derivative capacities and device must fit the native ABI"
            )
        if artifact.backend == "cpu" and device_id:
            raise ValueError("CPU execution does not accept a CUDA device ordinal")
        self._artifact, self._record_capacity, self._tile_capacity, self._device_id = (
            artifact,
            record_capacity,
            tile_capacity,
            device_id,
        )
        stride, columns = artifact.record_format.size, len(artifact.output_indices)
        record_bytes, output_bytes = (
            byte_product(record_capacity, stride),
            byte_product(tile_capacity, columns, 8),
        )
        native_device = (
            checked_bytes(record_bytes + output_bytes + 4)
            if artifact.backend == "cuda"
            else 0
        )
        native_budget = checked_bytes(output_bytes + native_device)
        if max(record_bytes, native_budget) >= 2 ** (8 * ct.sizeof(ct.c_size_t)):
            raise ValueError("second derivative capacities exceed native size_t")
        identity = ResourceIdentity(
            "integrals",
            "generated_second_derivative",
            artifact.backend,
            "fp64",
            json.dumps(
                {
                    "program": artifact.program_identity,
                    "artifact": artifact.native.metadata["key"],
                    "records": record_capacity,
                    "tiles": tile_capacity,
                    "device": device_id,
                }
            ),
            (artifact.integral.contractions[0].output,),
            "bounded_coordinate_tile_v1",
        )
        estimates = [
            ResourceEstimate(
                "native_publication", output_bytes, "pageable", 0, 1, "persistent"
            ),
            ResourceEstimate(
                "record_staging", record_bytes, "pageable", 0, 1, "persistent"
            ),
            ResourceEstimate(
                "chunk_result", output_bytes, "pageable", 0, 1, "persistent"
            ),
            ResourceEstimate(
                "detached_result", output_bytes, "pageable", 1, 1, "output"
            ),
            ResourceEstimate("packed_record", stride, "pageable", 1, 1),
        ]
        if native_device:
            estimates.append(
                ResourceEstimate(
                    "native_arena",
                    native_device,
                    f"device:{device_id}",
                    0,
                    1,
                    "persistent",
                )
            )
        request = ResourceRequest(
            "second_derivative",
            identity,
            (ResourceCandidate("generated_unscreened", "streamed", tuple(estimates)),),
            (
                "caller-owned primitive records",
                "Python/compiler metadata",
                "native call stacks and CUDA context",
            ),
        )
        self._resource_plan = plan_resources(
            (request,),
            budget or ResourceBudget(host_bytes=64 << 20, device_bytes=64 << 20),
        )
        if self.resource_plan.status != "feasible":
            raise ValueError(
                self.resource_plan.diagnostic
                or "second derivative resource budget exceeded"
            )
        if (
            file_hash(artifact.native.library)
            != artifact.native.metadata["binary_sha256"]
        ):
            raise ValueError("second derivative binary hash mismatch")
        lib = self._library = ct.CDLL(str(artifact.native.library))
        lib.vibeqc_second_identity_v1.restype = ct.c_char_p
        lib.vibeqc_second_stride_v1.restype = ct.c_size_t
        if (
            lib.vibeqc_second_identity_v1().decode() != artifact.program_identity
            or lib.vibeqc_second_stride_v1() != stride
        ):
            raise ValueError("second derivative native identity or stride mismatch")
        lib.vibeqc_second_create_v1.argtypes = (
            [ct.c_int] * 3
            + [ct.c_size_t] * 3
            + [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_size_t]
        )
        lib.vibeqc_second_run_v1.argtypes = (
            [ct.c_void_p, ct.c_void_p]
            + [ct.c_size_t] * 3
            + [ct.c_void_p, ct.c_int, ct.c_char_p, ct.c_size_t]
        )
        lib.vibeqc_second_storage_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_uint64),
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.vibeqc_second_destroy_v1.argtypes, lib.vibeqc_second_destroy_v1.restype = (
            [ct.c_void_p],
            None,
        )
        if artifact.backend == "cuda":
            lib.vibeqc_second_metrics_v1.argtypes = [
                ct.c_void_p,
                ct.POINTER(_Metrics),
                ct.c_char_p,
                ct.c_size_t,
            ]
        self._records = np.empty((record_capacity, stride), dtype=np.uint8)
        self._chunk = np.empty((tile_capacity, columns))
        target = artifact.native.metadata["identity"]["target"]
        major, minor = (
            (target["compute_capability_major"], target["compute_capability_minor"])
            if artifact.backend == "cuda"
            else (0, 0)
        )
        try:
            with _PREPARATION_LOCK:
                self._call(
                    "vibeqc_second_create_v1",
                    device_id,
                    major,
                    minor,
                    record_capacity,
                    tile_capacity,
                    native_budget,
                    ct.byref(self._handle),
                )
            amounts = (ct.c_uint64 * 2)()
            self._call("vibeqc_second_storage_v1", self._handle, amounts)
            if tuple(amounts) != (output_bytes, native_device):
                raise RuntimeError(
                    "second derivative native storage differs from its resource plan"
                )
        except BaseException:
            self.close()
            raise

    @property
    def artifact(self):
        """Immutable program metadata bound to the retained native handle."""
        return self._artifact

    @property
    def record_capacity(self):
        return self._record_capacity

    @property
    def tile_capacity(self):
        return self._tile_capacity

    @property
    def device_id(self):
        return self._device_id

    @property
    def resource_plan(self):
        """Shared resource estimate for the immutable prepared capacities."""
        return self._resource_plan

    def _call(self, name, *args):
        error = ct.create_string_buffer(1024)
        status = getattr(self._library, name)(*args, error, len(error))
        if status:
            exception = {
                1: ValueError,
                2: ValueError,
                5: FloatingPointError,
                7: MemoryError,
            }.get(status, RuntimeError)
            raise exception(
                error.value.decode() or f"second derivative native status {status}"
            )

    def contract(self, primitives, *, tile_count=1, profile=False):
        """Accumulate streamed fixed-weight primitives into detached output tiles."""
        from .second_derivatives_inputs import SecondShellStream

        with self._lock:
            if not self._handle.value:
                raise RuntimeError("second derivative plan is closed")
            checked_bytes(tile_count, "output tile count")
            if tile_count > self.tile_capacity:
                raise ValueError(
                    "second derivative output exceeds prepared tile capacity"
                )
            if type(profile) is not bool:
                raise TypeError("profile must be boolean")
            resource_plan = self.resource_plan
            if isinstance(primitives, SecondShellStream):
                if primitives.program_identity != self.artifact.program_identity:
                    raise ValueError(
                        "second public stream belongs to another compiled program"
                    )
                resource_plan = plan_resources(
                    (*resource_plan.requests, primitives.resource_request),
                    resource_plan.budget,
                )
                if resource_plan.status != "feasible":
                    raise ValueError(
                        resource_plan.diagnostic
                        or "combined second derivative resource budget exceeded"
                    )
            result = np.zeros((tile_count, len(self.artifact.output_indices)))
            count = records = chunks = 0
            digest, started = hashlib.sha256(), time.perf_counter()
            timing = {
                name: 0.0
                for name in ("device_ms", "input_ms", "output_ms", "kernel_ms")
            }

            def flush(count):
                nonlocal chunks
                self._call(
                    "vibeqc_second_run_v1",
                    self._handle,
                    self._records.ctypes.data,
                    count,
                    self.artifact.record_format.size,
                    tile_count,
                    self._chunk.ctypes.data,
                    int(profile),
                )
                with np.errstate(over="raise", invalid="raise"):
                    np.add(result, self._chunk[:tile_count], out=result)
                chunks += 1
                if profile and self.artifact.backend == "cuda":
                    metrics = _Metrics()
                    self._call(
                        "vibeqc_second_metrics_v1", self._handle, ct.byref(metrics)
                    )
                    for name in timing:
                        timing[name] += getattr(metrics, name)

            for primitive in primitives:
                blob = pack_second_primitive(self.artifact, primitive)
                if primitive.output_tile >= tile_count:
                    raise ValueError(
                        "second derivative primitive output tile exceeds requested rows"
                    )
                digest.update(blob)
                self._records[count] = np.frombuffer(blob, dtype=np.uint8)
                count, records = count + 1, records + 1
                if count == self.record_capacity:
                    flush(count)
                    count = 0
            if count:
                flush(count)
            result.setflags(write=False)
            diagnostics = {
                "program_identity": self.artifact.program_identity,
                "native_artifact": self.artifact.native.metadata["key"],
                "backend": self.artifact.backend,
                "integral": integral_to_payload(self.artifact.integral),
                "component_indices": list(self.artifact.component_indices),
                "output_indices": list(self.artifact.output_indices),
                "schedule": "bounded_coordinate_tile_v1",
                "experimental": True,
                "status": {
                    "source": "lowered",
                    "compiler": "passed",
                    "execution": "passed",
                    "independent_numerical_gate": "not_run_by_this_call",
                    "method_endpoint": "unavailable",
                },
                "screening": "disabled",
                "electronic_response": "excluded",
                "records": records,
                "chunks": chunks,
                "input_sha256": digest.hexdigest(),
                "wall_seconds": time.perf_counter() - started,
                "profile": profile,
                "device_timing": timing
                if profile and self.artifact.backend == "cuda"
                else None,
                "resources": resource_plan.to_dict(),
                "compiler_resources": self.artifact.native.metadata["resources"],
            }
            return SecondDerivativeExecution(result, diagnostics)

    def close(self):
        """Release the shared native arena once, respecting its preparation lock."""
        with self._lock, _PREPARATION_LOCK:
            if self._handle.value:
                self._library.vibeqc_second_destroy_v1(self._handle)
                self._handle = ct.c_void_p()
            self._records = self._chunk = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        with suppress(Exception):
            self.close()

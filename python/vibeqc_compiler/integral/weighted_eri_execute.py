"""Explicit compilation and bounded prepared execution of range-weighted ERIs.

The existing stream owns public cotangent pullbacks and normalization. This
adapter owns its compact upload/result buffers and reuses the shared native
compiler cache, resource planner and CUDA context. It never selects another
radial operator, enables screening or promotes an experimental schedule.
"""

from __future__ import annotations

import ctypes as ct
import json
import math
import os
import sys
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
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

from .blocks import (
    BlockRequest,
    ShellTile,
    TensorLayout,
    WeightTile,
)
from .ir import IntegralIR
from .ir_serialization import integral_to_payload
from .shell_signature import BasisConvention, CenterBinding
from .shell_spec import ShellClassSpec
from .weighted_eri import build_weighted_eri_kernel, canonical_range_weighted_eri_ir
from .weighted_eri_inputs import (
    PRIMITIVE_RANGE_RECORD,
    WeightedEriPrimitiveStream,
    prepare_weighted_eri_stream,
)
from .weighted_eri_native import (
    emit_weighted_eri_runtime,
    weighted_eri_metadata_identity,
    weighted_eri_program_identity,
)


@dataclass(frozen=True)
class CompiledWeightedEri:
    """Verified native artifact plus immutable mathematical/subset metadata."""

    native: CudaArtifact
    requested: IntegralIR
    integral: IntegralIR
    component_indices: tuple[int, ...]
    component_quantums: tuple[tuple[int, ...], ...]
    backend: str
    program_identity: str

    def validate(self):
        """Check the mathematical/subset binding before loading a native library.

        Frozen dataclasses can still be replaced, and native artifact metadata
        is shared with the cache API. Verify both against their content hashes
        on preparation instead of trusting a caller-supplied identity string.
        """
        canonical = canonical_range_weighted_eri_ir(self.requested)
        if (
            self.integral != canonical
            or weighted_eri_metadata_identity(
                canonical, self.component_indices, self.backend
            )
            != self.program_identity
        ):
            raise ValueError("weighted ERI mathematical metadata identity mismatch")
        spec = ShellClassSpec(
            "".join(
                "spdf"[angular_momentum]
                for angular_momentum in canonical.signature.angular
            ),
            canonical.signature.angular,
        )
        indices = self.component_indices
        if (
            not 1 <= len(indices) <= 64
            or len(set(indices)) != len(indices)
            or any(
                type(i) is not int or not 0 <= i < spec.component_count for i in indices
            )
        ):
            raise ValueError("weighted ERI compiled component subset is invalid")
        quantums = tuple(
            tuple(c.count(axis) for c in spec.components[i] for axis in "xyz")
            for i in indices
        )
        if quantums != self.component_quantums:
            raise ValueError("weighted ERI component quantum metadata mismatch")
        identity = self.native.metadata["identity"]
        if (
            canonical_hash(identity) != self.native.metadata["key"]
            or identity.get("backend", "cuda") != self.backend
        ):
            raise ValueError("weighted ERI native artifact identity mismatch")


def compile_weighted_eri(
    integral: IntegralIR, compiler, cache: Path, *, component_indices=None
) -> CompiledWeightedEri:
    """Compile one explicit range/subset using the common local artifact cache.

    Compilation needs no GPU and imports neither SciPy nor a reference engine.
    The source takes already-normalized record weights, so unit consumer
    factors are generated even when the public request has signed coefficients.
    Those coefficients remain owned by prepare_weighted_eri_stream.
    """
    if isinstance(compiler, CppCompilerAdapter):
        backend = "cpu"
    elif isinstance(compiler, CudaCompilerAdapter):
        backend = "cuda"
    else:
        raise TypeError("select an explicit CPU C++ or CUDA compiler adapter")
    canonical = canonical_range_weighted_eri_ir(integral)
    kernel = build_weighted_eri_kernel(canonical, component_indices)
    source = emit_weighted_eri_runtime(kernel, backend=backend)
    directory = Path(cache).resolve() / "generated-sources"
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".cu" if backend == "cuda" else ".cpp"
    path = directory / (canonical_hash({"source": source}) + suffix)
    if not path.exists():
        # Concurrent writers of the same deterministic source publish complete
        # bytes atomically; only the shared cache below owns compiled artifacts.
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
        raise ValueError("generated weighted source cache integrity failure")
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
    quantums = tuple(
        tuple(
            component.count(axis)
            for component in kernel.spec.components[i]
            for axis in "xyz"
        )
        for i in kernel.component_indices
    )
    return CompiledWeightedEri(
        native,
        integral,
        canonical,
        kernel.component_indices,
        quantums,
        backend,
        weighted_eri_program_identity(kernel, backend),
    )


@dataclass(frozen=True)
class WeightedEriExecution:
    """Detached tile rows: value followed by four ordered xyz center derivatives."""

    values: np.ndarray
    diagnostics: dict


class PreparedWeightedEri:
    """Retain bounded native/host buffers for one immutable range and subset.

    Calls on one instance are serialized. Stream geometry and normalized
    weights may change; every stream must match the compiled radial identity
    and angular coverage. Results are published only after all chunks succeed.
    Caller-owned stream storage, compiler graphs/cache files, Python object
    overhead and native call stacks/CUDA context are outside this owner's
    numeric resource request. Stream preparation has its own explicit budget.
    """

    def __init__(
        self,
        artifact: CompiledWeightedEri,
        *,
        record_capacity=256,
        tile_capacity=1,
        budget=None,
        device_id=0,
    ):
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self._library = None
        if not isinstance(artifact, CompiledWeightedEri):
            raise TypeError("expected a compiled weighted ERI artifact")
        artifact.validate()
        if sys.byteorder != "little":
            raise NotImplementedError(
                "weighted native record ABI requires a little-endian host"
            )
        for value, name in (
            (record_capacity, "record capacity"),
            (tile_capacity, "tile capacity"),
            (device_id, "device ordinal"),
        ):
            checked_bytes(value, name)
        if not record_capacity or not 0 < tile_capacity < 2**32:
            raise ValueError(
                "positive representable record/tile capacities are required"
            )
        if device_id >= 2**31:
            raise ValueError("visible device ordinal exceeds the native int32 ABI")
        if artifact.backend == "cpu" and device_id != 0:
            raise ValueError("CPU execution does not accept a CUDA device ordinal")
        self._artifact, self._device_id = artifact, device_id
        self._record_capacity, self._tile_capacity = record_capacity, tile_capacity
        record_bytes = byte_product(record_capacity, PRIMITIVE_RANGE_RECORD.size)
        output_bytes = byte_product(tile_capacity, 13, 8)
        if max(record_bytes, output_bytes) >= 2 ** (8 * ct.sizeof(ct.c_size_t)):
            raise ValueError("weighted ERI capacities exceed native size_t")
        native_device = (
            checked_bytes(record_bytes + output_bytes + 4)
            if artifact.backend == "cuda"
            else 0
        )
        if output_bytes + native_device >= 2 ** (8 * ct.sizeof(ct.c_size_t)):
            raise ValueError("weighted ERI combined native budget exceeds size_t")
        identity = ResourceIdentity(
            "integrals",
            "generated_weighted_eri",
            artifact.backend,
            "fp64",
            json.dumps(
                {
                    "program": artifact.program_identity,
                    "native_artifact": artifact.native.metadata["key"],
                    "record_capacity": record_capacity,
                    "tile_capacity": tile_capacity,
                    "device": device_id,
                }
            ),
            ("value", "first_nuclear_derivative"),
            "bounded_primitive_stream_v2",
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
            ResourceEstimate(
                "yielded_record", PRIMITIVE_RANGE_RECORD.size, "pageable", 1, 1
            ),
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
            "weighted_eri",
            identity,
            (ResourceCandidate("generated_unscreened", "streamed", tuple(estimates)),),
            (
                "caller-owned primitive streams",
                "Python/compiler object metadata",
                "native call stacks and CUDA context",
            ),
        )
        self._resource_plan = plan_resources(
            (request,),
            budget or ResourceBudget(host_bytes=64 << 20, device_bytes=64 << 20),
        )
        if self.resource_plan.status != "feasible":
            raise ValueError(
                self.resource_plan.diagnostic or "weighted ERI resource budget exceeded"
            )
        if (
            file_hash(artifact.native.library)
            != artifact.native.metadata["binary_sha256"]
        ):
            raise ValueError("weighted ERI binary hash mismatch")
        lib = self._library = ct.CDLL(str(artifact.native.library))
        lib.vibeqc_weighted_identity_v2.restype = ct.c_char_p
        if lib.vibeqc_weighted_identity_v2().decode() != artifact.program_identity:
            raise ValueError("weighted ERI compiled program identity mismatch")
        lib.vibeqc_weighted_create_v2.argtypes = (
            [ct.c_int] * 3
            + [ct.c_size_t] * 3
            + [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_size_t]
        )
        lib.vibeqc_weighted_run_v2.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_void_p,
            ct.c_int,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.vibeqc_weighted_storage_v2.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_uint64),
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.vibeqc_weighted_destroy_v2.argtypes = [ct.c_void_p]
        lib.vibeqc_weighted_destroy_v2.restype = None
        if artifact.backend == "cuda":
            lib.vibeqc_weighted_metrics_v2.argtypes = [
                ct.c_void_p,
                ct.POINTER(_Metrics),
                ct.c_char_p,
                ct.c_size_t,
            ]
        self._records = np.empty(
            (record_capacity, PRIMITIVE_RANGE_RECORD.size), dtype=np.uint8
        )
        self._chunk = np.empty((tile_capacity, 13), dtype=np.float64)
        target = artifact.native.metadata["identity"]["target"]
        major = target["compute_capability_major"] if artifact.backend == "cuda" else 0
        minor = target["compute_capability_minor"] if artifact.backend == "cuda" else 0
        try:
            with _PREPARATION_LOCK:
                self._call(
                    "vibeqc_weighted_create_v2",
                    device_id,
                    major,
                    minor,
                    record_capacity,
                    tile_capacity,
                    checked_bytes(output_bytes + native_device),
                    ct.byref(self._handle),
                )
            amounts = (ct.c_uint64 * 2)()
            self._call("vibeqc_weighted_storage_v2", self._handle, amounts)
            if tuple(amounts) != (output_bytes, native_device):
                raise RuntimeError(
                    "weighted ERI native storage differs from the resource plan"
                )
        except BaseException:
            self.close()
            raise

    @property
    def artifact(self):
        """Immutable mathematical/subset identity bound to the native handle."""
        return self._artifact

    @property
    def device_id(self):
        return self._device_id

    @property
    def record_capacity(self):
        return self._record_capacity

    @property
    def tile_capacity(self):
        return self._tile_capacity

    @property
    def resource_plan(self):
        """The shared planner's immutable estimate for this owner's buffers."""
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
                error.value.decode() or f"weighted ERI native status {status}"
            )

    def contract(
        self, streams, *, tile_count=None, profile=False
    ) -> WeightedEriExecution:
        """Contract normalized streams without retaining their primitive products."""
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("weighted ERI plan is closed")
            if type(profile) is not bool:
                raise TypeError("profile must be boolean")
            streams = (
                (streams,)
                if isinstance(streams, WeightedEriPrimitiveStream)
                else tuple(streams)
            )
            allowed = set(self.artifact.component_quantums)
            for stream in streams:
                if not isinstance(stream, WeightedEriPrimitiveStream):
                    raise TypeError("expected a prepared weighted primitive stream")
                integral = stream.request.integral
                if (
                    integral.operator != self.artifact.integral.operator
                    or integral.signature.angular
                    != self.artifact.integral.signature.angular
                    or stream.record_size != PRIMITIVE_RANGE_RECORD.size
                ):
                    raise ValueError(
                        "weighted ERI stream operator, omega or angular identity mismatch"
                    )
                if any(angular not in allowed for angular, _ in stream.components):
                    raise ValueError(
                        "weighted ERI stream exceeds its compiled component subset"
                    )
            minimum = max((stream.output_tile for stream in streams), default=-1) + 1
            tile_count = minimum if tile_count is None else tile_count
            checked_bytes(tile_count, "output tile count")
            if not minimum <= tile_count <= self.tile_capacity:
                raise ValueError(
                    "weighted ERI output tiles exceed the prepared capacity"
                )
            result = np.zeros((tile_count, 13))
            count = chunks = records = 0
            timing = {
                name: 0.0
                for name in ("device_ms", "input_ms", "output_ms", "kernel_ms")
            }
            started = time.perf_counter()

            def flush(count):
                nonlocal chunks
                self._call(
                    "vibeqc_weighted_run_v2",
                    self._handle,
                    self._records.ctypes.data,
                    count,
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
                        "vibeqc_weighted_metrics_v2", self._handle, ct.byref(metrics)
                    )
                    for name in timing:
                        timing[name] += getattr(metrics, name)

            for stream in streams:
                for blob in stream.records():
                    if len(blob) != PRIMITIVE_RANGE_RECORD.size:
                        raise ValueError("weighted ERI record stride mismatch")
                    self._records[count] = np.frombuffer(blob, dtype=np.uint8)
                    count += 1
                    records += 1
                    if count == self.record_capacity:
                        flush(count)
                        count = 0
            if count:
                flush(count)
            result.setflags(write=False)
            stream_identities = [
                canonical_hash(
                    {
                        "integral": integral_to_payload(stream.request.integral),
                        "center_bindings": [
                            asdict(binding)
                            for binding in stream.request.center_bindings
                        ],
                        "primitives": stream.primitives,
                        "centers": stream.centers,
                        "components": stream.components,
                        "fused_weights": stream.fused_weights,
                        "output_tile": stream.output_tile,
                    }
                )
                for stream in streams
            ]
            diagnostics = {
                "program_identity": self.artifact.program_identity,
                "native_artifact": self.artifact.native.metadata["key"],
                "operator": self.artifact.integral.operator.coulomb_kernel.to_payload(),
                "backend": self.artifact.backend,
                "schedule": "bounded_primitive_stream_v2",
                "experimental": True,
                "screening": "disabled",
                "profile": profile,
                "records": records,
                "chunks": chunks,
                "wall_seconds": time.perf_counter() - started,
                "stream_identities": stream_identities,
                "device_timing": timing
                if profile and self.artifact.backend == "cuda"
                else None,
                "resources": self.resource_plan.to_dict(),
            }
            return WeightedEriExecution(result, diagnostics)

    def close(self):
        """Release native resources once, with the shared allocation snapshot lock."""
        with self._lock, _PREPARATION_LOCK:
            if self._handle.value:
                self._library.vibeqc_weighted_destroy_v2(self._handle)
                self._handle = ct.c_void_p()
            self._records = self._chunk = None

    def raw(
        self,
        primitives,
        centers,
        component_indices,
        *,
        projections=None,
        adapter_budget_bytes=4 << 20,
        profile=False,
    ) -> WeightedEriExecution:
        """Evaluate a bounded selection of raw contracted public shell components.

        Indices are flattened in the requested signature's public AO order;
        each returned row contains one value and twelve shell-center derivatives.
        Primitives use the stream adapter's radial-normalized coefficients.
        Spherical signatures require the existing Cartesian pullback matrices.
        Unit cotangents run through the same weighted provider, with no dense
        molecular four-index response and no truncation of an unsupported
        Cartesian pullback. The raw adapter has its own bounded workspace.
        """
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("weighted ERI plan is closed")
            indices = tuple(component_indices)
            signature = self.artifact.requested.signature
            count = math.prod(signature.component_shape)
            for index in indices:
                checked_bytes(index, "raw public component")
                if index >= count:
                    raise ValueError("raw public component is outside the shell")
            if len(indices) > self.tile_capacity:
                raise ValueError(
                    "raw component count exceeds the prepared tile capacity"
                )
            if all(
                s.convention == BasisConvention.CARTESIAN for s in signature.shells
            ) and not set(indices) <= set(self.artifact.component_indices):
                raise ValueError("raw request exceeds the compiled Cartesian subset")
            if not indices:
                empty = self.contract((), tile_count=0, profile=profile)
                return WeightedEriExecution(
                    empty.values, {**empty.diagnostics, "raw_component_indices": []}
                )
            checked_bytes(adapter_budget_bytes, "raw adapter budget")
            topology = json.dumps(
                {
                    "program": self.artifact.program_identity,
                    "public_components": indices,
                    "adapter_budget_bytes": adapter_budget_bytes,
                }
            )
            raw_request = ResourceRequest(
                "raw_adapter",
                ResourceIdentity(
                    "integrals",
                    "raw_unit_weights",
                    self.artifact.backend,
                    "fp64",
                    topology,
                    ("value", "first_nuclear_derivative"),
                    "bounded_public_components",
                ),
                (
                    ResourceCandidate(
                        "shared_weighted_provider",
                        "streamed",
                        (
                            ResourceEstimate(
                                "raw_stream", adapter_budget_bytes, "pageable", 1, 1
                            ),
                            ResourceEstimate("nested_result", 13 * 8, "pageable", 1, 1),
                        ),
                    ),
                ),
            )
            raw_plan = plan_resources(
                (*self.resource_plan.requests, raw_request), self.resource_plan.budget
            )
            if raw_plan.status != "feasible":
                raise ValueError(
                    raw_plan.diagnostic or "raw ERI adapter resource budget exceeded"
                )
            consumer = self.artifact.integral.contractions[0]
            layout = TensorLayout(signature.tensor_indices, (1, 1, 1, 1))
            consumer = replace(
                consumer,
                weights=replace(consumer.weights, layout=layout),
                memory_budget_bytes=adapter_budget_bytes,
            )
            integral = replace(
                self.artifact.integral, spec=signature, contractions=(consumer,)
            )
            output = np.zeros((len(indices), 13))
            row_diagnostics = []
            started = time.perf_counter()
            for row, index in enumerate(indices):
                offsets = tuple(
                    int(i) for i in np.unravel_index(index, signature.component_shape)
                )
                request = BlockRequest(
                    f"raw/{index}",
                    integral,
                    ShellTile(offsets, (1, 1, 1, 1)),
                    center_bindings=tuple(CenterBinding(i, i) for i in range(4)),
                )
                stream = prepare_weighted_eri_stream(
                    request,
                    primitives,
                    centers,
                    lambda descriptor, _: WeightTile(descriptor.layout, (1.0,)),
                    projections=projections,
                )
                result = self.contract(stream, profile=profile)
                output[row] = result.values[0]
                row_diagnostics.append(result.diagnostics)
                # Only one raw unit stream and one nested result are live.
                del stream, result
            diagnostics = dict(row_diagnostics[-1])
            diagnostics.update(
                raw_component_indices=list(indices),
                public_component_shape=list(signature.component_shape),
                public_conventions=[s.convention.value for s in signature.shells],
                records=sum(d["records"] for d in row_diagnostics),
                chunks=sum(d["chunks"] for d in row_diagnostics),
                stream_identities=[
                    key for d in row_diagnostics for key in d["stream_identities"]
                ],
                resources=raw_plan.to_dict(),
                wall_seconds=time.perf_counter() - started,
            )
            if profile and self.artifact.backend == "cuda":
                diagnostics["device_timing"] = {
                    name: sum(d["device_timing"][name] for d in row_diagnostics)
                    for name in diagnostics["device_timing"]
                }
            output.setflags(write=False)
            return WeightedEriExecution(output, diagnostics)

    def __enter__(self):
        if not self._handle.value:
            raise RuntimeError("weighted ERI plan is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        # Explicit close/context management is the reporting boundary. Garbage
        # collection may run after a partial constructor or interpreter teardown.
        with suppress(Exception):
            self.close()

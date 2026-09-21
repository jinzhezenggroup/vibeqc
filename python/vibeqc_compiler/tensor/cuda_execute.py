"""Compile, prepare and execute typed TensorIR programs on CUDA.

Compilation is explicit and may run without a GPU. Preparation, probing and
execution require the caller's allocated GPU (on this workstation, Slurm).
There is no installation-time tuning, implicit CPU arithmetic fallback or
per-contraction Python loop.
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
import typing
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.capture import (
    GRAPH_MODES,
    MAX_CAPTURE_LAUNCHES,
    CaptureContract,
    _GraphMetrics,
)
from vibeqc_compiler.common.cuda_runtime import (
    _PREPARATION_LOCK,
    CudaArtifact,
    _Metrics,
)
from vibeqc_compiler.common.paths import LAYOUT_VERSION, asset_path, source_hashes
from vibeqc_compiler.common.provenance import (
    atomic_json,
    canonical_hash,
    file_hash,
    toolchain_identity,
)
from vibeqc_compiler.common.specialization import (
    CompilationIdentity,
    GuardPredicate,
    SpecializationGuard,
    TargetCapabilities,
    WorkloadSignature,
)

from .cuda_dtype import compile_options, scalar_type, symmetry_tolerance
from .cuda_emit import emit_cuda
from .cuda_gemm import gemm_contract
from .cuda_plan import (
    VALIDATION_CHUNK,
    TensorPlan,
    _index_table_values,
    static_data_slices,
)
from .cuda_resources import parse_resources

if typing.TYPE_CHECKING:
    from typing_extensions import Self

    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.resources import ResourcePlan

    from .ir import Node

# Allocation snapshots for provider accounting must not race another owned
# handle's creation/destruction. Executions themselves remain independent.


@dataclass(frozen=True)
class CudaExecution:
    """Detached named outputs and complete execution/profiling measurements."""

    outputs: dict[str, np.ndarray]
    metrics: dict[str, typing.Any]
    backend: str = "cuda-fp64-ordinary-stream"


def tensor_source_identity() -> str:
    """Inventory tensor sources separately from the shell-class source catalog.

    Reuse #136's canonical/file hashes and atomic publication, but never use a
    shell profile identity for tensor code it does not inventory.
    """
    return canonical_hash(
        {
            "layout_version": LAYOUT_VERSION,
            "sources": source_hashes(
                "tensor",
                "integral",
                "common",
                assets=(
                    "src/tensor/cuda_runtime.cuh",
                    "src/runtime/bounded_workspace.hpp",
                    "src/runtime/cuda_resources.cuh",
                    "src/runtime/resource_cuda.cuh",
                    "src/runtime/resource_ledger.hpp",
                    "src/tensor/cuda_graph_context.cuh",
                    "src/runtime/cuda_graph_region.cuh",
                    "src/tensor/cuda_error.hpp",
                    "src/tensor/metrics.hpp",
                    "src/runtime/allocation_measurement.hpp",
                ),
            ),
        }
    )


def tensor_static_data(plan: TensorPlan) -> bytes:
    """Serialize immutable constants/maps without device-alignment padding."""
    parts = []
    payload_bytes = 0
    for step_index, kind, _arena, payload_offset, size_bytes in static_data_slices(
        plan
    ):
        node = plan.steps[step_index].node
        if kind == "constant":
            scalar = scalar_type(node.spec.dtype)
            values = [scalar.coefficient(pair) for pair in node.attrs["values"]]
            payload = np.asarray(values, dtype=node.spec.dtype).tobytes(order="C")
        else:
            values = _index_table_values(node)
            assert values is not None
            payload = np.asarray(values, dtype=np.int64).tobytes(order="C")
        if len(payload) != size_bytes:
            raise AssertionError("tensor static-data byte accounting mismatch")
        if payload_offset != payload_bytes:
            raise AssertionError("tensor static-data payload offsets are not compact")
        parts.append(payload)
        payload_bytes += len(payload)
    result = b"".join(parts)
    if len(result) != plan.static_data_bytes:
        raise AssertionError("tensor static-data plan size mismatch")
    return result


def _read_static_data(artifact: CudaArtifact, expected_bytes: int) -> np.ndarray:
    """Read bounded bytes and bind the uploaded buffer to the compiled identity."""
    descriptor = artifact.metadata.get("identity", {}).get("static_data", {})
    size, digest = descriptor.get("bytes"), descriptor.get("sha256")
    if (
        type(size) is not int
        or size != expected_bytes
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or artifact.metadata.get("static_data_bytes") != size
        or artifact.metadata.get("static_data_sha256") != digest
    ):
        raise ValueError("tensor static-data descriptor differs from compiled identity")
    path = artifact.library.parent / "static.bin"
    try:
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != size:
                raise ValueError("tensor static-data size mismatch")
            payload = np.fromfile(stream, dtype=np.uint8, count=size)
            if os.fstat(stream.fileno()).st_size != size:
                raise ValueError("tensor static-data size changed during read")
    except OSError as error:
        raise ValueError("tensor static-data file is unavailable") from error
    if (
        payload.nbytes != size
        or hashlib.sha256(memoryview(payload)).hexdigest() != digest
    ):
        raise ValueError("tensor static-data bytes differ from compiled identity")
    return payload


def compile_cuda(
    plan: TensorPlan, compiler: CudaCompilerAdapter, cache: Path
) -> CudaArtifact:
    """Compile/cache one whole plan with a finite NVCC process-tree timeout.

    Concurrent compilers publish only complete directories. Every cache hit
    verifies the native binary hash; mismatches fail closed instead of loading
    an unverified executable. This is a local trusted-code cache, not a format
    for downloading arbitrary native libraries from another user.
    """
    if compiler.target != plan.target:
        raise ValueError("compiler target does not match tensor plan target")
    options = compile_options(plan)
    source = emit_cuda(plan, embed_static_data=False)
    static_data = tensor_static_data(plan)
    static_sha256 = hashlib.sha256(static_data).hexdigest()
    host_compiler = os.environ.get("NVCC_CCBIN") or shutil.which("gcc")
    if not host_compiler:
        raise ValueError("NVCC host compiler cannot be identified")
    host_version = subprocess.run(
        [host_compiler, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    identity = {
        "schema": LAYOUT_VERSION,
        "plan": plan.identity,
        "source": tensor_source_identity(),
        "generated": canonical_hash(source),
        "static_data": {"bytes": len(static_data), "sha256": static_sha256},
        "toolchain": toolchain_identity(compiler.nvcc),
        "host_compiler": host_version.stdout + host_version.stderr,
        "compiler_environment": {
            name: os.environ.get(name, "")
            for name in (
                "NVCC_PREPEND_FLAGS",
                "NVCC_APPEND_FLAGS",
                "NVCC_CCBIN",
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "LIBRARY_PATH",
            )
        },
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": platform.libc_ver(),
        },
        "options": [*options, "c++17", "O3", "shared", "fPIC", "cublas"],
    }
    key = canonical_hash(identity)
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    if not destination.exists():
        with tempfile.TemporaryDirectory(
            prefix=".tensor-build-", dir=cache
        ) as temporary:
            directory = Path(temporary)
            cu, library = directory / "program.cu", directory / "program.so"
            static_path = directory / "static.bin"
            cu.write_text(source)
            static_path.write_bytes(static_data)
            result = compiler.compile_shared(
                cu,
                library,
                includes=(asset_path("src/tensor"),),
                libraries=("cublas",),
                options=options,
            )
            (directory / "compiler.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"TensorIR NVCC compilation failed ({result.returncode}):\n{result.stdout}{result.stderr}"
                )
            metadata = {
                "identity": identity,
                "key": key,
                "binary_sha256": file_hash(library),
                "static_data_sha256": file_hash(static_path),
                "static_data_bytes": static_path.stat().st_size,
                "compile_seconds": result.duration_seconds,
                "generated_source_bytes": len(source.encode("utf-8")),
                "binary_bytes": library.stat().st_size,
                "artifact_bytes": library.stat().st_size + static_path.stat().st_size,
                "resources": [
                    dataclasses.asdict(r) for r in parse_resources(result.stderr)
                ],
            }
            atomic_json(directory / "artifact.json", metadata)
            atomic_json(directory / "plan.json", plan.to_payload())
            # rename is atomic on the cache filesystem. Another complete
            # winner is safe; an incomplete destination is rejected below.
            try:
                os.rename(directory, destination)
            except OSError:
                if not destination.is_dir():
                    raise
    metadata = json.loads((destination / "artifact.json").read_text())
    library = destination / "program.so"
    static_path = destination / "static.bin"
    if (
        metadata.get("identity") != json.loads(json.dumps(identity))
        or metadata.get("key") != key
        or file_hash(library) != metadata.get("binary_sha256")
        or metadata.get("static_data_bytes") != len(static_data)
        or metadata.get("static_data_sha256") != static_sha256
        or not static_path.is_file()
        or static_path.stat().st_size != metadata.get("static_data_bytes")
        or file_hash(static_path) != metadata.get("static_data_sha256")
    ):
        raise ValueError(
            "tensor artifact identity, binary, or static-data hash mismatch"
        )
    return CudaArtifact(library, metadata)


def tensor_capture_contract(
    plan: TensorPlan,
    artifact: CudaArtifact,
    device: Mapping[str, typing.Any],
    *,
    resource_plan: ResourcePlan | None = None,
) -> CaptureContract:
    """Qualify only the fixed-topology, device-only emitted launch sequence."""
    launches = 1  # per-run arithmetic-error reset
    for step in plan.steps:
        if (
            step.virtual
            or step.node.op in ("input", "constant")
            or not step.node.spec.size
        ):
            continue
        if step.gemm == "none":
            launches += 1
            continue
        g = gemm_contract(step.node)
        if g is None:
            raise ValueError("GEMM capture step requires a contraction node")
        if not g.k:
            launches += 1
        elif step.gemm.startswith("direct-"):
            launches += 2
        else:
            from math import prod

            tiles = [
                (n + t - 1) // t
                for n, t in zip(
                    (g.m, g.n, g.k),
                    (plan.schedule.tile_m, plan.schedule.tile_n, plan.schedule.tile_k),
                )
            ]
            launches += g.batch * prod(tiles[:2]) * (2 * tiles[2] + 1)
    effects = (
        ()
        if resource_plan is None
        else ("graph retained storage is not yet covered by the global resource plan",)
    )
    workload = WorkloadSignature(
        "tensorir",
        (
            ("plan", plan.identity),
            ("precision", plan.precision),
            ("layout", "fixed device arena; host staging outside capture"),
            ("launches", launches),
        ),
    )
    return CaptureContract(
        compilation=CompilationIdentity(
            plan.program.logical_hash, artifact.metadata["key"]
        ),
        workload=workload,
        target=TargetCapabilities(plan.target.target_info),
        guard=SpecializationGuard(
            (
                GuardPredicate("workload", "launches", "le", MAX_CAPTURE_LAUNCHES),
                GuardPredicate("workload", "precision", "eq", "fp64"),
                GuardPredicate("target", "backend", "eq", "cuda"),
            )
        ),
        artifact_key=artifact.metadata["key"],
        schedule_hash=canonical_hash(dataclasses.asdict(plan.schedule)),
        runtime_hash=canonical_hash(device),
        unsupported_effects=effects,
    )


class PreparedCuda:
    """Own independent native allocations, stream, cuBLAS handle and host staging.

    Repeated calls on one object serialize; different objects may run from
    different host threads. Shapes and budgets are fixed at preparation.
    Calls return independent output sets and never modify caller inputs.
    ``execution_mode="cuda-graph"`` opts into a warmup/capture/replay path.
    Host transfers and arithmetic checks remain outside the captured region.
    Ordinary execution remains the default and the safe ineligibility fallback.
    """

    def __init__(
        self,
        plan: TensorPlan,
        artifact: CudaArtifact,
        *,
        device: int = 0,
        resource_plan: ResourcePlan | None = None,
        resource_owner: str | None = None,
        execution_mode: str = "ordinary",
    ) -> None:
        if execution_mode not in ("ordinary", "cuda-graph"):
            raise ValueError("execution_mode must be ordinary or cuda-graph")
        self.execution_mode = execution_mode
        self.graph_status = "ordinary: graph mode not requested"
        if type(device) is not int or device < 0:
            raise ValueError("device must be a nonnegative visible CUDA ordinal")
        self._lock = threading.Lock()
        self._pointer = ctypes.c_void_p()
        self.plan = plan
        self.artifact = artifact
        self._prepared_plan, self._prepared_artifact = plan, artifact
        self.resource_plan = resource_plan
        self.resource_owner = resource_owner
        if resource_plan is not None:
            if resource_owner is None:
                raise ValueError("resource plan requires a named owner")
            resource_plan.require_feasible()
            request = next(
                (r for r in resource_plan.requests if r.name == resource_owner), None
            )
            if request is None or request.identity.provider != "tensorir-cuda":
                raise ValueError("global resource plan has no matching tensor owner")
            chosen = dict(resource_plan.selections)[resource_owner]
            candidate = next(c for c in request.candidates if c.name == chosen)
            if dict(candidate.decisions).get("tensor_plan") != plan.identity:
                raise ValueError(
                    "tensor plan differs from the globally selected alternative"
                )
            if json.loads(request.identity.topology)["device"] != device:
                raise ValueError("tensor device differs from its resource plan")
        elif resource_owner is not None:
            raise ValueError("resource owner requires a global plan")
        if file_hash(artifact.library) != artifact.metadata.get("binary_sha256"):
            raise ValueError("tensor artifact binary hash mismatch")
        from vibeqc_compiler.common.resources import ResourceAllocationError

        # Validate before retaining staging buffers or touching the native device.
        self._inputs, self._scratch, self._mask = [], [], None
        try:
            static_data = _read_static_data(artifact, plan.static_data_bytes)
        except MemoryError as error:
            raise ResourceAllocationError("host", str(error)) from error
        lib = self._library = ctypes.CDLL(str(artifact.library))
        lib.tensor_plan_identity.restype = ctypes.c_char_p
        if lib.tensor_plan_identity().decode() != plan.identity:
            raise ValueError("native tensor plan identity mismatch")
        lib.tensor_create.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.tensor_create.restype = ctypes.c_int
        lib.tensor_static_bytes.argtypes = []
        lib.tensor_static_bytes.restype = ctypes.c_size_t
        lib.tensor_static_initialize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.tensor_static_initialize.restype = ctypes.c_int
        lib.tensor_destroy.argtypes = [ctypes.c_void_p]
        lib.tensor_destroy.restype = None
        lib.tensor_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.POINTER(_Metrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.tensor_run.restype = ctypes.c_int
        lib.tensor_probe.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
        lib.tensor_probe.restype = ctypes.c_int
        error = ctypes.create_string_buffer(2048)
        if lib.tensor_probe(device, error, len(error)):
            raise RuntimeError(error.value.decode())
        self.device = json.loads(error.value)
        if self.device["architecture"] != plan.target.architecture:
            raise ValueError("tensor plan/device architecture mismatch")
        self.identity = canonical_hash(
            {
                "artifact": artifact.metadata["key"],
                "device": self.device,
                "precision": plan.precision,
                "host_layout": "C staging; arbitrary caller strides",
                "python": platform.python_version(),
                "numpy": np.__version__,
            }
        )
        self.capture_contract = tensor_capture_contract(
            plan, artifact, self.device, resource_plan=resource_plan
        )
        self._capture_identity = self.capture_contract.identity
        self._graph_enabled = (
            execution_mode == "cuda-graph" and self.capture_contract.eligible
        )
        self._graph_needs_setup = self._graph_enabled
        if execution_mode == "cuda-graph":
            self.graph_status = (
                "warmup required"
                if self._graph_enabled
                else "fallback: " + "; ".join(self.capture_contract.failures)
            )
        lib.tensor_graph_configure.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.tensor_graph_configure.restype = ctypes.c_int
        lib.tensor_run_graph.argtypes = [
            *lib.tensor_run.argtypes[:5],
            ctypes.POINTER(_GraphMetrics),
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.tensor_run_graph.restype = ctypes.c_int
        from vibeqc_compiler.common.resources import ResourceAllocationError

        try:
            self._inputs = [
                np.empty(
                    plan.steps[i].node.spec.shape, dtype=plan.steps[i].node.spec.dtype
                )
                for i in plan.inputs
            ]
            self._scratch = (
                [np.empty(VALIDATION_CHUNK, dtype=np.float64) for _ in range(2)]
                if plan.inputs
                else []
            )
            self._mask = (
                np.empty(VALIDATION_CHUNK, dtype=np.bool_) if plan.inputs else None
            )
        except MemoryError as error:
            # Tracebacks may retain this failed object while a group retries.
            # Release host owners now instead of relying on object collection.
            self._inputs, self._scratch, self._mask = [], [], None
            raise ResourceAllocationError("host", str(error)) from error
        if lib.tensor_static_bytes() != static_data.nbytes:
            raise ValueError("native tensor static-data size mismatch")
        static_pointer = (
            ctypes.c_void_p(static_data.ctypes.data) if static_data.nbytes else None
        )
        with _PREPARATION_LOCK:
            status = lib.tensor_create(
                device, ctypes.byref(self._pointer), error, len(error)
            )
            if not status:
                status = lib.tensor_static_initialize(
                    self._pointer,
                    static_pointer,
                    static_data.nbytes,
                    error,
                    len(error),
                )
        if status:
            if self._pointer:
                lib.tensor_destroy(self._pointer)
                self._pointer = ctypes.c_void_p()
            self._inputs, self._scratch, self._mask = [], [], None
            if status in (2, 3):
                raise ResourceAllocationError(
                    f"device:{device}" if status == 2 else "host", error.value.decode()
                )
            raise RuntimeError(error.value.decode())

        if self._graph_enabled:
            with _PREPARATION_LOCK:
                status = lib.tensor_graph_configure(
                    self._pointer,
                    1,
                    self._capture_identity.encode(),
                    error,
                    len(error),
                )
            if status:
                self.close()
                raise RuntimeError(error.value.decode())

    def invalidate_graph(self) -> None:
        """Discard replay state without changing buffers or the immutable plan.

        Shape/schedule/artifact/device changes require a new PreparedCuda owner.
        This hook supports explicit method-state epoch transitions in future
        consumers; it never edits scientific data or relaxes convergence checks.
        """
        with self._lock:
            if not self._pointer:
                raise RuntimeError("tensor plan is closed")
            if self._graph_enabled:
                error = ctypes.create_string_buffer(2048)
                with _PREPARATION_LOCK:
                    status = self._library.tensor_graph_configure(
                        self._pointer,
                        1,
                        self._capture_identity.encode(),
                        error,
                        len(error),
                    )
                if status:
                    raise RuntimeError(error.value.decode())
                self._graph_needs_setup = True
                self.graph_status = "invalidated; warmup required"

    def _validate(self, value: np.ndarray, node: Node) -> None:
        """Bound validation scratch even for transposed symmetry partners."""
        if self._mask is None:
            raise RuntimeError("tensor validation scratch is closed")
        if node.spec.dtype == "int64":
            return
        flat = value.reshape(-1)
        for start in range(0, flat.size, VALIDATION_CHUNK):
            chunk = flat[start : start + VALIDATION_CHUNK]
            mask = self._mask[: chunk.size]
            np.isfinite(chunk, out=mask)
            if not mask.all():
                raise ValueError(f"non-finite tensor input: {node.attrs['name']}")
        if not value.size:
            return
        atol, rtol = symmetry_tolerance(node.spec.dtype)
        for symmetry in node.spec.symmetries:
            iterator = np.nditer(
                (value, value.transpose(symmetry.permutation)),
                flags=("external_loop", "buffered"),
                op_flags=(("readonly",), ("readonly",)),
                buffersize=VALIDATION_CHUNK,
                order="C",
            )
            for left, right in iterator:
                delta, tolerance = (v[: left.size] for v in self._scratch)
                np.multiply(right, symmetry.sign, out=delta, dtype=np.float64)
                np.subtract(left, delta, out=delta)
                np.abs(delta, out=delta)
                np.abs(right, out=tolerance)
                np.multiply(tolerance, rtol, out=tolerance)
                np.add(tolerance, atol, out=tolerance)
                mask = self._mask[: left.size]
                np.less_equal(delta, tolerance, out=mask)
                if not mask.all():
                    raise ValueError(
                        f"input {node.attrs['name']} violates its declared symmetry"
                    )

    def execute(
        self,
        feeds: Mapping[str, typing.Any],
        *,
        profile: bool = False,
        diagnostics: bool = False,
    ) -> CudaExecution:
        """Stage/validate feeds, then make one native call for the whole program.

        The endpoint timer includes Python validation and layout staging,
        device transfers, every contraction/packing kernel, error checks and
        detached result allocation. Optional section profiling synchronizes
        individual sections and must not be used for optimization selection.
        ``diagnostics=True`` adds native region-submission/graph counters to an
        ordinary run without inserting section fences; graph mode always reports them.
        """
        with self._lock:
            if not self._pointer:
                raise RuntimeError("tensor plan is closed")
            if (
                self.plan is not self._prepared_plan
                or self.artifact is not self._prepared_artifact
            ):
                raise ValueError("prepared plan/artifact changed; create a new owner")
            if not isinstance(feeds, Mapping):
                raise TypeError("tensor feeds must be a mapping")
            started = time.perf_counter()
            for i, staged in zip(self.plan.inputs, self._inputs, strict=True):
                node = self.plan.steps[i].node
                name = node.attrs["name"]
                if name not in feeds:
                    raise ValueError(f"missing tensor input: {name}")
                # Requiring an ndarray avoids an unbudgeted whole-input copy
                # before shape/dtype checks. Noncontiguous ndarrays are legal.
                value = feeds[name]
                if (
                    not isinstance(value, np.ndarray)
                    or value.dtype != np.dtype(node.spec.dtype)
                    or value.shape != node.spec.shape
                ):
                    raise ValueError(
                        f"input {name} must be a {node.spec.dtype} ndarray with shape {node.spec.shape}"
                    )
                np.copyto(staged, value)
                self._validate(staged, node)
            outputs = {
                name: np.empty(
                    self.plan.steps[i].node.spec.shape,
                    dtype=self.plan.steps[i].node.spec.dtype,
                )
                for name, i in self.plan.outputs
            }
            input_ptrs = (ctypes.c_void_p * max(1, len(self._inputs)))(
                *[a.ctypes.data for a in self._inputs]
            )
            output_ptrs = (ctypes.c_void_p * len(outputs))(
                *[a.ctypes.data for a in outputs.values()]
            )
            native, error = _Metrics(), ctypes.create_string_buffer(2048)
            graph_metrics = {}
            args = (
                self._pointer,
                input_ptrs,
                output_ptrs,
                profile,
                ctypes.byref(native),
            )
            if self.execution_mode == "cuda-graph" or diagnostics:
                graph, reason = _GraphMetrics(), ctypes.create_string_buffer(512)
                # Serialize capture/allocation snapshots, not steady-state replay.
                with _PREPARATION_LOCK if self._graph_needs_setup else nullcontext():
                    status = self._library.tensor_run_graph(
                        *args,
                        ctypes.byref(graph),
                        reason,
                        len(reason),
                        error,
                        len(error),
                    )
                if graph.mode in (2, 3, 4):
                    self._graph_needs_setup = False
                mode = GRAPH_MODES[graph.mode]
                message = reason.value.decode()
                if self.execution_mode == "cuda-graph" and not self._graph_enabled:
                    mode, message = (
                        "fallback",
                        "; ".join(self.capture_contract.failures),
                    )
                self.graph_status = f"{mode}: {message}"
                graph_metrics = {
                    f"graph_{name}": getattr(graph, name)
                    for name, _ in graph._fields_
                    if name != "mode"
                }
                graph_metrics.update(
                    graph_mode=mode,
                    graph_reason=message,
                    capture_contract=self._capture_identity,
                )
            else:
                status = self._library.tensor_run(*args, error, len(error))
            if status:
                raise RuntimeError(error.value.decode())
            metrics = {name: getattr(native, name) for name, _ in native._fields_}
            traffic = self.plan.semantic_traffic
            metrics.update(
                endpoint_ms=(time.perf_counter() - started) * 1000,
                predicted_peak_bytes=self.plan.peak_bytes,
                host_buffer_bytes=self.plan.host_bytes,
                observed_semantic_traffic_bytes=traffic["total_bytes"],
                observed_logical_tensor_bytes=traffic["logical_tensor_bytes"],
                observed_layout_conversion_bytes=traffic["layout_conversion_bytes"],
                observed_host_to_device_bytes=traffic["host_to_device_bytes"],
                observed_device_to_host_bytes=traffic["device_to_host_bytes"],
                observed_traffic_scope=traffic["scope"]
                + "; bound to a successfully executed endpoint, not a hardware DRAM counter",
                profiled=bool(profile),
                **graph_metrics,
            )
            if (
                native.owned_device_bytes != self.plan.allocation_bytes
                or native.provider_retained_bytes > self.plan.provider_bytes
            ):
                raise RuntimeError(
                    "native tensor allocation disagrees with the memory plan"
                )
            if self.resource_plan is not None:
                host_owned = sum(
                    a.nbytes for a in (*self._inputs, *self._scratch, *outputs.values())
                )
                host_owned += 0 if self._mask is None else self._mask.nbytes
                if host_owned > self.plan.host_bytes:
                    raise RuntimeError(
                        "owned tensor host buffers exceed the resource plan"
                    )
                metrics.update(
                    resource_plan_id=self.resource_plan.identity,
                    resource_owner=self.resource_owner,
                    tracked_device_bytes=native.owned_device_bytes
                    + native.provider_retained_bytes,
                    tracked_host_bytes=host_owned,
                    resource_tracking_scope="owned ndarrays, native device buffers and measured retained provider allocations; iterator/runtime overhead reported separately",
                )
            metrics["precision"] = self.plan.precision
            backend = (
                f"cuda-{self.plan.precision}-graph-replay"
                if graph_metrics.get("graph_mode") in ("captured", "replay")
                else f"cuda-{self.plan.precision}-ordinary-stream"
            )
            return CudaExecution(outputs, metrics, backend)

    def close(self) -> None:
        """Release resources once; cannot race an execution using their pointers."""
        with self._lock:
            if self._pointer:
                with _PREPARATION_LOCK:
                    self._library.tensor_destroy(self._pointer)
                self._pointer = ctypes.c_void_p()
            self._inputs, self._scratch, self._mask = [], [], None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_pointer", None):
            self.close()

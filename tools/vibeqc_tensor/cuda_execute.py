"""Compile, prepare and execute complete FP64 TensorIR programs on CUDA.

Compilation is explicit and may run without a GPU. Preparation, probing and
execution require the caller's allocated GPU (on this workstation, Slurm).
There is no installation-time tuning, implicit CPU arithmetic fallback or
per-contraction Python loop.
"""

from __future__ import annotations

import ctypes
import dataclasses
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vibeqc.profiles import atomic_json, canonical_hash, file_hash, toolchain_identity

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter

from .cuda_emit import emit_cuda
from .cuda_plan import VALIDATION_CHUNK, TensorPlan
from .cuda_resources import parse_resources

# Allocation snapshots for provider accounting must not race another owned
# handle's creation/destruction. Executions themselves remain independent.
_PREPARATION_LOCK = threading.RLock()


class _Metrics(ctypes.Structure):
    _fields_ = [
        ("owned_device_bytes", ctypes.c_uint64),
        ("provider_retained_bytes", ctypes.c_uint64),
        ("prepare_device_delta", ctypes.c_uint64),
        ("observed_device_delta", ctypes.c_uint64),
        *[
            (name, ctypes.c_double)
            for name in (
                "device_ms",
                "input_ms",
                "output_ms",
                "packing_ms",
                "library_ms",
                "kernel_ms",
            )
        ],
    ]


@dataclass(frozen=True)
class CudaArtifact:
    """Content-verified native artifact; compilation alone is not validation."""

    library: Path
    metadata: dict


@dataclass(frozen=True)
class CudaExecution:
    """Detached named outputs and complete execution/profiling measurements."""

    outputs: dict[str, np.ndarray]
    metrics: dict
    backend: str = "cuda-fp64-ordinary-stream"


def tensor_source_identity() -> str:
    """Inventory tensor sources separately from the shell-class source catalog.

    Reuse #136's canonical/file hashes and atomic publication, but never use a
    shell profile identity for tensor code it does not inventory.
    """
    root = Path(__file__).resolve().parents[2]
    paths = [*Path(__file__).parent.glob("*.py"), *(root / "src/tensor").glob("*.cuh")]
    paths += list((root / "tools/vibeqc_codegen").glob("*.py"))
    paths += list((root / "tools/vibeqc_validation").glob("*.py"))
    paths += [root / "benchmarks/aot_shell_batch_gate.py"]
    paths += [root / "python/vibeqc/profiles.py"]
    return canonical_hash(
        {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}
    )


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
    source = emit_cuda(plan)
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
        "schema": 1,
        "plan": plan.identity,
        "source": tensor_source_identity(),
        "generated": canonical_hash(source),
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
        "options": ["--fmad=false", "c++17", "O3", "shared", "fPIC", "cublas"],
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
            cu.write_text(source)
            result = compiler.compile_shared(
                cu,
                library,
                includes=(Path(__file__).resolve().parents[2] / "src/tensor",),
                libraries=("cublas",),
                options=("--fmad=false",),
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
                "compile_seconds": result.duration_seconds,
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
    if (
        metadata.get("identity") != json.loads(json.dumps(identity))
        or metadata.get("key") != key
        or file_hash(library) != metadata.get("binary_sha256")
    ):
        raise ValueError("tensor artifact identity or binary hash mismatch")
    return CudaArtifact(library, metadata)


class PreparedCuda:
    """Own independent native allocations, stream, cuBLAS handle and host staging.

    Repeated calls on one object serialize; different objects may run from
    different host threads. Shapes and budgets are fixed at preparation.
    Calls return independent output sets and never modify caller inputs.
    Only ordinary streams are implemented: ``graph_status`` reports this
    explicitly, so graph capture is never a hidden correctness prerequisite.
    """

    graph_status = "ordinary-stream: graph capture is not enabled for TensorIR"

    def __init__(self, plan: TensorPlan, artifact: CudaArtifact, *, device: int = 0):
        if type(device) is not int or device < 0:
            raise ValueError("device must be a nonnegative visible CUDA ordinal")
        self._lock = threading.Lock()
        self._pointer = ctypes.c_void_p()
        self.plan = plan
        self.artifact = artifact
        if file_hash(artifact.library) != artifact.metadata.get("binary_sha256"):
            raise ValueError("tensor artifact binary hash mismatch")
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
                "precision": "fp64",
                "host_layout": "C staging; arbitrary caller strides",
                "python": platform.python_version(),
                "numpy": np.__version__,
            }
        )
        self._inputs = [
            np.empty(plan.steps[i].node.spec.shape, dtype=np.float64)
            for i in plan.inputs
        ]
        self._scratch = (
            [np.empty(VALIDATION_CHUNK, dtype=np.float64) for _ in range(2)]
            if plan.inputs
            else []
        )
        self._mask = np.empty(VALIDATION_CHUNK, dtype=np.bool_) if plan.inputs else None
        with _PREPARATION_LOCK:
            if lib.tensor_create(
                device, ctypes.byref(self._pointer), error, len(error)
            ):
                raise RuntimeError(error.value.decode())

    def _validate(self, value, node):
        """Bound validation scratch even for transposed symmetry partners."""
        flat = value.reshape(-1)
        for start in range(0, flat.size, VALIDATION_CHUNK):
            chunk = flat[start : start + VALIDATION_CHUNK]
            mask = self._mask[: chunk.size]
            np.isfinite(chunk, out=mask)
            if not mask.all():
                raise ValueError(f"non-finite tensor input: {node.attrs['name']}")
        if not value.size:
            return
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
                np.multiply(right, symmetry.sign, out=delta)
                np.subtract(left, delta, out=delta)
                np.abs(delta, out=delta)
                np.abs(right, out=tolerance)
                np.multiply(tolerance, 1e-10, out=tolerance)
                np.add(tolerance, 1e-11, out=tolerance)
                mask = self._mask[: left.size]
                np.less_equal(delta, tolerance, out=mask)
                if not mask.all():
                    raise ValueError(
                        f"input {node.attrs['name']} violates its declared symmetry"
                    )

    def execute(self, feeds: Mapping, *, profile: bool = False) -> CudaExecution:
        """Stage/validate feeds, then make one native call for the whole program.

        The endpoint timer includes Python validation and layout staging,
        device transfers, every contraction/packing kernel, error checks and
        detached result allocation. Optional section profiling synchronizes
        individual sections and must not be used for optimization selection.
        """
        with self._lock:
            if not self._pointer:
                raise RuntimeError("tensor plan is closed")
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
                    or value.dtype != np.float64
                    or value.shape != node.spec.shape
                ):
                    raise ValueError(
                        f"input {name} must be an FP64 ndarray with shape {node.spec.shape}"
                    )
                np.copyto(staged, value)
                self._validate(staged, node)
            outputs = {
                name: np.empty(self.plan.steps[i].node.spec.shape, dtype=np.float64)
                for name, i in self.plan.outputs
            }
            input_ptrs = (ctypes.c_void_p * max(1, len(self._inputs)))(
                *[a.ctypes.data for a in self._inputs]
            )
            output_ptrs = (ctypes.c_void_p * len(outputs))(
                *[a.ctypes.data for a in outputs.values()]
            )
            native, error = _Metrics(), ctypes.create_string_buffer(2048)
            if self._library.tensor_run(
                self._pointer,
                input_ptrs,
                output_ptrs,
                profile,
                ctypes.byref(native),
                error,
                len(error),
            ):
                raise RuntimeError(error.value.decode())
            metrics = {name: getattr(native, name) for name, _ in native._fields_}
            metrics.update(
                endpoint_ms=(time.perf_counter() - started) * 1000,
                predicted_peak_bytes=self.plan.peak_bytes,
                host_buffer_bytes=self.plan.host_bytes,
                profiled=bool(profile),
            )
            if (
                native.owned_device_bytes != self.plan.allocation_bytes
                or native.provider_retained_bytes > self.plan.provider_bytes
            ):
                raise RuntimeError(
                    "native tensor allocation disagrees with the memory plan"
                )
            return CudaExecution(outputs, metrics)

    def close(self):
        """Release resources once; cannot race an execution using their pointers."""
        with self._lock:
            if self._pointer:
                with _PREPARATION_LOCK:
                    self._library.tensor_destroy(self._pointer)
                self._pointer = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        if getattr(self, "_pointer", None):
            self.close()

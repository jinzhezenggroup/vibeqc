"""Hash-checked shared compilation and bounded native XC tile execution."""

import ctypes as ct
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.cuda_runtime import _PREPARATION_LOCK, _Metrics
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.dft.grid import checked_int

from .cuda_emit import XCSchedule, emit_cuda
from .program import validate_features

DOUBLE = ct.POINTER(ct.c_double)


@dataclass(frozen=True)
class XCArtifact:
    """Runtime artifact plus generated consumer/schedule contract."""

    runtime: object
    contract: dict


def compile_cuda(program, compiler, cache, *, schedule=None):
    """Reuse the existing compiler/resource/cache adapter; compilation is CPU-only."""
    source, contract, headers = emit_cuda(program, schedule)
    directory = Path(cache).resolve() / "source" / contract["identity"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "xc.cu"
    if path.exists() and path.read_text() != source:
        raise ValueError("XC source identity mismatch")
    path.write_text(source)
    options = ("--fmad=false", *(f"-I{p}" for p in sorted({h.parent for h in headers})))
    runtime = compile_runtime(
        compiler, cache, path, headers=headers, libraries=("cublas",), options=options
    )
    # The runtime has no cuBLAS handle/workspace. The common header references
    # cuBLAS symbols, so linking the shared library remains explicit.
    return XCArtifact(runtime, contract)


@dataclass(frozen=True)
class XCTilePlan:
    """Conservative simultaneous numeric capacities, excluding caller input."""

    tile_points: int
    device_bytes: int
    host_bytes: int
    budget_bytes: int
    identity: str

    @property
    def allocation_bytes(self):
        return self.device_bytes + self.host_bytes


def plan_tiles(program, *, tile_points=256, budget_bytes=64 << 20):
    """Bound owned device IO/error storage and validation/output host scratch."""
    checked_int(tile_points, "XC tile points", low=1, high=1 << 24)
    checked_int(budget_bytes, "XC budget", low=1)
    count = len(program.spec.features) + len(program.outputs)
    device = tile_points * count * 8 + 256
    # Host validation: input copy plus density/sigma masks and arithmetic;
    # output is detached. Charge 32 doubles/point beyond complete IO arrays.
    host = tile_points * (count + 32) * 8
    if device + host > budget_bytes:
        raise ValueError("XC numeric capacities exceed budget")
    identity = canonical_hash(
        {
            "expression": program.expression_hash,
            "tile_points": tile_points,
            "device": device,
            "host": host,
            "budget": budget_bytes,
        }
    )
    return XCTilePlan(tile_points, device, host, budget_bytes, identity)


class CudaXC:
    """Persistent private stream/arena; explicit upload, device arithmetic, download.

    A call evaluates one bounded tile and allocates no device storage. Calls on
    one owner are serialized. Independent owners never share mutable buffers.
    No CPU arithmetic fallback is selected on compilation/execution failure.
    """

    def __init__(
        self, program, artifact, *, tile_points=256, budget_bytes=64 << 20, device_id=0
    ):
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self.program = program
        self.plan = plan_tiles(
            program, tile_points=tile_points, budget_bytes=budget_bytes
        )
        checked_int(device_id, "visible device ordinal", low=0)
        if artifact.contract["expression_hash"] != program.expression_hash:
            raise ValueError("XC artifact/program mismatch")
        schedule = XCSchedule(
            artifact.contract["variant"],
            artifact.contract["threads"],
            len(artifact.contract["groups"][0]),
        )
        _, expected, _ = emit_cuda(program, schedule)
        if expected["identity"] != artifact.contract["identity"]:
            raise ValueError("XC artifact/generator contract mismatch")
        runtime = artifact.runtime
        if file_hash(runtime.library) != runtime.metadata["binary_sha256"]:
            raise ValueError("XC binary hash mismatch")
        self.artifact = artifact
        lib = self._library = ct.CDLL(str(runtime.library))
        lib.xc_identity_v1.restype = ct.c_char_p
        if lib.xc_identity_v1().decode() != artifact.contract["identity"]:
            raise ValueError("XC loaded artifact identity mismatch")
        lib.xc_create_v1.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.xc_destroy_v1.argtypes = [ct.c_void_p]
        lib.xc_destroy_v1.restype = None
        lib.xc_run_v1.argtypes = [
            ct.c_void_p,
            DOUBLE,
            ct.c_size_t,
            DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.xc_metrics_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(_Metrics),
            ct.POINTER(ct.c_int),
            ct.c_char_p,
            ct.c_size_t,
        ]
        architecture = runtime.metadata["identity"]["target"]["architecture"]
        number = int(architecture.removeprefix("sm_"))
        with _PREPARATION_LOCK:
            self._call(
                "xc_create_v1",
                device_id,
                number // 10,
                number % 10,
                tile_points,
                self.plan.device_bytes,
                ct.byref(self._handle),
            )

    def _call(self, name, *args):
        error = ct.create_string_buffer(2048)
        if getattr(self._library, name)(*args, error, len(error)):
            raise RuntimeError(error.value.decode())

    def _check_open(self):
        if not self._handle:
            raise RuntimeError("XC CUDA plan is closed")

    def evaluate(self, features):
        """Evaluate one tile in published output order, rejecting invalid domains."""
        with self._lock:
            self._check_open()
            raw = np.asarray(features)
            if raw.ndim != 2 or raw.shape[1] > self.plan.tile_points:
                raise ValueError("features exceed XC prepared tile shape")
            x, _ = validate_features(self.program.spec, raw, order=self.program.order)
            result = np.empty((len(self.program.outputs), x.shape[1]))
            self._call(
                "xc_run_v1",
                self._handle,
                x.ctypes.data_as(DOUBLE),
                x.shape[1],
                result.ctypes.data_as(DOUBLE),
            )
            return result

    def metrics(self):
        """Synchronized cumulative timings, capacities and actual CUDA versions."""
        with self._lock:
            self._check_open()
            metrics = _Metrics()
            versions = (ct.c_int * 2)()
            self._call("xc_metrics_v1", self._handle, ct.byref(metrics), versions)
            return {
                **{k: getattr(metrics, k) for k, _ in metrics._fields_},
                "runtime_version": versions[0],
                "driver_version": versions[1],
            }

    def close(self):
        """Release the arena and stream; repeated closure is harmless."""
        with self._lock, _PREPARATION_LOCK:
            if self._handle:
                self._library.xc_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()

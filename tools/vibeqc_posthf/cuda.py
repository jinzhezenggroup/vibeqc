"""Explicit CUDA compilation and resident cuBLAS MO-block execution."""

from __future__ import annotations

import ctypes as ct
import threading
from pathlib import Path

import numpy as np
from vibeqc.profiles import file_hash

from tools.vibeqc_codegen.native_runtime import compile_runtime
from tools.vibeqc_tensor.cuda_execute import _PREPARATION_LOCK, _Metrics

from .reference import immutable
from .sources import _DOUBLE, _SIZE, pointer


def compile_cuda(compiler, cache):
    """Build the bounded transform through the shared native-runtime cache."""
    root = Path(__file__).resolve().parents[2]
    return compile_runtime(
        compiler,
        cache,
        root / "src/posthf/cuda_transform.cu",
        headers=(root / "src/tensor/cuda_runtime.cuh",),
        libraries=("cublas",),
    )


class CudaTransform:
    """Own one resident MO output plus reusable cuBLAS stream/workspaces.

    ``device_pointer`` is borrowed only while this object remains open; the
    object must accompany any downstream native CC call. ``to_host`` is an
    explicit copy. Ordinary streams are used; no CUDA graph support is claimed.
    """

    def __init__(self, artifact, plan, coefficients, *, device_id=0):
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self.plan = plan
        self.artifact = artifact
        if type(device_id) is not int or device_id < 0:
            raise ValueError("invalid visible device ordinal")
        if file_hash(artifact.library) != artifact.metadata["binary_sha256"]:
            raise ValueError("CUDA binary hash mismatch")
        lib = self._library = ct.CDLL(str(artifact.library))
        lib.posthf_cuda_create_v1.argtypes = [
            ct.c_int,
            ct.c_size_t,
            _SIZE,
            _SIZE,
            _DOUBLE,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.posthf_cuda_destroy_v1.argtypes = [ct.c_void_p]
        lib.posthf_cuda_destroy_v1.restype = None
        lib.posthf_cuda_add_v1.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            _SIZE,
            _SIZE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.posthf_cuda_download_v1.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            ct.c_size_t,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.posthf_cuda_metrics_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(_Metrics),
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.posthf_cuda_pointer_v1.argtypes = [ct.c_void_p]
        lib.posthf_cuda_pointer_v1.restype = ct.c_void_p
        lib.posthf_cuda_versions_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_int),
            ct.c_char_p,
            ct.c_size_t,
        ]
        packed = np.concatenate([c.ravel() for c in coefficients])
        with _PREPARATION_LOCK:
            self._call(
                "posthf_cuda_create_v1",
                device_id,
                coefficients[0].shape[0],
                (ct.c_size_t * 4)(*plan.shape),
                (ct.c_size_t * 4)(*plan.tile_shape),
                pointer(packed),
                plan.allocation_bytes,
                ct.byref(self._handle),
            )
        self.device_id = device_id

    def _call(self, name, *args):
        error = ct.create_string_buffer(2048)
        if getattr(self._library, name)(*args, error, len(error)):
            raise RuntimeError(error.value.decode())

    def _check_open(self):
        if not self._handle:
            raise RuntimeError("CUDA MO block is closed")

    def add(self, tile, offsets):
        """Accumulate one AO shell tile after four device-only transformations."""
        with self._lock:
            self._check_open()
            if (
                tile.dtype != np.float64
                or not tile.flags.c_contiguous
                or tile.ndim != 4
            ):
                raise ValueError("CUDA AO tile must be contiguous FP64 rank four")
            self._call(
                "posthf_cuda_add_v1",
                self._handle,
                pointer(tile),
                (ct.c_size_t * 4)(*offsets),
                (ct.c_size_t * 4)(*tile.shape),
            )

    @property
    def device_pointer(self):
        self._check_open()
        return self._library.posthf_cuda_pointer_v1(self._handle)

    def to_host(self):
        """Return an immutable detached FP64 block; account transfers separately."""
        with self._lock:
            self._check_open()
            out = np.empty(self.plan.shape)
            self._call("posthf_cuda_download_v1", self._handle, pointer(out), out.size)
            return immutable(out)

    def metrics(self):
        """Measured owned allocations and synchronized section timings."""
        with self._lock:
            self._check_open()
            record = _Metrics()
            self._call("posthf_cuda_metrics_v1", self._handle, ct.byref(record))
            versions = (ct.c_int * 3)()
            self._call("posthf_cuda_versions_v1", self._handle, versions)
            return {
                **{name: getattr(record, name) for name, _ in record._fields_},
                "runtime_version": versions[0],
                "driver_version": versions[1],
                "cublas_version": versions[2],
            }

    def close(self):
        with self._lock, _PREPARATION_LOCK:
            if self._handle:
                self._library.posthf_cuda_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()

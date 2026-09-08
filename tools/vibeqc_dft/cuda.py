"""Explicit CUDA AO/feature execution with a persistent private plan/arena."""

from __future__ import annotations

import ctypes as ct
import threading
from pathlib import Path

import numpy as np
from vibeqc.profiles import file_hash

from tools.vibeqc_codegen.native_runtime import compile_runtime
from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_tensor.cuda_execute import _PREPARATION_LOCK, _Metrics

from .ao import DOUBLE, SIZE, jet_indices, pointer
from .features import spin_densities
from .grid import checked_int
from .plan import plan_tiles


def compile_cuda(compiler, cache):
    """Compile the device runtime without running a GPU or importing PySCF."""
    root = Path(__file__).resolve().parents[2]
    return compile_runtime(
        compiler,
        cache,
        root / "src/dft/cuda_grid.cu",
        headers=(root / "src/tensor/cuda_runtime.cuh",),
        libraries=("cublas",),
    )


class CudaGrid:
    """Own basis/D/tiles and execute AO jets, cuBLAS contractions and invariants.

    Grid generation and partition weights remain on the CPU. Points upload
    explicitly, features download explicitly, and AO jets download only when
    requested for validation. No full molecular AO grid is retained. Calls on
    one plan are serialized; different plans own independent arenas/streams.
    """

    def __init__(
        self,
        basis,
        artifact,
        *,
        order=1,
        tile_points=256,
        budget_bytes=256 << 20,
        device_id=0,
        grid=None,
    ):
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self._density_ready = False
        checked_int(device_id, "visible device ordinal", low=0)
        self.plan = plan_tiles(
            basis,
            backend="cuda",
            order=order,
            tile_points=tile_points,
            budget_bytes=budget_bytes,
            grid=grid,
        )
        if file_hash(artifact.library) != artifact.metadata["binary_sha256"]:
            raise ValueError("CUDA grid binary hash mismatch")
        self.basis_identity = basis.identity
        self.artifact = artifact
        self.device_id = device_id
        lib = self._library = ct.CDLL(str(artifact.library))
        lib.grid_cuda_create_v1.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_int,
            SIZE,
            DOUBLE,
            ct.c_size_t,
            ct.c_uint,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.grid_cuda_destroy_v1.argtypes = [ct.c_void_p]
        lib.grid_cuda_destroy_v1.restype = None
        lib.grid_cuda_density_v1.argtypes = [
            ct.c_void_p,
            DOUBLE,
            ct.c_size_t,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.grid_cuda_run_v1.argtypes = [
            ct.c_void_p,
            DOUBLE,
            ct.c_size_t,
            ct.c_int,
            DOUBLE,
            DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        lib.grid_cuda_metrics_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(_Metrics),
            ct.POINTER(ct.c_int),
            ct.c_char_p,
            ct.c_size_t,
        ]
        architecture = artifact.metadata["identity"]["target"]["architecture"]
        number = int(architecture.removeprefix("sm_"))
        with _PREPARATION_LOCK:
            self._call(
                "grid_cuda_create_v1",
                device_id,
                number // 10,
                number % 10,
                (ct.c_size_t * 3)(basis.natom, basis.nprimitive, basis.nao),
                pointer(basis.packed),
                tile_points,
                order,
                self.plan.allocation_bytes,
                ct.byref(self._handle),
            )

    def _call(self, name, *args):
        error = ct.create_string_buffer(2048)
        if getattr(self._library, name)(*args, error, len(error)):
            raise RuntimeError(error.value.decode())

    def _check_open(self):
        if not self._handle:
            raise RuntimeError("CUDA grid plan is closed")

    def set_density(self, density):
        """Validate and replace both spin matrices; no old-density reuse is implicit."""
        with self._lock:
            self._check_open()
            d = spin_densities(density, self.plan.nao)
            self._call("grid_cuda_density_v1", self._handle, pointer(d), d.size)
            self._density_ready = True

    def evaluate(self, points, *, features=True, download_jets=False):
        """Return one detached result tile; no downstream CPU arithmetic fallback."""
        raw = np.asarray(points)
        if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) > self.plan.tile_points:
            raise ValueError("grid points exceed the prepared tile shape")
        if (
            type(features) is not bool
            or type(download_jets) is not bool
            or not (features or download_jets)
        ):
            raise ValueError("request features and/or AO jets")
        with self._lock:
            self._check_open()
            if features and (self.plan.order < 1 or not self._density_ready):
                raise ValueError(
                    "features require first derivatives and supplied density"
                )
            points = immutable(raw)
            values = np.empty((13, len(points))) if features else None
            jets = (
                np.empty(
                    (len(jet_indices(self.plan.order)), len(points), self.plan.nao)
                )
                if download_jets
                else None
            )
            self._call(
                "grid_cuda_run_v1",
                self._handle,
                pointer(points),
                len(points),
                int(features),
                pointer(values) if values is not None else None,
                pointer(jets) if jets is not None else None,
            )
            result = {}
            if features:
                # Only views/layout publication occur here. All four invariants
                # have already been contracted on the GPU, including sigma.
                result = {
                    "rho": immutable(values[[0, 5]]),
                    "gradient": immutable(
                        values[[1, 2, 3, 6, 7, 8]]
                        .reshape(2, 3, len(points))
                        .transpose(0, 2, 1)
                    ),
                    "tau": immutable(values[[4, 9]]),
                    "sigma": immutable(values[10:13]),
                }
            if download_jets:
                result["ao_jets"] = immutable(jets)
            return result

    def metrics(self):
        """Synchronized cumulative timings, owned allocations and loaded versions."""
        with self._lock:
            self._check_open()
            metrics = _Metrics()
            versions = (ct.c_int * 3)()
            self._call(
                "grid_cuda_metrics_v1", self._handle, ct.byref(metrics), versions
            )
            return {
                **{name: getattr(metrics, name) for name, _ in metrics._fields_},
                "runtime_version": versions[0],
                "driver_version": versions[1],
                "cublas_version": versions[2],
            }

    def close(self):
        with self._lock, _PREPARATION_LOCK:
            if self._handle:
                self._library.grid_cuda_destroy_v1(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()

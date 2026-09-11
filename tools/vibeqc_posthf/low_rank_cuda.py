"""Resident CUDA prefix projection and J/K with explicit CPU pivot/source work."""

import ctypes as ct
import json
import time
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cuda_runtime import _PREPARATION_LOCK, _Metrics
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.common.resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    checked_bytes,
)

from .low_rank import IncrementalCholesky
from .low_rank_consumers import LowRankJk
from .sources import _DOUBLE, pointer

WORKSPACE_BYTES = 4 << 20
PROVIDER_ALLOWANCE = 96 << 20


def compile_cholesky_cuda(compiler, cache):
    """Compile the generic native policy through the shared cache; no GPU probe."""
    root = Path(__file__).resolve().parents[2]
    return compile_runtime(
        compiler,
        cache,
        root / "src/posthf/cuda_low_rank.cu",
        headers=(root / "src/tensor/cuda_runtime.cuh",),
        libraries=("cublas",),
        options=("--fmad=false",),
    )


def cuda_factor_request(nbf, rank_capacity, device_id):
    """Plan the resident mirror, reusable matrices and cuBLAS allowance exactly."""
    checked_bytes(nbf, "AO dimension")
    checked_bytes(rank_capacity, "rank capacity")
    checked_bytes(device_id, "visible device ordinal")
    if not nbf or nbf * nbf > 2**31 - 1 or device_id >= 2**31:
        raise ValueError("CUDA low-rank dimensions exceed the native/cuBLAS ABI")
    pairs = nbf * (nbf + 1) // 2
    if rank_capacity > pairs:
        raise ValueError("rank capacity exceeds pair dimension")
    numeric = byte_product(8, rank_capacity * pairs + 2 * pairs + 8 * nbf * nbf + 1)
    arena = ((numeric + 255) // 256) * 256 + 256 + WORKSPACE_BYTES
    request = ResourceRequest(
        "cuda_factor_execution",
        ResourceIdentity(
            "posthf",
            "cholesky_prefix_and_jk",
            "cuda",
            "fp64",
            json.dumps({"nbf": nbf, "rank_capacity": rank_capacity}),
            ("Schur_column", "J_K"),
            "resident_prefix_cublas_v1",
        ),
        (
            ResourceCandidate(
                "resident_prefix",
                "resident",
                (
                    ResourceEstimate(
                        "cuda_arena", arena, f"device:{device_id}", 0, 1, "persistent"
                    ),
                    ResourceEstimate(
                        "cublas_retained_allowance",
                        PROVIDER_ALLOWANCE,
                        f"device:{device_id}",
                        0,
                        1,
                        "library",
                        accounting="runtime_allowance",
                    ),
                    ResourceEstimate(
                        "jk_host_staging",
                        byte_product(8, 20, nbf, nbf),
                        "pageable",
                        0,
                        1,
                    ),
                ),
            ),
        ),
        ("CUDA context and driver code", "CUDA call stacks", "Python metadata"),
    )
    return request, arena


class _NativePrefix:
    """One shared Context; host and native generations must agree on every call."""

    def __init__(self, artifact, nbf, capacity, device_id, arena):
        self._handle = ct.c_void_p()
        metadata = artifact.metadata
        if (
            canonical_hash(metadata["identity"]) != metadata["key"]
            or file_hash(artifact.library) != metadata["binary_sha256"]
        ):
            raise ValueError("CUDA low-rank artifact identity mismatch")
        target = metadata["identity"]["target"]
        lib = self._library = ct.CDLL(str(artifact.library))
        error = [ct.c_char_p, ct.c_size_t]
        lib.posthf_cholesky_create_v1.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(ct.c_void_p),
            *error,
        ]
        lib.posthf_cholesky_destroy_v1.argtypes = [ct.c_void_p]
        lib.posthf_cholesky_destroy_v1.restype = None
        lib.posthf_cholesky_project_v1.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            _DOUBLE,
            _DOUBLE,
            ct.c_size_t,
            *error,
        ]
        lib.posthf_cholesky_commit_v1.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            _DOUBLE,
            ct.c_size_t,
            *error,
        ]
        lib.posthf_cholesky_jk_v1.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            _DOUBLE,
            ct.c_uint,
            _DOUBLE,
            ct.c_size_t,
            *error,
        ]
        lib.posthf_cholesky_metrics_v1.argtypes = [
            ct.c_void_p,
            ct.POINTER(_Metrics),
            *error,
        ]
        with _PREPARATION_LOCK:
            self.call(
                "posthf_cholesky_create_v1",
                device_id,
                target["compute_capability_major"],
                target["compute_capability_minor"],
                nbf,
                capacity,
                arena,
                ct.byref(self._handle),
            )

    def call(self, name, *args):
        error = ct.create_string_buffer(2048)
        if getattr(self._library, name)(*args, error, len(error)):
            raise RuntimeError(error.value.decode())

    def project(self, column, pivot, rank):
        result = np.empty_like(column)
        self.call(
            "posthf_cholesky_project_v1",
            self._handle,
            rank,
            pivot,
            pointer(column),
            pointer(result),
            column.size,
        )
        return result

    def commit(self, column, rank):
        self.call(
            "posthf_cholesky_commit_v1",
            self._handle,
            rank,
            pointer(column),
            column.size,
        )

    def jk(self, density, rank):
        spins = density.shape[0]
        n = density.shape[1]
        result = np.empty((spins + 1, n, n))
        self.call(
            "posthf_cholesky_jk_v1",
            self._handle,
            rank,
            pointer(density),
            spins,
            pointer(result),
            result.size,
        )
        return result

    def metrics(self):
        value = _Metrics()
        self.call("posthf_cholesky_metrics_v1", self._handle, ct.byref(value))
        return {name: getattr(value, name) for name, _ in value._fields_}

    def close(self):
        with _PREPARATION_LOCK:
            if self._handle:
                self._library.posthf_cholesky_destroy_v1(self._handle)
                self._handle = ct.c_void_p()


class CudaIncrementalCholesky(IncrementalCholesky):
    """Keep actual factors on CPU and CUDA while extending the same prefix.

    CPU owns the source, deterministic pivot selection, PSD/roundoff checks and
    final commit decision. CUDA subtracts the retained prefix and evaluates J/K.
    Only raw/new column vectors cross the device boundary during refinement;
    old factor columns are never reuploaded or recomputed. The host mirror is
    explicitly budgeted and also supplies CPU MO transformations. This mixed
    execution is experimental and does not claim a resident GPU integral source.
    """

    def __init__(
        self,
        columns,
        artifact,
        *,
        rank_capacity,
        pair_tile=256,
        export_rank_tile=1,
        budget=None,
        device_id=0,
    ):
        self._native = None
        self._failed = False
        request, arena = cuda_factor_request(
            columns.space.nbf, rank_capacity, device_id
        )
        super().__init__(
            columns,
            rank_capacity=rank_capacity,
            pair_tile=pair_tile,
            export_rank_tile=export_rank_tile,
            budget=budget,
            _execution_requests=(request,),
        )
        self.artifact = artifact
        try:
            self._native = _NativePrefix(
                artifact, columns.space.nbf, rank_capacity, device_id, arena
            )
            observed = self._native.metrics()
            if (
                observed["owned_device_bytes"] != arena
                or observed["provider_retained_bytes"] > PROVIDER_ALLOWANCE
            ):
                raise RuntimeError(
                    "CUDA observed allocation differs from the low-rank plan"
                )
        except BaseException:
            self.close()
            raise

    def _check(self):
        super()._check()
        if self._failed:
            raise RuntimeError(
                "CUDA factor commit failed; close and rebuild the provider"
            )

    def _project_column(self, column, pivot):
        return self._native.project(column, pivot, self.rank)

    def _commit_native_column(self, column):
        try:
            self._native.commit(column, self.rank)
        except BaseException:
            self._failed = True
            raise

    def _native_jk(self, density, resource_plan):
        """Called under the consumer/factor locks after density and budget checks."""
        started = time.perf_counter()
        n = self.space.nbf
        spins = 1 if density.ndim == 2 else 2
        before = self._native.metrics()
        result = self._native.jk(
            np.ascontiguousarray(density.reshape(spins, n, n)), self.rank
        )
        after = self._native.metrics()
        return LowRankJk(
            immutable(result[0]),
            immutable(result[1] if spins == 1 else result[1:]),
            self.hamiltonian_id,
            {
                "rank": self.rank,
                "spins": spins,
                "consumer_backend": "cuda",
                "wall_seconds": time.perf_counter() - started,
                "resource_plan": resource_plan.to_dict(),
                "derivatives": "unsupported",
                "native_metrics": after,
                "section_ms": {
                    name: after[name] - before[name]
                    for name in after
                    if name.endswith("_ms") and name != "device_ms"
                },
                "timing_scope": "section events enabled; device_ms not measured; outputs explicitly copied to host",
            },
        )

    def diagnostics(self):
        with self._lock:
            result = super().diagnostics()
            result.update(
                execution="CPU source/pivot/PSD; CUDA resident Schur projection",
                native_metrics=self._native.metrics(),
                native_artifact=self.artifact.metadata["key"],
            )
            return result

    def close(self):
        with self._lock:
            if self._native is not None:
                self._native.close()
            super().close()

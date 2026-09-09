"""Density-response J/K backends used by matrix-free response operators."""

from __future__ import annotations

import ctypes as ct
import hashlib
import time

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.df import MetricFactor
from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_posthf.sources import _DOUBLE, pointer


def _checked_density(density, nbf):
    value = np.asarray(density)
    if value.shape != (nbf, nbf):
        raise ValueError(f"density response must have shape ({nbf},{nbf})")
    if np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError("density response must be finite real FP64")
    if np.max(np.abs(value - value.T)) > 1e-12:
        raise ValueError("density response must be symmetric")
    return 0.5 * (value + value.T)


class DenseAOResponseBackend:
    """Independent dense AO J/K oracle for tiny systems and unit tests.

    The full AO ERI tensor is intentional here.  This backend exists only as
    an independent algebraic oracle for the streaming native backend and is
    never used by a production response path.
    """

    def __init__(self, eri, *, maximum_n=12):
        value = np.asarray(eri)
        if (
            value.ndim != 4
            or value.shape != (value.shape[0],) * 4
            or value.shape[0] > min(maximum_n, 12)
            or np.iscomplexobj(value)
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                "dense response backend requires a tiny finite real AO ERI"
            )
        self.eri = immutable(value)
        self.nbf = value.shape[0]
        self.identity = canonical_hash(
            {
                "backend": "dense-ao-jk-oracle",
                "nbf": self.nbf,
                "eri_sha256": hashlib.sha256(
                    self.eri.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
            }
        )
        self.statistics = {
            "actions": 0,
            "tiles": 0,
            "seconds": 0.0,
            "peak_bytes": 3 * self.eri.size * 8,
        }

    def coulomb_exchange(self, density):
        """Return ``J[D]`` and ``K[D]`` for one AO density response."""
        started = time.perf_counter()
        d = _checked_density(density, self.nbf)
        coulomb = np.einsum("pqrs,rs->pq", self.eri, d, optimize=False)
        exchange = np.einsum("prqs,rs->pq", self.eri, d, optimize=False)
        self.statistics["actions"] += 1
        self.statistics["tiles"] += 1
        self.statistics["seconds"] += time.perf_counter() - started
        return immutable(coulomb), immutable(exchange)


class NativeJKBackend:
    """Matrix-free AO J/K actions through the existing native shell-tile source.

    Each action streams the four-center ERI source and forms only the two
    O(N^2) response matrices.  No AO N^4 tensor or MO integral cache is
    retained.  The source identity, representation and tile policy are part of
    the backend identity so retained Krylov state cannot cross Hamiltonians.
    """

    def __init__(
        self,
        source,
        *,
        axis_tile=2,
        budget_bytes=64 << 20,
        backend="cpu",
    ):
        if backend != "cpu":
            raise NotImplementedError(
                "native direct J/K response is currently CPU-streamed; "
                "use a validated GPU DF backend when available"
            )
        if (
            type(axis_tile) is not int
            or axis_tile < 1
            or type(budget_bytes) is not int
            or budget_bytes < 1
        ):
            raise ValueError("axis_tile and budget_bytes must be positive integers")
        source._check_open()
        self.source = source
        self.axis_tile = axis_tile
        self.budget_bytes = budget_bytes
        self.backend = backend
        self.nbf = source.nbf
        self.hamiltonian_id = "conventional-unscreened"
        self.identity = canonical_hash(
            {
                "backend": "native-cpu-shell-tile-jk",
                "source_identity": source.identity,
                "representation": source.representation,
                "axis_tile": axis_tile,
                "budget_bytes": budget_bytes,
                "hamiltonian": self.hamiltonian_id,
            }
        )
        self.statistics = {
            "actions": 0,
            "tiles": 0,
            "seconds": 0.0,
            "source_seconds": 0.0,
            "peak_bytes": 0,
        }

    def coulomb_exchange(self, density):
        """Stream ``(pq|rs) D_rs`` and ``(pr|qs) D_rs`` without an AO N^4 cache."""
        started = time.perf_counter()
        d = _checked_density(density, self.nbf)
        coulomb = np.zeros((self.nbf, self.nbf))
        exchange = np.zeros((self.nbf, self.nbf))
        tiles = 0
        source_seconds = 0.0
        for request in self.source.requests(
            "four_center_eri",
            axis_tile=self.axis_tile,
            budget_bytes=min(self.budget_bytes, 16 * self.axis_tile**4),
        ):
            source_started = time.perf_counter()
            tile = self.source.tile(request)
            begin = self.source.global_offsets(request)
            source_seconds += time.perf_counter() - source_started
            u, v, w, x = (slice(b, b + n) for b, n in zip(begin, tile.shape))
            coulomb[u, v] += np.einsum("uvwx,wx->uv", tile, d[w, x], optimize=False)
            exchange[u, w] += np.einsum("uvwx,vx->uw", tile, d[v, x], optimize=False)
            tiles += 1
        coulomb = 0.5 * (coulomb + coulomb.T)
        exchange = 0.5 * (exchange + exchange.T)
        self.statistics["actions"] += 1
        self.statistics["tiles"] += tiles
        self.statistics["seconds"] += time.perf_counter() - started
        self.statistics["source_seconds"] += source_seconds
        self.statistics["peak_bytes"] = max(
            self.statistics["peak_bytes"],
            3 * self.nbf * self.nbf * 8 + 4 * self.axis_tile**4 * 8,
        )
        return immutable(coulomb), immutable(exchange)

    def validate_reference(self, reference):
        """Reject a same-sized but scientifically unrelated reference."""
        if reference.geometry_hash != self.source.geometry_hash:
            raise ValueError("native backend/reference geometry mismatch")
        if reference.basis_hash != self.source.basis_hash:
            raise ValueError("native backend/reference basis mismatch")
        if reference.representation != self.source.representation:
            raise ValueError("native backend/reference representation mismatch")
        if reference.hamiltonian_id != self.hamiltonian_id:
            raise ValueError("native backend/reference Hamiltonian mismatch")
        return self


class CudaDFJKBackend:
    """Streamed native CUDA density-fitting J/K response backend.

    The backend owns one prepared CUDA DF source/plan pair.  It is intentionally
    separate from the CPU direct backend because it implements a different
    Hamiltonian identity (the declared DF metric), and it fails closed when no
    auxiliary basis or real CUDA build is available.
    """

    def __init__(
        self,
        source,
        *,
        device_id=0,
        metric_threshold=1e-10,
        hamiltonian_id=None,
        metric=None,
    ):
        if type(device_id) is not int or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")
        if not np.isfinite(metric_threshold) or not 0 < metric_threshold < 1:
            raise ValueError("metric_threshold must be finite and in (0,1)")
        if not hasattr(source, "_handle") or not getattr(source, "naux", 0):
            raise ValueError("CUDA DF response requires a source with auxiliary basis")
        self.source = source
        self.device_id = device_id
        self.metric_threshold = float(metric_threshold)
        source._check_open()
        # A label alone cannot certify the Hamiltonian of the native plan.
        # Reuse the exporter's checked metric when supplied, or construct one
        # from this source before any CUDA plan/device setup occurs.
        if metric is None:
            metric = MetricFactor.from_source(
                source, relative_threshold=self.metric_threshold
            )
        if (
            not isinstance(metric, MetricFactor)
            or metric.geometry_hash != source.geometry_hash
            or metric.auxiliary_hash != source.auxiliary_hash
            or metric.inverse_square_root.shape != (source.naux, source.naux)
            or metric.relative_threshold != self.metric_threshold
        ):
            raise ValueError("CUDA DF backend/source/metric mismatch")
        if hamiltonian_id is not None and hamiltonian_id != metric.hamiltonian_id:
            raise ValueError("CUDA DF backend/metric Hamiltonian identity mismatch")
        self.hamiltonian_id = metric.hamiltonian_id
        self._handle = ct.c_void_p()
        library = source._library
        library.vibeqc_posthf_rhf_jk_plan_create_v1.argtypes = [
            ct.c_void_p,
            ct.c_int,
            ct.c_double,
            ct.POINTER(ct.c_void_p),
            _DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        library.vibeqc_posthf_rhf_jk_plan_execute_v1.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            ct.c_size_t,
            _DOUBLE,
            _DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        library.vibeqc_posthf_rhf_jk_plan_destroy_v1.argtypes = [ct.c_void_p]
        library.vibeqc_posthf_rhf_jk_plan_destroy_v1.restype = None
        diagnostics = np.empty(6)
        source._call(
            "vibeqc_posthf_rhf_jk_plan_create_v1",
            source._handle,
            device_id,
            self.metric_threshold,
            ct.byref(self._handle),
            pointer(diagnostics),
        )
        self.nbf = int(diagnostics[0])
        self.naux = int(diagnostics[1])
        if self.nbf != source.nbf or self.naux != source.naux:
            self.close()
            raise RuntimeError(
                "CUDA DF response plan dimensions do not match the source"
            )
        self.device_resident_bytes = int(diagnostics[2])
        self.peak_device_bytes = int(diagnostics[3])
        self.host_resident_bytes = int(diagnostics[4])
        self.identity = canonical_hash(
            {
                "backend": "cuda-df-streamed-jk",
                "source_identity": source.identity,
                "device_id": device_id,
                "metric_threshold": self.metric_threshold,
                "hamiltonian_id": self.hamiltonian_id,
                "nbf": self.nbf,
                "naux": self.naux,
            }
        )
        self.statistics = {
            "actions": 0,
            "seconds": 0.0,
            "peak_bytes": self.peak_device_bytes,
            "device_resident_bytes": self.device_resident_bytes,
            "host_resident_bytes": self.host_resident_bytes,
        }

    def coulomb_exchange(self, density):
        """Apply one device-resident RHF DF J/K contraction."""
        started = time.perf_counter()
        d = _checked_density(density, self.nbf)
        density_buffer = np.ascontiguousarray(d)
        coulomb = np.empty((self.nbf, self.nbf))
        exchange = np.empty((self.nbf, self.nbf))
        self.source._call(
            "vibeqc_posthf_rhf_jk_plan_execute_v1",
            self._handle,
            pointer(density_buffer),
            d.size,
            pointer(coulomb),
            pointer(exchange),
        )
        self.statistics["actions"] += 1
        self.statistics["seconds"] += time.perf_counter() - started
        return immutable(coulomb), immutable(exchange)

    def validate_reference(self, reference):
        """Require the exact source geometry/basis and declared DF Hamiltonian."""
        if reference.geometry_hash != self.source.geometry_hash:
            raise ValueError("CUDA DF backend/reference geometry mismatch")
        if reference.basis_hash != self.source.basis_hash:
            raise ValueError("CUDA DF backend/reference basis mismatch")
        if reference.representation != self.source.representation:
            raise ValueError("CUDA DF backend/reference representation mismatch")
        if not self.hamiltonian_id or reference.hamiltonian_id != self.hamiltonian_id:
            raise ValueError("CUDA DF backend/reference Hamiltonian identity mismatch")
        return self

    def close(self):
        """Release the prepared plan; repeated close is safe."""
        if self._handle:
            self.source._library.vibeqc_posthf_rhf_jk_plan_destroy_v1(self._handle)
            self._handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()

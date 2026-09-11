"""Optional DF Hamiltonian with explicit metric policy and bounded B tiles.

This provider performs transformations on the CPU. A generated CUDA raw source
may be supplied, in which case its tiles are explicitly staged to the host.
That combination is not advertised as a resident GPU DF transformation.
"""

from __future__ import annotations

import ctypes as ct
import threading
import time
from dataclasses import dataclass
from math import prod

import numpy as np
from vibeqc.profiles import canonical_hash

from .providers import BlockResult
from .reference import immutable
from .sources import _DOUBLE, CPU_SOURCE_SCRATCH, pointer


@dataclass(frozen=True)
class MetricFactor:
    """Square symmetric M**(-1/2), retaining zeroed dependent directions.

    Eigenvalues <= relative_threshold*lambda_max are discarded exactly as in
    VibeQC's SCF metric factor. The auxiliary axis stays uncompressed; rank is
    diagnostic. No Cholesky gauge or conventional ERI Hamiltonian is implied.
    """

    inverse_square_root: np.ndarray
    auxiliary_hash: str
    geometry_hash: str
    relative_threshold: float
    rank: int
    absolute_threshold: float
    condition_number: float
    metric_hash: str
    identity: str
    setup_seconds: float
    host_peak_bytes: int
    device_bytes: int
    convention: str = "square-symmetric-thresholded-inverse-square-root"

    @classmethod
    def from_source(cls, source, *, relative_threshold=1e-10, budget_bytes=128 << 20):
        if not 0 < relative_threshold < 1 or not source.naux:
            raise ValueError(
                "DF requires an auxiliary basis and relative cutoff in (0,1)"
            )
        n = source.naux
        # Input, copied native matrix, eigenvectors/work, output and immutable
        # publication coexist. The existing factor does dense O(naux**2) work.
        peak = CPU_SOURCE_SCRATCH + source.numeric_bytes + 8 * 12 * n * n
        if peak > budget_bytes:
            raise MemoryError(f"metric setup requires {peak} numeric bytes")
        start = time.perf_counter()
        metric = source._read("coulomb_metric", (0, 0), (n, n))
        root = np.empty_like(metric)
        diagnostics = np.empty(3)
        library = source._library
        library.vibeqc_posthf_metric_v1.argtypes = [
            _DOUBLE,
            ct.c_size_t,
            ct.c_double,
            _DOUBLE,
            _DOUBLE,
            ct.c_char_p,
            ct.c_size_t,
        ]
        source._call(
            "vibeqc_posthf_metric_v1",
            pointer(metric),
            n,
            relative_threshold,
            pointer(root),
            pointer(diagnostics),
        )
        root = immutable(root)
        if int(diagnostics[0]) == 0:
            raise ValueError("DF metric has no retained directions")
        metric_hash = canonical_hash(metric.tolist())
        identity = canonical_hash(
            {
                "auxiliary": source.auxiliary_hash,
                "geometry": source.geometry_hash,
                "metric": metric_hash,
                "threshold": relative_threshold,
                "rank": int(diagnostics[0]),
                "convention": "square-symmetric-thresholded-inverse-square-root",
            }
        )
        return cls(
            root,
            source.auxiliary_hash,
            source.geometry_hash,
            relative_threshold,
            int(diagnostics[0]),
            float(diagnostics[1]),
            float(diagnostics[2]),
            metric_hash,
            identity,
            time.perf_counter() - start,
            peak - getattr(source, "source_device_bytes", 0),
            getattr(source, "source_device_bytes", 0),
        )

    @property
    def hamiltonian_id(self):
        return "density-fitting:" + self.identity


class DFProvider:
    """CPU staged B[P,p,q] tiles and retained chemists' MO blocks.

    A full four-index AO tensor is never constructed. ``three_index`` returns
    only its explicit auxiliary range. ``get`` streams those ranges and caches
    the final requested MO block, so residual reuse does not re-transform it.
    The cache pins blocks and rejects capacity exhaustion without eviction.
    """

    def __init__(
        self,
        snapshot,
        source,
        metric,
        *,
        budget_bytes=256 << 20,
        axis_tile=2,
        auxiliary_tile=3,
    ):
        if (
            type(budget_bytes) is not int
            or budget_bytes < 1
            or type(axis_tile) is not int
            or axis_tile < 1
            or type(auxiliary_tile) is not int
            or auxiliary_tile < 1
        ):
            raise ValueError("positive integer DF budgets/tiles required")
        if (
            snapshot.geometry_hash != source.geometry_hash
            or metric.geometry_hash != source.geometry_hash
            or snapshot.nmo != source.nbf
            or snapshot.representation != source.representation
            or snapshot.screening_tolerance != 0
            or snapshot.basis_hash != source.basis_hash
            or snapshot.hamiltonian_id != metric.hamiltonian_id
            or metric.auxiliary_hash != source.auxiliary_hash
            or metric.inverse_square_root.shape != (source.naux, source.naux)
        ):
            raise ValueError("DF snapshot/source/metric Hamiltonian mismatch")
        self.snapshot = snapshot
        self.source = source
        self.metric = metric
        self.budget_bytes = budget_bytes
        self.axis_tile = axis_tile
        self.auxiliary_tile = auxiliary_tile
        self._cache = {}
        self._retained = 0
        self._lock = threading.RLock()
        self._closed = False
        self.statistics = {
            "transformations": 0,
            "hits": 0,
            "peak_bytes": 0,
            "source_tiles": 0,
            "source_seconds": 0.0,
            "transformation_seconds": 0.0,
        }

    def _check(self):
        if self._closed:
            raise RuntimeError("DF provider is closed")
        self.source._check_open()

    def _capacity(self, p, q, count):
        n = self.source.nbf
        tile = min(
            self.axis_tile,
            max((*self.source.shell_sizes, *self.source.auxiliary_sizes)),
        )
        stage = max(
            tile**3,
            tile * tile * len(p),
            tile * len(p) * len(q),
            count * len(p) * len(q),
        )
        return (
            CPU_SOURCE_SCRATCH
            + self.source.numeric_bytes
            + self.snapshot.numeric_bytes
            + self.metric.inverse_square_root.nbytes
            + 8 * (6 * stage + 3 * count * len(p) * len(q) + 2 * n * (len(p) + len(q)))
        )

    def three_index(self, p, q, *, auxiliary_begin=0, auxiliary_count=None):
        """Return B[Q,p,q]=sum_(mu,nu,P) C[mu,p]C[nu,q]A[mu,nu,P]M^-1/2[P,Q]."""
        with self._lock:
            self._check()
            p, q = tuple(p), tuple(q)
            if (
                any(
                    type(i) is not int or not 0 <= i < self.snapshot.nmo
                    for i in (*p, *q)
                )
                or len(set(p)) != len(p)
                or len(set(q)) != len(q)
            ):
                raise ValueError("invalid DF MO columns")
            count = (
                self.source.naux - auxiliary_begin
                if auxiliary_count is None
                else auxiliary_count
            )
            if (
                type(auxiliary_begin) is not int
                or type(count) is not int
                or auxiliary_begin < 0
                or count < 0
                or auxiliary_begin > self.source.naux
                or count > self.source.naux - auxiliary_begin
            ):
                raise ValueError("invalid DF auxiliary tile")
            peak = self._retained + self._capacity(p, q, count)
            if peak > self.budget_bytes:
                raise MemoryError(f"DF tile requires {peak} numeric bytes")
            self.statistics["peak_bytes"] = max(self.statistics["peak_bytes"], peak)
            result = np.zeros((len(p), len(q), count))
            if result.size:
                cp = np.ascontiguousarray(self.snapshot.coefficients[:, p])
                cq = np.ascontiguousarray(self.snapshot.coefficients[:, q])
                for request in self.source.requests(
                    "three_center_eri",
                    axis_tile=self.axis_tile,
                    budget_bytes=16 * self.axis_tile**3,
                ):
                    started = time.perf_counter()
                    tile = self.source.tile(request)
                    self.statistics["source_seconds"] += time.perf_counter() - started
                    started = time.perf_counter()
                    u, v, P = self.source.global_offsets(request)
                    nu, nv, nP = tile.shape
                    first = (tile.reshape(nu, -1).T @ cp[u : u + nu]).reshape(
                        nv, nP, len(p)
                    )
                    second = (first.reshape(nv, -1).T @ cq[v : v + nv]).reshape(
                        nP, len(p) * len(q)
                    )
                    result += (
                        second.T
                        @ self.metric.inverse_square_root[
                            P : P + nP, auxiliary_begin : auxiliary_begin + count
                        ]
                    ).reshape(result.shape)
                    self.statistics["source_tiles"] += 1
                    self.statistics["transformation_seconds"] += (
                        time.perf_counter() - started
                    )
            return immutable(result.transpose(2, 0, 1))

    def get(self, block):
        """Reconstruct only requested g[p,q,r,s]=sum_Q B[Q,p,q]B[Q,r,s]."""
        with self._lock:
            self._check()
            block.validate(self.snapshot)
            start = time.perf_counter()
            if block.slots in self._cache:
                self.statistics["hits"] += 1
                previous = self._cache[block.slots]
                return BlockResult(
                    block,
                    previous.values,
                    previous.reference_id,
                    previous.hamiltonian_id,
                    {
                        **previous.diagnostics,
                        "cache_hit": True,
                        "reuse_seconds": time.perf_counter() - start,
                    },
                )
            p, q, r, s = block.slots
            count = min(self.auxiliary_tile, self.source.naux)
            # The reconstruction output and both B tiles remain live while a
            # new B tile is formed. Reserve them before any source evaluation.
            persistent = 8 * (
                3 * prod(block.shape) + count * (len(p) * len(q) + len(r) * len(s))
            )
            peak = (
                self._retained
                + persistent
                + max(self._capacity(p, q, count), self._capacity(r, s, count))
            )
            if peak > self.budget_bytes:
                raise MemoryError(f"DF MO block requires {peak} numeric bytes")
            self.statistics["peak_bytes"] = max(self.statistics["peak_bytes"], peak)
            values = np.zeros(block.shape)
            self._retained += persistent
            try:
                if values.size:
                    for begin in range(0, self.source.naux, self.auxiliary_tile):
                        size = min(self.auxiliary_tile, self.source.naux - begin)
                        left = self.three_index(
                            p, q, auxiliary_begin=begin, auxiliary_count=size
                        )
                        right = self.three_index(
                            r, s, auxiliary_begin=begin, auxiliary_count=size
                        )
                        started = time.perf_counter()
                        values += (
                            left.reshape(size, -1).T @ right.reshape(size, -1)
                        ).reshape(block.shape)
                        self.statistics["transformation_seconds"] += (
                            time.perf_counter() - started
                        )
                values = immutable(values)
            finally:
                self._retained -= persistent
            result = BlockResult(
                block,
                values,
                self.snapshot.identity,
                self.snapshot.hamiltonian_id,
                {
                    "backend": "cpu-reference-df-staged-fp64",
                    "source_backend": self.source.backend,
                    "source_host_staging": self.source.backend.startswith("cuda"),
                    "cache_hit": False,
                    "endpoint_seconds": time.perf_counter() - start,
                    "peak_bytes": peak,
                    "host_peak_bytes": peak
                    - getattr(self.source, "source_device_bytes", 0),
                    "device_peak_bytes": getattr(self.source, "source_device_bytes", 0),
                    "output_placement": "host",
                    "metric_identity": self.metric.identity,
                    "metric_rank": self.metric.rank,
                    "metric_threshold": self.metric.relative_threshold,
                    "inverse_convention": self.metric.convention,
                    "metric_setup_seconds": self.metric.setup_seconds,
                    "cumulative_source_seconds": self.statistics["source_seconds"],
                    "cumulative_transformation_seconds": self.statistics[
                        "transformation_seconds"
                    ],
                    "cuda_source": self.source.source_metrics()
                    if hasattr(self.source, "source_metrics")
                    else None,
                },
            )
            self._cache[block.slots] = result
            self._retained += values.nbytes
            self.statistics["transformations"] += 1
            return result

    def close(self):
        self._cache.clear()
        self._retained = 0
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

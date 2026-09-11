"""Bounded conventional integral transforms with explicit retained-block caches."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np

from .plan import plan_block
from .reference import immutable


@dataclass(frozen=True)
class BlockResult:
    """An immutable host MO block or an owned, explicitly downloadable device block."""

    block: object
    values: object
    reference_id: str
    hamiltonian_id: str
    diagnostics: dict

    def to_host(self):
        """Download only on explicit request; CPU outputs already own host data."""
        if isinstance(self.values, np.ndarray):
            return self.values
        return self.values.to_host()


def transform_tile(tile, coefficients):
    """Four cyclic staged contractions; no Kronecker product or AO N**4 tensor.

    Each step contracts the first remaining AO axis and appends its MO axis:
    uvwx -> vwxp -> wxpq -> xpqr -> pqrs. Matmul performs each complete stage.
    """
    stage = tile
    for c in coefficients:
        stage = (stage.reshape(stage.shape[0], -1).T @ c).reshape(
            (*stage.shape[1:], c.shape[1])
        )
    return stage


class ConventionalProvider:
    """Transform once per explicit block/reference and retain under a budget.

    The default ``pin`` policy never silently evicts a CC residual dependency.
    ``lru`` is opt-in and its evictions/recomputations are diagnosed. Callers
    must release externally retained results themselves; those arrays/resources
    stay charged here until ``clear``/``close`` and must not be used afterward
    for CUDA. A new reference always requires a new provider.
    """

    def __init__(
        self,
        snapshot,
        source,
        *,
        budget_bytes=256 << 20,
        axis_tile=2,
        backend="cpu",
        cache_policy="pin",
        cuda_artifact=None,
        device_id=0,
    ):
        if type(budget_bytes) is not int or budget_bytes < 1:
            raise ValueError("budget_bytes must be positive")
        if cache_policy not in ("pin", "lru"):
            raise ValueError("cache policy must be pin or lru")
        if (
            snapshot.hamiltonian_id != "conventional-unscreened"
            or snapshot.screening_tolerance != 0
        ):
            raise ValueError(
                "conventional source implements only the unscreened Hamiltonian"
            )
        self.snapshot = snapshot
        self.source = source
        self.budget_bytes = budget_bytes
        self.axis_tile = axis_tile
        self.backend = backend
        self.cache_policy = cache_policy
        self.cuda_artifact = cuda_artifact
        self.device_id = device_id
        self._cache = OrderedDict()
        self._retained = 0
        self._lock = threading.RLock()
        self._closed = False
        self.statistics = {
            "transformations": 0,
            "hits": 0,
            "evictions": 0,
            "source_tiles": 0,
            "peak_bytes": 0,
        }
        self._source_identity = source.identity

    def plan(self, block):
        """Dry-run a complete requested output before reading any AO integral."""
        return plan_block(
            self.snapshot,
            self.source,
            block,
            axis_tile=self.axis_tile,
            backend=self.backend,
        )

    def get(self, block):
        """Return g[p,q,r,s] for explicit global MO columns; reuse exact identity."""
        with self._lock:
            if self._closed:
                raise RuntimeError("integral provider is closed")
            if self.source.identity != self._source_identity:
                raise ValueError("integral source changed; construct a fresh provider")
            self.source._check_open()
            started = time.perf_counter()
            plan = self.plan(block)
            if plan.peak_bytes > self.budget_bytes:
                raise MemoryError(
                    f"MO block needs {plan.peak_bytes} numeric bytes, budget is {self.budget_bytes}"
                )
            key = (self.snapshot.identity, self.source.identity, block.slots)
            if key in self._cache:
                self._cache.move_to_end(key)
                self.statistics["hits"] += 1
                result = self._cache[key][0]
                return BlockResult(
                    result.block,
                    result.values,
                    result.reference_id,
                    result.hamiltonian_id,
                    {
                        **result.diagnostics,
                        "cache_hit": True,
                        "reuse_seconds": time.perf_counter() - started,
                    },
                )
            while (
                self._cache
                and self._retained + plan.peak_bytes > self.budget_bytes
                and self.cache_policy == "lru"
            ):
                _, (old, cost) = self._cache.popitem(last=False)
                if hasattr(old.values, "close"):
                    old.values.close()
                self._retained -= cost
                self.statistics["evictions"] += 1
            peak = self._retained + plan.peak_bytes
            if peak > self.budget_bytes:
                raise MemoryError(
                    f"MO block needs {peak} numeric bytes, budget is {self.budget_bytes}"
                )
            self.statistics["peak_bytes"] = max(self.statistics["peak_bytes"], peak)
            setup = time.perf_counter()
            coefficients = [
                np.ascontiguousarray(self.snapshot.coefficients[:, slot])
                for slot in block.slots
            ]
            if plan.output_elements == 0:
                values = immutable(np.zeros(block.shape))
                retained = values.nbytes
                engine = None
            elif self.backend == "cuda":
                from .cuda import CudaTransform

                if self.cuda_artifact is None:
                    raise ValueError(
                        "CUDA provider requires an explicitly compiled artifact"
                    )
                engine = CudaTransform(
                    self.cuda_artifact, plan, coefficients, device_id=self.device_id
                )
                values = engine
                retained = plan.device_bytes
            else:
                engine = None
                values = np.zeros(block.shape)
                retained = values.nbytes
            setup_seconds = time.perf_counter() - setup
            source_seconds = transform_seconds = 0.0
            tiles = 0
            try:
                if plan.output_elements:
                    for request in self.source.requests(
                        "four_center_eri",
                        axis_tile=self.axis_tile,
                        budget_bytes=16 * int(np.prod(plan.tile_shape)),
                    ):
                        source_begin = time.perf_counter()
                        tile = self.source.tile(request)
                        offsets = self.source.global_offsets(request)
                        source_seconds += time.perf_counter() - source_begin
                        transform_begin = time.perf_counter()
                        if engine is None:
                            panels = [
                                c[b : b + n]
                                for c, b, n in zip(coefficients, offsets, tile.shape)
                            ]
                            values += transform_tile(tile, panels)
                        else:
                            engine.add(tile, offsets)
                        transform_seconds += time.perf_counter() - transform_begin
                        tiles += 1
                    if engine is None:
                        values = immutable(values)
            except BaseException:
                if engine is not None:
                    engine.close()
                raise
            diagnostics = {
                "backend": self.backend + "-staged-fp64",
                "source_backend": self.source.backend,
                "source_host_staging": self.backend == "cuda",
                "output_placement": "device" if engine else "host",
                "ordinary_stream": engine is not None,
                "cache_hit": False,
                "setup_seconds": setup_seconds,
                "source_seconds": source_seconds,
                "transformation_seconds": transform_seconds,
                "endpoint_seconds": time.perf_counter() - started,
                "source_tiles": tiles,
                "plan": asdict(plan),
                "peak_bytes_including_cached_blocks": peak,
                "reference_id": self.snapshot.identity,
                "hamiltonian_id": self.snapshot.hamiltonian_id,
            }
            if engine is not None:
                diagnostics["cuda"] = engine.metrics()
            result = BlockResult(
                block,
                values,
                self.snapshot.identity,
                self.snapshot.hamiltonian_id,
                diagnostics,
            )
            self._cache[key] = (result, retained)
            self._retained += retained
            self.statistics["transformations"] += 1
            self.statistics["source_tiles"] += tiles
            return result

    def clear(self):
        """Release retained outputs; CUDA block objects become explicitly closed."""
        with self._lock:
            for value, _ in self._cache.values():
                if hasattr(value.values, "close"):
                    value.values.close()
            self._cache.clear()
            self._retained = 0

    def close(self):
        """Close this provider, without taking ownership of the source lifetime."""
        self.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

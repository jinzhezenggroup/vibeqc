"""Bounded CUDA (T) triples tile execution via the #420 resident owner (slice B).

No CUDA call, compilation or GPU allocation occurs at import time.  Callers
must supply a verified :class:`CudaCompilerAdapter` and a writable cache
directory.  All real-device validation runs on qz (inspire); local machines
without CUDA raise an explicit error at preparation time.

The host extracts tile-sized sub-blocks from the full-system tensors before
each tile upload, so the device plan only ever sees tile-sized intermediates.
Full ``nocc^3 * nvir^3`` T3 or denominator tensors are never allocated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .triples_tiles import (
    TriplesTileEnumerator,
    build_tile_triples_program,
    tile_triples_energy_masked,
)


@dataclass(frozen=True)
class TriplesTileConfig:
    """Immutable shape, budget and scheduling decisions for a single tile plan."""

    nocc: int
    nvir: int
    vir_chunk_size: int
    max_bytes: int
    device: int = 0

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in (self.nocc, self.nvir)):
            raise ValueError("triples require nonempty occupied and virtual spaces")
        if type(self.vir_chunk_size) is not int or self.vir_chunk_size < 1:
            raise ValueError("vir_chunk_size must be positive")
        if type(self.max_bytes) is not int or self.max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if type(self.device) is not int or self.device < 0:
            raise ValueError("device must be a nonnegative visible CUDA ordinal")


@dataclass
class CudaTriplesResult:
    """Complete GPU (T) execution including per-tile diagnostics."""

    et: float
    per_tile: list[float]
    per_tile_masked_cpu: list[float]
    tile_count: int
    vir_chunk_size: int
    nocc: int
    nvir: int
    peak_device_bytes: int
    plan_identity: str = ""
    artifact_key: str = ""
    runtime_device: dict | None = None
    timing: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


class CudaTriplesTiles:
    """Compile one tile plan and evaluate every tile through the resident owner.

    Parameters
    ----------
    config : TriplesTileConfig
        Immutable shape, budget and device selection.
    compiler : CudaCompilerAdapter
        Verified NVCC wrapper (must match the target GPU architecture).
    cache : Path
        Writable compilation cache directory.
    """

    def __init__(self, config, compiler, cache):
        from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
        from vibeqc_compiler.tensor.cuda_resident import (
            PreparedResident,
            compile_resident,
        )

        self.config = config
        # Build the tile program once — shape is fixed.
        tile_program = build_tile_triples_program(
            config.nocc, config.nvir
        )
        schedule = TensorSchedule()
        self.plan = plan_cuda(
            tile_program,
            compiler.target,
            max_bytes=config.max_bytes,
            schedule=schedule,
        )
        self.compiler = compiler
        self.cache = cache
        self._compile_resident = compile_resident
        self._PreparedResident = PreparedResident
        self._tile_program = tile_program
        self._artifact = None
        self._resident = None

    @property
    def peak_bytes(self):
        return self.plan.peak_bytes

    def _prepare(self):
        """Lazily compile and prepare the resident owner (once)."""
        if self._resident is not None:
            return
        artifact = self._compile_resident(self.plan, self.compiler, self.cache)
        self._artifact = artifact
        self._resident = self._PreparedResident(
            self.plan, artifact, device=self.config.device
        )

    def run_tiles(self, arrays, *, profile=False):
        """Evaluate all tiles and return :class:`CudaTriplesResult`.

        Parameters
        ----------
        arrays : dict
            The eight full-system float64 ndarrays keyed by
            ``ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v``.
        profile : bool
            Forwarded to ``resident.run(profile=...)``.

        Returns
        -------
        CudaTriplesResult
            Total E_T, per-tile GPU scalars, per-tile masked CPU reference,
            timing breakdown and provenance.
        """
        self._prepare()
        nocc = self.config.nocc
        nvir = self.config.nvir
        chunk = self.config.vir_chunk_size

        ovvv = arrays["ovvv"]
        ovoo = arrays["ovoo"]
        ovov = arrays["ovov"]
        fov = arrays["fov"]
        t1 = arrays["t1"]
        t2 = arrays["t2"]
        eps_o = arrays["eps_o"]
        eps_v = arrays["eps_v"]

        # Validate shapes once
        expected = {
            "ovvv": (nocc, nvir, nvir, nvir),
            "ovoo": (nocc, nvir, nocc, nocc),
            "ovov": (nocc, nvir, nocc, nvir),
            "fov": (nocc, nvir),
            "t1": (nocc, nvir),
            "t2": (nocc, nocc, nvir, nvir),
            "eps_o": (nocc,),
            "eps_v": (nvir,),
        }
        for name, shape in expected.items():
            arr = arrays[name]
            if not isinstance(arr, np.ndarray) or arr.dtype != np.float64:
                raise ValueError(f"{name} must be a float64 ndarray")
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} != expected {shape}")

        enumerator = TriplesTileEnumerator(nocc, nvir, vir_chunk_size=chunk)
        tiles = list(enumerator)

        timing = {
            "extract_s": 0.0,
            "upload_s": 0.0,
            "run_s": 0.0,
            "download_s": 0.0,
            "tile_count": len(tiles),
        }
        per_tile = []
        per_tile_masked_cpu = []
        et = 0.0

        t0_total = time.perf_counter()
        resident = self._resident

        for tile in tiles:
            # Extract sub-blocks: the tile covers virtual [0, a_end) but
            # the triangular loops only process a in [a_start, a_end).
            # We upload the first a_end columns of all virtual-indexed tensors.
            a_end = tile.a_end
            t0 = time.perf_counter()
            sub_feeds = {
                "ovvv": np.ascontiguousarray(ovvv[:, :a_end, :a_end, :a_end]),
                "ovoo": np.ascontiguousarray(ovoo[:, :a_end, :, :]),
                "ovov": np.ascontiguousarray(ovov[:, :a_end, :, :a_end]),
                "fov": np.ascontiguousarray(fov[:, :a_end]),
                "t1": np.ascontiguousarray(t1[:, :a_end]),
                "t2": np.ascontiguousarray(t2[:, :, :a_end, :a_end]),
                "eps_o": eps_o,
                "eps_v": eps_v[:a_end],
            }
            # Use the tile program for this a-range
            tile_prog = build_tile_triples_program(
                nocc, a_end, vir_chunk=(tile.a_start, tile.a_end)
            )
            timing["extract_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            resident.upload(sub_feeds)
            timing["upload_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            leases, metrics = resident.run(profile=profile)
            timing["run_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            et_tile = float(
                resident.download(leases["triples_energy"])[()]
            )
            timing["download_s"] += time.perf_counter() - t0

            per_tile.append(et_tile)
            et += et_tile

            # CPU masked reference for per-tile comparison
            cpu_masked = tile_triples_energy_masked(
                tile, nocc,
                ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v,
            )
            per_tile_masked_cpu.append(cpu_masked)

        timing["total_s"] = time.perf_counter() - t0_total

        return CudaTriplesResult(
            et=et,
            per_tile=per_tile,
            per_tile_masked_cpu=per_tile_masked_cpu,
            tile_count=len(tiles),
            vir_chunk_size=chunk,
            nocc=nocc,
            nvir=nvir,
            peak_device_bytes=self.plan.peak_bytes,
            plan_identity=self.plan.identity,
            artifact_key=self._artifact.metadata.get("key", "")
            if self._artifact
            else "",
            runtime_device=getattr(resident, "device", None),
            timing=timing,
            provenance={
                "schema": "vibeqc.ccsd-t.cuda-tile/1",
                "plan_identity": self.plan.identity,
                "tile_shapes": [(tile.a_end - tile.a_start, nocc) for tile in tiles],
            },
        )

    def close(self):
        if self._resident is not None:
            self._resident.close()
            self._resident = None
        self._artifact = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Standalone utility: run tiles on CPU only (no CUDA imports at all)
# ---------------------------------------------------------------------------


def cpu_triples_tiles(nocc, nvir, arrays, *, vir_chunk_size=1, denominator_threshold=1e-10):
    """CPU-only tile loop; mirror of :meth:`CudaTriplesTiles.run_tiles`.

    Returns the same :class:`CudaTriplesResult` shape (``peak_device_bytes=0``,
    no plan identity).  Useful for testing parity on systems without a GPU.
    """
    ovvv = arrays["ovvv"]
    ovoo = arrays["ovoo"]
    ovov = arrays["ovov"]
    fov = arrays["fov"]
    t1 = arrays["t1"]
    t2 = arrays["t2"]
    eps_o = arrays["eps_o"]
    eps_v = arrays["eps_v"]

    enumerator = TriplesTileEnumerator(nocc, nvir, vir_chunk_size=vir_chunk_size)
    tiles = list(enumerator)
    per_tile = []
    per_tile_masked_cpu = []
    et = 0.0

    for tile in tiles:
        from .triples_tiles import tile_triples_energy, tile_triples_energy_masked

        et_tile = tile_triples_energy(
            tile, nocc,
            ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v,
            denominator_threshold=denominator_threshold,
        )
        per_tile.append(et_tile)
        et += et_tile

        cpu_masked = tile_triples_energy_masked(
            tile, nocc,
            ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v,
        )
        per_tile_masked_cpu.append(cpu_masked)

    return CudaTriplesResult(
        et=et,
        per_tile=per_tile,
        per_tile_masked_cpu=per_tile_masked_cpu,
        tile_count=len(tiles),
        vir_chunk_size=vir_chunk_size,
        nocc=nocc,
        nvir=nvir,
        peak_device_bytes=0,
    )
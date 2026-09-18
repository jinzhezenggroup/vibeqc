"""Bounded CUDA (T) triples tile execution via the #420 resident owner (slice B).

No CUDA call, compilation or GPU allocation occurs at import time.  Callers
must supply a verified :class:`CudaCompilerAdapter` and a writable cache
directory.  Real-device validation requires CUDA; machines without CUDA
raise an explicit error at preparation time.

The host extracts exact-shape sub-blocks from the full-system tensors before
each tile upload.  Label axes are prefix-bounded by the tile's ``a_end``,
while the W1 virtual summation axes remain full ``nvir``.  One resident plan
is compiled per tile range/shape; compilation is transparently cached to disk
via ``compile_resident``.  Each resident is created per tile and closed before
the next tile, so no full ``nocc³ × nvir³`` T3 or denominator tensor is ever
allocated.

The shared input guards from ``triples._validate`` and
``triples._check_denominators`` run once before any GPU work, matching the
CPU/TensorIR entry-point contract.  CPU oracle comparison is **opt-in**
(``oracle=True``) and **never runs** on the default production path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .triples import _check_denominators, _validate
from .triples_tiles import (
    TriplesTileEnumerator,
    _tile_input_feeds,
    build_tile_triples_program,
)


@dataclass(frozen=True)
class TriplesTileConfig:
    """Immutable shape, budget and scheduling decisions for a tile session."""

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
    """Complete GPU (T) execution including per-tile diagnostics.

    ``per_tile_masked_cpu`` is populated only when ``oracle=True`` was passed
    to :meth:`CudaTriplesTiles.run_tiles`.
    """

    et: float
    per_tile: list[float]
    tile_count: int
    vir_chunk_size: int
    nocc: int
    nvir: int
    peak_device_bytes: int
    per_tile_masked_cpu: list[float] | None = None
    peak_bytes_per_tile: list[int] = field(default_factory=list)
    plan_identity: str = ""
    artifact_keys: list[str] = field(default_factory=list)
    runtime_device: dict | None = None
    timing: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


class CudaTriplesTiles:
    """Compile per-tile plans and evaluate each through its own resident owner.

    One :class:`PreparedResident` is created per tile (with the exact
    sub-block virtual dimension and triangular range), used for one
    upload/run/download cycle, then closed.  Compilation is transparently
    cached to disk so repeated tile shapes reuse the cached artifact.

    The shared input/denominator guards from :mod:`triples` run once before
    any GPU work.  The CPU masked-oracle comparison is opt-in (``oracle=True``)
    and never runs on the default production path.

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
        from vibeqc_compiler.tensor.cuda_plan import plan_cuda
        from vibeqc_compiler.tensor.cuda_resident import (
            PreparedResident,
            compile_resident,
        )

        self.config = config
        self.compiler = compiler
        self.cache = cache
        self._plan_cuda = plan_cuda
        self._compile_resident = compile_resident
        self._PreparedResident = PreparedResident

    def run_tiles(self, arrays, *, oracle=False, profile=False):
        """Evaluate all tiles and return :class:`CudaTriplesResult`.

        The shared finite/canonical/denominator guards run once before any
        GPU uploads or compilation, matching the CPU/TensorIR entry-point
        contract.  CPU masked-oracle comparison is opt-in via ``oracle=True``
        and **never runs on the default production path**.

        Parameters
        ----------
        arrays : dict
            The eight full-system float64 ndarrays keyed by
            ``ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v``.
        oracle : bool
            When True, also compute the per-tile masked CPU reference
            (``CudaTriplesResult.per_tile_masked_cpu``).  Off by default;
            intended for validation/debug use only.
        profile : bool
            Forwarded to ``resident.run(profile=...)``.

        Returns
        -------
        CudaTriplesResult
            Total E_T, per-tile GPU scalars, (optionally) per-tile masked
            CPU reference, timing breakdown and provenance.
        """
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

        # --- shared input guard (#150 contract) ---
        _validate(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
        _check_denominators(eps_o, eps_v, 1e-10)

        # The tile program separates bounded label axes (a,b,c) from the
        # full virtual summation axis f.  In the original tensors the W1
        # f-axis is ovvv axis 2 and t2 axis 3; those stay full while label
        # axes are prefix-bounded to a_end.  This makes every upload shape
        # exactly match its TensorSpec without truncating the contraction.

        enumerator = TriplesTileEnumerator(nocc, nvir, vir_chunk_size=chunk)
        tiles = list(enumerator)

        timing = {
            "extract_s": 0.0,
            "compile_s": 0.0,
            "upload_s": 0.0,
            "run_s": 0.0,
            "download_s": 0.0,
            "oracle_s": 0.0,
            "tile_count": len(tiles),
        }
        per_tile = []
        per_tile_masked_cpu = [] if oracle else None
        peak_bytes_per_tile = []
        artifact_keys = []
        runtime_device = None
        et = 0.0

        t0_total = time.perf_counter()

        for tile in tiles:
            # 1. Extract exact-shape feeds: label axes are bounded by a_end,
            #    while the W1 f-summation axes remain full nvir.
            t0 = time.perf_counter()
            sub_feeds = _tile_input_feeds(arrays, tile.a_end)
            timing["extract_s"] += time.perf_counter() - t0

            # 2. Build the exact per-tile program, plan it, and compile
            #    (compilation is transparently cached to disk).  Distinct
            #    TensorIR spaces keep label extents at a_end and f at nvir.
            t0 = time.perf_counter()
            tile_prog = build_tile_triples_program(
                nocc, nvir, vir_chunk=(tile.a_start, tile.a_end)
            )
            plan = self._plan_cuda(
                tile_prog,
                self.compiler.target,
                max_bytes=self.config.max_bytes,
            )
            artifact = self._compile_resident(plan, self.compiler, self.cache)
            timing["compile_s"] += time.perf_counter() - t0
            peak_bytes_per_tile.append(plan.peak_bytes)
            artifact_keys.append(artifact.metadata.get("key", ""))

            # 3. Create a per-tile resident owner, upload exact-shape feeds,
            #    run, download the scalar, then close it.
            with self._PreparedResident(
                plan, artifact, device=self.config.device
            ) as resident:
                t0 = time.perf_counter()
                resident.upload(sub_feeds)
                timing["upload_s"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                leases, _metrics = resident.run(profile=profile)
                timing["run_s"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                et_tile = float(resident.download(leases["triples_energy"])[()])
                timing["download_s"] += time.perf_counter() - t0
                if runtime_device is None:
                    # Captured once from the first resident's device probe.
                    runtime_device = resident.device

            per_tile.append(et_tile)
            et += et_tile

        timing["total_s"] = time.perf_counter() - t0_total

        # Opt-in CPU masked reference (validation/debug only)
        if oracle:
            t0 = time.perf_counter()
            from .triples_tiles import tile_triples_energy_masked

            for tile in tiles:
                cpu_masked = tile_triples_energy_masked(
                    tile,
                    nocc,
                    ovvv,
                    ovoo,
                    ovov,
                    fov,
                    t1,
                    t2,
                    eps_o,
                    eps_v,
                )
                per_tile_masked_cpu.append(cpu_masked)
            timing["oracle_s"] = time.perf_counter() - t0

        return CudaTriplesResult(
            et=et,
            per_tile=per_tile,
            per_tile_masked_cpu=per_tile_masked_cpu,
            tile_count=len(tiles),
            vir_chunk_size=chunk,
            nocc=nocc,
            nvir=nvir,
            peak_device_bytes=max(peak_bytes_per_tile) if peak_bytes_per_tile else 0,
            peak_bytes_per_tile=peak_bytes_per_tile,
            artifact_keys=artifact_keys,
            runtime_device=runtime_device,
            timing=timing,
            provenance={
                "schema": "vibeqc.ccsd-t.cuda-tile/1",
                "tile_shapes": [
                    {
                        "a_start": tile.a_start,
                        "a_end": tile.a_end,
                        "plan_peak_bytes": pb,
                    }
                    for tile, pb in zip(tiles, peak_bytes_per_tile)
                ],
                "oracle_enabled": oracle,
            },
        )

    def close(self):
        pass  # no persistent resources; each tile creates and closes its own

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Standalone utility: run tiles on CPU only (no CUDA imports at all)
# ---------------------------------------------------------------------------


def cpu_triples_tiles(
    nocc, nvir, arrays, *, vir_chunk_size=1, denominator_threshold=1e-10
):
    """CPU-only tile loop; mirror of :meth:`CudaTriplesTiles.run_tiles`.

    Returns the same :class:`CudaTriplesResult` shape (``peak_device_bytes=0``,
    no plan identity).  Useful for testing parity on systems without a GPU.

    The masked CPU oracle is never computed by this function; use
    :func:`tools.vibeqc_cc.triples_tiles.tile_triples_energy_masked`
    directly for per-tile validation.
    """
    ovvv = arrays["ovvv"]
    ovoo = arrays["ovoo"]
    ovov = arrays["ovov"]
    fov = arrays["fov"]
    t1 = arrays["t1"]
    t2 = arrays["t2"]
    eps_o = arrays["eps_o"]
    eps_v = arrays["eps_v"]

    # Shared input guards
    _validate(nocc, nvir, ovvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v)
    _check_denominators(eps_o, eps_v, denominator_threshold)

    from .triples_tiles import tile_triples_energy

    enumerator = TriplesTileEnumerator(nocc, nvir, vir_chunk_size=vir_chunk_size)
    tiles = list(enumerator)
    per_tile = []
    et = 0.0

    for tile in tiles:
        et_tile = tile_triples_energy(
            tile,
            nocc,
            ovvv,
            ovoo,
            ovov,
            fov,
            t1,
            t2,
            eps_o,
            eps_v,
            denominator_threshold=denominator_threshold,
        )
        per_tile.append(et_tile)
        et += et_tile

    return CudaTriplesResult(
        et=et,
        per_tile=per_tile,
        tile_count=len(tiles),
        vir_chunk_size=vir_chunk_size,
        nocc=nocc,
        nvir=nvir,
        peak_device_bytes=0,
    )

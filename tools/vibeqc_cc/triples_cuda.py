"""Bounded CUDA (T) triples tile execution via the #420 resident owner (slice B).

No CUDA call, compilation or GPU allocation occurs at import time.  Callers
must supply a verified :class:`CudaCompilerAdapter` and a writable cache
directory.  Real-device validation requires CUDA; machines without CUDA
raise an explicit error at preparation time.

The #783 path builds one fixed-capacity runtime-indexed TensorIR graph for the
largest triangular virtual tile in the session. Full scientific inputs are
uploaded once; each logical tile updates only int64 coordinate maps plus its
active/degeneracy controls. One compiled resident owner therefore executes all
tile ranges without rebuilding the scientific graph or materializing a full
``nocc³ × nvir³`` T3/denominator tensor.

The shared input guards from ``triples._validate`` and
``triples._check_denominators`` run once before any GPU work, matching the
CPU/TensorIR entry-point contract.  CPU oracle comparison is **opt-in**
(``oracle=True``) and **never runs** on the default production path.
"""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass, field

from .triples import _check_denominators, _validate
from .triples_tiles import (
    TriplesTileEnumerator,
    build_runtime_tile_triples_program,
    runtime_tile_capacity,
    runtime_tile_control_batches,
    runtime_tile_static_feeds,
)


@dataclass(frozen=True)
class TriplesTileConfig:
    """Immutable shape, budget and scheduling decisions for a tile session."""

    nocc: int
    nvir: int
    vir_chunk_size: int
    max_bytes: int
    device: int = 0

    def __post_init__(self) -> None:
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
    """Execute runtime-indexed tiles through one fixed-capacity resident owner.

    One :class:`PreparedResident` is compiled/prepared per run and reused for
    every logical tile. Scientific arrays stay resident while only bounded
    coordinate/control vectors change between runs.

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

    def __init__(
        self, config: typing.Any, compiler: typing.Any, cache: typing.Any
    ) -> None:
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

    def plan_runtime_domain(self) -> typing.Any:
        """Select the largest simple bounded lane capacity that fits the budget.

        The scientific graph remains identical apart from the q-domain extent.
        Capacity is reduced only after the planner proves the larger candidate
        infeasible; unrelated planning errors propagate unchanged.
        """

        nocc = self.config.nocc
        nvir = self.config.nvir
        chunk = self.config.vir_chunk_size
        capacity = runtime_tile_capacity(nocc, nvir, chunk)
        attempts = []
        while True:
            program = build_runtime_tile_triples_program(nocc, nvir, capacity=capacity)
            try:
                plan = self._plan_cuda(
                    program,
                    self.compiler.target,
                    max_bytes=self.config.max_bytes,
                )
            except ValueError as error:
                if "infeasible tensor byte budget" not in str(error) or capacity == 1:
                    raise
                attempts.append(
                    {
                        "capacity": capacity,
                        "status": "infeasible",
                        "reason": str(error),
                    }
                )
                capacity = max(1, capacity // 2)
                continue
            attempts.append(
                {
                    "capacity": capacity,
                    "status": "selected",
                    "peak_bytes": plan.peak_bytes,
                }
            )
            return capacity, plan, tuple(attempts)

    def run_tiles(
        self,
        arrays: typing.Any,
        *,
        oracle: typing.Any = False,
        profile: typing.Any = False,
    ) -> typing.Any:
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

        # #783: one runtime-indexed graph is selected under the caller's
        # byte budget. Logical a-tiles may be split into smaller runtime batches
        # without changing that graph, artifact, or resident owner.
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
        et = 0.0
        t0_total = time.perf_counter()

        t0 = time.perf_counter()
        capacity, plan, capacity_attempts = self.plan_runtime_domain()
        artifact = self._compile_resident(plan, self.compiler, self.cache)
        timing["compile_s"] += time.perf_counter() - t0
        runtime_batch_count = sum(
            (tile.ntriples + capacity - 1) // capacity for tile in tiles
        )
        timing["runtime_batch_count"] = runtime_batch_count
        peak_bytes_per_tile = [plan.peak_bytes] * len(tiles)
        artifact_keys = [artifact.metadata.get("key", "")]

        t0 = time.perf_counter()
        static_feeds = runtime_tile_static_feeds(arrays)
        timing["extract_s"] += time.perf_counter() - t0

        with self._PreparedResident(
            plan, artifact, device=self.config.device
        ) as resident:
            runtime_device = resident.device
            t0 = time.perf_counter()
            resident.upload(static_feeds)
            timing["upload_s"] += time.perf_counter() - t0

            for tile in tiles:
                et_tile = 0.0
                controls_iterator = iter(runtime_tile_control_batches(tile, capacity))
                while True:
                    t0 = time.perf_counter()
                    try:
                        controls = next(controls_iterator)
                    except StopIteration:
                        break
                    timing["extract_s"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    resident.upload(controls)
                    timing["upload_s"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    leases, _metrics = resident.run(profile=profile)
                    timing["run_s"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    et_batch = float(resident.download(leases["triples_energy"])[()])
                    timing["download_s"] += time.perf_counter() - t0
                    et_tile += et_batch
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
            plan_identity=plan.identity,
            artifact_keys=artifact_keys,
            runtime_device=runtime_device,
            timing=timing,
            provenance={
                "schema": "vibeqc.ccsd-t.cuda-runtime-domain/2",
                "runtime_domain_capacity": capacity,
                "runtime_batch_count": runtime_batch_count,
                "capacity_selection": list(capacity_attempts),
                "artifact_reuse": "one compiled plan and resident owner across all logical tiles/runtime batches",
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

    def close(self) -> typing.Any:
        pass  # run_tiles owns one bounded resident for the duration of each run

    def __enter__(self) -> typing.Any:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Standalone utility: run tiles on CPU only (no CUDA imports at all)
# ---------------------------------------------------------------------------


def cpu_triples_tiles(
    nocc: typing.Any,
    nvir: typing.Any,
    arrays: typing.Any,
    *,
    vir_chunk_size: typing.Any = 1,
    denominator_threshold: typing.Any = 1e-10,
) -> typing.Any:
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

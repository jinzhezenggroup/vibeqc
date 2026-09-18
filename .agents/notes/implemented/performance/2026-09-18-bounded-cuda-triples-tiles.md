# Decision: Bounded CUDA (T) triples via per-tile shape-matched resident plans

Status: implemented
Date: 2026-09-18

## Problem

Issue #150 slice B requires the non-iterative (T) correction to run on CUDA
through the #420 resident TensorIR owner, without ever allocating a full
`nocc³ × nvir³` T3 or denominator tensor, while keeping the triangular
`a>=b>=c` degeneracy contract of the slice A reference (`triples_energy`).

Two tensions shaped the design:

1. TensorIR `gather`/`slice` positions are compile-time constants, so a single
   compiled plan cannot sweep tile ranges at runtime.
2. `PreparedResident.upload` requires each feed shape to exactly match the
   compiled `TensorSpec`, so prefix-shaped partial tiles cannot share one
   full-`nvir` plan (the failure mode caught in PR #448 review round 1).

## Decision

**Per-tile shape-matched plans, one resident owner per tile, NVCC-cached
artifacts.**

- `tools/vibeqc_cc/triples_tiles.py` provides `TriplesTileEnumerator`
  (a-chunked triangular virtual domain; occupied space is never chunked),
  `TileSpec`, the per-tile CPU reference `tile_triples_energy`, the
  masked-domain oracle `tile_triples_energy_masked`, and the tile-sized
  TensorIR lowering `build_tile_triples_program(nocc, a_end, vir_chunk=(a_start, a_end))`.
- `tools/vibeqc_cc/triples_cuda.py` `CudaTriplesTiles.run_tiles` loops tiles:
  for each tile it builds the exact `(a_start, a_end)` program, plans it under
  the caller's `max_bytes`, compiles (disk-cached so repeated shapes reuse
  artifacts), creates a `PreparedResident`, uploads exact-shape sub-block
  feeds, runs, downloads the scalar, and closes the resident. Device memory
  at any moment is bounded by the single-tile plan peak; the largest tensor
  on device is tile-sized.
- The shared guards `triples._validate` + `triples._check_denominators` run
  once before any GPU work (fail-closed for non-finite inputs, noncanonical
  or near-zero denominators — the #150 step-7 contract).
- The CPU masked oracle is **opt-in** (`oracle=True`): validation code
  (`tools/validate_cc_triples_tiles.py`) passes it; the default production
  path never computes CPU reference work.
- `CudaTriplesResult.runtime_device` is captured from the first resident's
  device probe (name/architecture), so evidence manifests prove which GPU
  executed the run.

## Rejected alternatives

- **One full-nvir plan with prefix feeds** — violates exact-shape upload;
  also recomputes the full a-range per tile. (PR #448 review round 1.)
- **Runtime-masked fixed-shape plan** (pad sub-blocks to `nvir`, mask a-range
  at runtime) — avoids per-shape recompilation but forces the full
  `nvir`-sized arena to be resident, breaking the bounded-memory contract at
  large `nvir`, and needs a runtime-indexable denominator build inside
  TensorIR that does not exist.
- **Single resident reused across tiles with upload-only swaps** — the goal
  statement preferred this ("具名输入上传一次"), but identical shapes across
  tiles only exist when `vir_chunk_size` divides `nvir` evenly; the
  per-tile-owner design keeps every tile exactly shape-matched at the cost of
  one create/destroy per tile (amortized by the on-disk compile cache).

## Invariants

- Never allocate a full `nocc³ × nvir³` T3 or full denominator tensor on
  device or host production paths.
- Tile TensorIR inventory is bit-identical to slice A (`W_TERMS`, `V_TERMS`,
  `R3`, `SLOW_TABLE`, `_degeneracy` imported, not re-derived).
- CPU reference and oracle (`tile_triples_energy{,_masked}`) remain callable
  with no CUDA imports (`cpu_triples_tiles` mirrors the GPU loop shape).
- `oracle=True` comparison code must never run on the default path.
- Guards (`_validate`, `_check_denominators`) must run before any device
  compilation or upload.

## Evidence

- `tests/python/test_cc_triples_tiles.py` — 14 tests: enumerator coverage
  (every `a>=b>=c` triple exactly once), tile-sum parity vs `triples_energy`
  (≤1e-10), masked-oracle parity, TensorIR roundtrip/differentiability,
  determinism, chunk-size independence, endpoint regression
  (h2/he ≈ 0; h2o/nh3/ch4 vs pinned PySCF 2.14.0 truth ≤1e-9).
- `tools/validate_cc_triples_tiles.py` — qz GPU validation driver: per
  molecule, per chunk size (`nvir` and `nvir//2`), per budget (256/512 MiB):
  per-tile GPU-vs-masked-CPU ≤1e-10, total ≤1e-9, two-run bitwise determinism,
  peak-bytes records, explicit infeasible-budget failure.
- Real-device run manifest: retained under
  `benchmarks/results/cc-triples-b/` when executed on qz (GPU, architecture
  and artifact keys recorded in `runtime_device` / `artifact_keys`).

## Consequences

- Compile latency: each unique tile shape costs one NVCC compile (cached
  across runs). For tiny endpoint systems (nvir ≤ 4) this is seconds per
  shape; larger systems amortize via cache reuse.
- Data movement: sub-block extraction on host is O(nocc × a_end³) per tile
  plus one H2D upload per tensor per tile; acceptable for slice B's target
  sizes, and bounded by design.

## Revisit when

- A runtime tile-range primitive lands in TensorIR (compile-time-constant
  `gather` constraint removed) — then a single shape-padded plan with runtime
  masking could replace per-shape compilation.
- Systems grow to where per-tile owner create/destroy dominates (profile
  `timing["compile_s"]`/`["upload_s"]` in the manifest); then bucket tiles by
  shape and reuse residents within a bucket.
- The occupancy chunking question (i>=j>=k triangular occupied tiles)
  resurfaces — currently occupied space is never chunked; if nocc³ becomes
  the binding memory term, chunking i similarly is the natural extension.

## References

- Issue: #150 (slice B); tracking contract #137
- PR: #448
- Prior art: #346 (slice A reference), #420 (resident TensorIR owner)
- Docs: `docs/rccsd_t.md`

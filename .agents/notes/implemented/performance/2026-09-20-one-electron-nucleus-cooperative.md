# Decision: add an explicit generated nucleus-cooperative one-electron derivative schedule

Status: implemented candidate; default promotion pending endpoint qualification
Date: 2026-09-20

## Problem

The generated `shell_warp` derivative schedule promoted by #543 assigns warp lanes to AO components. On the 96-atom / 768-AO water-32mer endpoint this leaves low-component shell pairs under-filled and makes every active lane walk all nuclear centers serially. The pinned #670 evidence measured the generated one-electron kernel at about 1.071 s versus 0.128 s for the retained native cooperative route, with an approximately 0.945 s complete E+F gap in the clean factorial comparison.

## Decision

Add schedule `nucleus_cooperative` (`3`) while keeping the compiler-generated S/T/V derivative DAG as the only generated scientific implementation. One warp owns one triangular AO pair. Lane zero evaluates the nucleus-independent S/T response once; attraction centers are processed in warp-sized tiles with one nuclear center owned by each lane. A `PairGeometry` instance is materialized once per primitive pair and nucleus tile in warp-private shared memory and reused by all lanes. Center-A/B attraction responses are reduced across the warp; each valid nuclear-center lane accumulates its own C response before one bounded global update per coordinate.

The current production default remains `shell_warp`. Schedule 3 is an explicit candidate until clean Release/sm_120 endpoint and holdout evidence justify a selector or promotion.

## Rejected alternatives

Copying the retained handwritten Hermite recurrence was rejected because it would recreate duplicate scientific ownership. Globally changing the default to AO-thread or nucleus-cooperative from the 768-AO point alone was rejected because #670 requires shape-independent selection evidence and known small/resident counterexamples. Materializing `Natom x NAO^2` derivative records was rejected because the derivative contract requires bounded scratch. Sharing Boys values across nuclear lanes was rejected because the Boys argument depends on the nuclear center; only center-independent pair geometry is shared in this first generated cooperative candidate.

## Invariants

- Generated `overlap_kinetic_gradient` and `attraction_gradient` remain the mathematical source of S/T/V derivatives.
- RHF/UHF density and Pulay weights, external sign, nuclear charge, Cartesian/spherical expansion, and failed-item masking are unchanged.
- S/T are not repeated per nucleus tile.
- Tail nuclear lanes participate in every full-warp collective; only center-specific arithmetic and writes are masked.
- Coincident A/B/C physical atoms are handled by additive accumulation rather than post-hoc force correction.
- No global derivative tensor or runtime JIT/NVCC is introduced.
- Generated CUDA errors remain errors; there is no silent fallback to retained native science.

## Evidence

- CUDA 12.9 / sm_120 dev build compiles the new schedule and relinks successfully.
- Compiler selector contract test passes; compiler structure check reports 247 modules and zero dependency errors.
- Compiler/CPU one-electron derivative and selector tests: 57 passed.
- `ruff` and `git diff --check` pass for the touched test/source set.
- Slurm RTX 5090 correctness gate: 25 passed, 4 deselected in 624.81 s. This includes Cartesian/spherical arbitrary nonsymmetric S/T/V weights across schedules 0/1/2/3, RHF/UHF complete forces for Direct and DF at batch 1/3, reused-plan selector changes, independent PySCF force checks, and failed-neighbor isolation.
- Clean Release/sm_120 endpoint timing remains a promotion gate; this commit does not change the production default.

## Consequences

The runtime now exposes a generated schedule that restores the nuclear-center parallel dimension without changing the scientific DAG. Schedule 3 uses the existing triangular AO-pair metadata and a fixed per-warp shared `PairGeometry`; production SCF does not allocate a new `Natom x NAO^2` structure. The retained native cooperative route remains available as an explicit performance/reference comparator while qualification is incomplete.

## Revisit when

Promote or select this schedule only after the #670 matrix passes: independent raw/arbitrary-weight validation, changed geometry and failed-neighbor coverage, clean 384/768 and small/holdout endpoint measurements, plus the existing retained-native counterexamples. If the generated candidate still trails native materially, extend the compiler ABI to expose more pair-invariant attraction intermediates rather than copying handwritten recurrence code.

## References

- #670 generated nucleus-cooperative derivative schedule
- #669 768-AO regression isolation
- #543 generated derivative default promotion
- #357 one-electron derivative retirement boundary
- `docs/one_electron_derivatives.md`

---
Agent: ChatGPT
Model: GPT-5.6 Sol

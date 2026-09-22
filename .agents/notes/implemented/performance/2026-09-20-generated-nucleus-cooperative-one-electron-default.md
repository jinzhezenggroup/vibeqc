# Decision: Make generated nucleus-cooperative one-electron derivatives the default

Status: implemented
Date: 2026-09-20

## Problem

The generated `shell_warp` one-electron derivative schedule kept compiler ownership but
regressed large-system complete RHF E+F performance because low-component AO pairs leave
most warp lanes idle while each active lane serially traverses nuclei. The retained native
cooperative implementation showed that distributing nuclei across lanes removes this gap.

## Decision

Use generated schedule 3, `nucleus_cooperative`, as the default when
`VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING` is unset. Keep the generated mathematical DAG
as the only production scientific implementation. Preserve explicit `shell_warp`,
`thread`, `serial`, and retained `reference` selections for diagnostics and bounded
fallback/control comparisons.

A shape threshold is deliberately not introduced: clean holdouts from 29 through 768 AO,
including Cartesian high-angular-momentum, UHF, and batch-3 cases, showed no material
regression from the cooperative mapping.

## Rejected alternatives

- Keep `shell_warp` as the default: rejected because the 768-AO endpoint loses about
  0.97 s and the 384/192-AO endpoints also regress materially.
- Select by an exact AO threshold: rejected because the measured domain shows the same
  direction across small and large systems, and fingerprint-like thresholds would be
  fragile policy.

## Invariants

- Do not change S/T/V derivative mathematics, signs, density/Pulay weighting, or SCF
  convergence/final-state contracts.
- Keep `VIBEQC_ONE_ELECTRON_DERIVATIVES=reference` available as an independent control.
- Ordinary force calls remain AOT/offline-capable and must not invoke NVCC/JIT.
- Future replacement of the retained native control requires independent correctness and
  complete-endpoint evidence.

## Evidence

RTX 5090, Release/sm_120, head `8871a8194f3e8f6ae6466ec0dcbe5515816c2881`,
library SHA-256 `eadb9b06f99c3158e845270a27292047d441311615116a559c7f0f351caedf89`.
All measurements are fixed-warm-density clean E+F timings with profiler disabled.

- 768 AO batch 1: shell_warp 5.593508 s; generated cooperative 4.625324 s;
  native cooperative 4.654379 s.
- 384 AO batch 1: 1.356426 s; 1.219489 s; 1.228868 s.
- 192 AO batch 1: 0.448943 s; 0.421578 s; 0.424384 s.
- 192 AO batch 3: 1.115578 s; 1.060496 s; 1.066809 s.
- 29 AO ammonia RHF: 0.111079 s; 0.107578 s; 0.108231 s.
- sdf18 Cartesian high-angular-momentum RHF: 0.018685 s; 0.018133 s; 0.018225 s.
- OH def2-SVP spherical UHF: 0.108892 s; 0.108421 s; 0.108808 s.

The branch correctness matrix independently covers arbitrary weights, RHF/UHF, Direct/DF,
batch 1/3, reused-plan selector changes, PySCF force checks, and failed-neighbor isolation.

## Revisit when

Revisit selection only if a reproducible topology/device class shows a material complete
endpoint regression, not from an isolated kernel microbenchmark.

## References

Issue #670; PR #693; parent performance investigation #669.

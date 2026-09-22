# Decision: resident compiler-generated CUDA KS initial state

Status: implemented; standalone branch qualification in progress
Date: 2026-09-23

## Problem

Issue #1101: after GPU quadrature removes serial grid construction, water48
(384 spherical def2-SVP AOs) still spends 26.5435 s in CUDA PBE preparation.
Slurm 11349's existing reference-work observer attributes 14.0008 s to overlap
orthogonalization and 11.6858 s to the core seed. Their reference eigensolve
leaves take 13.9847 s and 11.6315 s. These are host component diagnostics of a
GPU endpoint, not CPU benchmark results.

## Decision

Reuse the ordinary-stream GPU eigen provider from #1091 twice during setup.
The compiler lowers the existing SCF weighted-projector TensorIR topology for
both the full symmetric overlap inverse square root and occupied core density;
the shared matrix-function emitter remains the inverse-square-root scalar owner.
Native code binds matrices, stream, existing matrix products and allocation.
Iteration scratch is borrowed before the first iteration. One separately charged
cold-density buffer per spin remains immutable for cold retries. No cold matrix
is downloaded and reuploaded. Shared explicit warm-input normalization and strict
validation remain input-boundary operations.

The common UHF frontier rotation preserves the existing seed policy, including
zero-beta and equal-spin branches. Orthogonalization rejects eigenvalues below
1e-10 (equality is allowed) and nonfinite values; the full X^T S X identity check
uses 1e-8. Failed spectrum/metric/solver status aborts preparation after draining
stack-backed scalar downloads. There is no CPU eigensolver fallback.

## Rejected alternatives

Replacing scalar CPU Jacobi with another CPU library retains the production
reference dependency and matrix staging. Duplicating overlap/density formulas in
DFT creates a second scientific owner. Overwriting mutable warm/proposal state
with the cold guess breaks retry semantics. A new BLAS handle for setup is not
needed to remove this cliff; retain common matrix kernels and charged provider
workspace. XC contraction lowering is tracked separately in #1102.

## Evidence

Slurm 11356 passes the allocated analytic projector, spectrum cutoff/nonfinite,
RHF/UKS/empty-beta and metric acceptance/rejection gates. The first broader run
exposed 7 energy-diagnostic test failures caused by implicit force dispatch;
the pre-change binary reproduces the same 7 failures (Slurm 11360). Energy
history/transport tests now explicitly request energy, preserving their original
scientific assertions; this is not qualification of the independent force path.

Slurm 11361, composed integration: analytic native gates plus 26 Python cases
pass (2 expected skips). Coverage includes independent PySCF/Libxc energy and
physical components, direct/DF semilocal RKS/UKS, cold/warm/geometry replay,
final-state handoff, failed-warm cold retry, exact resource budget and failed
preparation cleanup. Independent energy/component gates remain 1e-8 Eh, physical
residual 1e-9, and strict density/electron gates are unchanged.

Same allocated run, full preparation only on the original water geometries:

| Atoms / AOs | Preparation seconds | GPU initial-state scope seconds |
| --- | ---: | ---: |
| 48 / 384 | 0.951763178 | 0.106503747 |
| 96 / 768 | 4.028605681 | 0.126303528 |

Both were bounded by a 45 s process timeout. The observer reports no reference
solve leaves. These measurements include the GPU-grid and other preceding repair
overlays and are not standalone PR-head or converged SCF results. Full 48/96
SCF remains paused pending #1102, following the previous 120 s water48 timeout.
README benchmark publication remains paused.

## Consequences and revisit conditions

Retained device storage grows by spins*n*n doubles while retained host X/seed
storage is removed. Setup does two eigensolves, five common dense matrix
products (including metric validation) and two weighted projectors; no work is
repeated per SCF iteration. Revisit the projector/matrix schedule when complete
preparation evidence shows these operations matter, preserving numerical gates
and resource accounting. Explicit warm validation may still use reference
algebra by contract; do not confuse it with default cold preparation.

## References

- #1091 ordinary-stream eigensolver; #1101 setup cliff; #1102 XC contractions.
- `docs/developer/ks_diagnostics.md` and `docs/maintainer/performance_engineering.md`.

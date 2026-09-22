# Decision: batch full-rank DF panel charge contractions

Status: implemented
Date: 2026-09-22

## Problem

The source-backed full-rank DF force-response path selected
`contract_full_rank_panels()` with BLAS matrix products, but its charge
contraction still launched `charge_kernel()` once per auxiliary function.
For RHF there is one density term, so each launch activates one useful thread
which serially sums all `N_AO^2` elements.

A 384-AO Nsight capture on RTX 5090 attributed 4.114 s across 384
`charge_kernel` launches, 91.2% of all GPU kernel time in that endpoint.
At 768 AO, component tracing placed 36.183 s of the 39.063 s warm endpoint
inside force response. This was not the historical 77-panel raw-transfer
amplification: the trace used two response panels, zero tensor H2D, and the
expected 1,536 AO response products.

## Decision

When `blas_products` is enabled, contract every fitted auxiliary panel against
all density terms in one cuBLAS GEMM. Treat fitted values as column-major
`[ij,P]`, densities as `[ij,t]`, and write the resulting `[P,t]` panel
straight into the existing `potentials[t*naux + P]` layout by using
`ldc=naux`.

Keep the original scalar `charge_kernel` path when scalar response algebra is
explicitly selected. The change adds no allocation, changes no response tiling,
and does not alter owner or lifetime policy.

## Rejected alternatives

- Optimize the one-thread scalar kernel while continuing one launch per
  auxiliary function. The contraction is already a GEMM-shaped operation and
  the BLAS response mode explicitly asks for this lowering.
- Force occupied/resident response merely to avoid this path. Owner selection
  and response algebra are separate concerns; #890 addresses automatic owner
  routing.
- Relax energy, force, metric, or convergence thresholds. No numerical gate is
  changed.

## Invariants

- The public charge layout remains `potentials[t*naux + P]`.
- Scalar response algebra retains the scalar contraction for diagnostics and
  ablation.
- Existing bounded response memory, panel widths, source ownership, and
  synchronization semantics are unchanged.
- Independent energy and force gates remain unchanged.

## Evidence

Causal A/B used the same `b5e73236` source, CUDA 12.9, RTX 5090, spherical
def2-SVP for both orbital and auxiliary spaces, batch 1, and three warm repeats.
Only the panel charge contraction was changed.

| Equal-basis RHF endpoint | Before | Batched charge | Change |
| --- | ---: | ---: | ---: |
| 384 AO warm energy+force | 4.6855 s | 0.6169 s | 7.59x faster |
| 768 AO warm energy+force | 39.1395 s | 6.3350 s | 6.18x faster |

The candidate retained two VibeQC warm SCF updates at both sizes. Against the
independent GPU4PySCF calculation, maximum 384-AO errors were
`1.182e-11 Ha` and `1.317e-10 Ha/Bohr`; maximum 768-AO errors were
`2.956e-11 Ha` and `1.054e-10 Ha/Bohr`. All existing gates passed.

GPU4PySCF used one warm SCF update in these ordinary comparisons, so its
0.612 s / 2.269 s timings are context rather than iteration-matched speed
claims.

## Consequences

The 384-AO regression is largely removed without changing owner selection.
The remaining 768-AO gap is a separate target: automatic resident-owner
recovery (#890), remaining response transforms, and SCF work can now be
measured without the serial charge reduction dominating the profile.

Longer term, this contraction should become compiler-owned contraction IR so
the backend can choose generated reductions versus BLAS from shape and target
information. Runtime ownership and live-memory admission remain planner
responsibilities.

## Revisit when

Revisit the explicit BLAS dispatch after the compiler owns DF response
contractions and can prove an equivalent or faster lowering across small,
bounded, resident, and batched shapes.

## References

- #439
- #890
- #614
- `src/scf/cuda/df_response_weights.cu`
- `tests/python/test_df_shell_derivatives_cuda.py`

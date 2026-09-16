# Decision: retain SYRK after the bounded split Gram experiment

Status: rejected for automatic or permanent opt-in production use
Date: 2026-09-17

## Problem

The resident occupied RI-K Gram has a long FP64 reduction dimension. Splitting
it could expose more BLAS parallelism, but changed reduction order can affect
normal SCF convergence. Kernel speed alone does not qualify a complete endpoint.

## Decision

Keep the existing SYRK plus mirror policy. The bounded split4 candidate passed
fixed-U numerical/performance gates, integrated force tests and sanitizers, but
failed the predeclared complete-endpoint work gate. On the identical frozen
768-AO density, the first baseline force sample converged in three updates;
split4 needed five. Its complete force call took 2.979252 s against 2.559862 s
for the baseline. This single changed-work pair is retained as a failed gate,
not a paired performance estimate. Both results passed the unchanged independent
energy and complete-force gates.

The runner stopped immediately. There is no search for another starting density,
looser tolerance or extra split counts to rescue the campaign. Intrusive follow-up
records energy/force work and sampled resources separately. The production
candidate control and native changes are removed; its exact source and validation
patches remain reproducible in the evidence bundle. The post-#415 derivative
mapping and all default behavior are unchanged.

## Attempted design

The candidate divided physical column-major U over L = naux * rank into four
contiguous disjoint ranges, always retaining lda=L. Equal ranges used one
strided-batched GEMM; an unequal final range used a second GEMM. The existing
ascending FP64 exchange reduction consumed four complete K matrices.

Only already charged `exchange_intermediate` was borrowed: 18 MiB at 768 AOs,
with zero extra owned allocation. Raw A (including discarded metric directions)
and the exact-state final-U lease remained intact. Full GEMM computed both
triangles, roughly twice the leading Gram FLOPs of SYRK. Capacity/index/byte
checks preserved zero-rank, short, constrained and nonresident fallbacks.

Eight and sixteen splits offered no material improvement on real 768 inputs.
Repacking U would have changed the measured operation. Borrowing the raw-A buffer
would have invalidated forces. No new permanent handwritten reduction was needed.

## Evidence

Stage A tested twelve fixed-U fixtures, six candidates and seven interleaved
repeats per arm. Four real 768 inputs gave split4/SYRK ratios 0.87568--0.87582.
Independent tractable real-U and small references covered unequal tails, short
reductions, zero rank and both occupation conventions. Maximum split4 K error
was 2.512e-12, RMS 1.629e-14, symmetry error zero and exchange-energy error
4.048e-11. Initial memcheck/initcheck reported zero errors; that first memcheck
did not request full leak checking.

The integrated candidate passed two CPU policy tests, the native DF suite and
68 Python GPU tests covering complete forces, discarded metric directions,
policy changes, constrained storage, UHF and projection reuse. Native memcheck
with full leak checking and initcheck both reported zero errors. One initial
UHF trace-harness failure is retained: an already captured replay emits no new
per-product trace records. Its corrected test requires occupied execution
provenance and validated generations; RHF still requires eager final Gram events.

The endpoint protocol held merged #415's derivative mapping fixed and required
seven matched-work pairs, at least 1% complete 768 force saving, bootstrap upper
ratio below one, and unchanged 1e-9 Eh / 1e-8 Eh/Bohr gates. The 384 energy/force
regressions passed. The first 768 pair failed the work requirement, so magnitude
and stability gates were not evaluated and the remaining qualifying runs were
not continued. Follow-up diagnostic timings cannot replace missing clean pairs.

The independent intrusive follow-up reproduces three/five updates for both
energy and force endpoints. Each arm accepts one final Fock without correction
or rejection, and both force arms reuse the final projection once. J/K builds
and eigensolves (including seed factorization) increase from four to six.
Total leading Gram FLOPs increase from 290,287,779,840 to 869,730,877,440.
Both observable processes have the same sampled combined-policy device peak,
24,631,050,240 bytes including initialization; this does not establish a memory
improvement or separate per-policy peaks. Independent scientific gates still pass.

## Consequences and revisit conditions

A deterministic, numerically valid reduction change can add SCF iterations and
erase its local kernel gain. Preserve actual operator counts, frozen starting
states and normal convergence in later optimization decisions. Do not add the
Stage-A saving to #404's derivative gain or imply stock GPU4PySCF superiority.

Revisit only with a separately justified provider/representation change (for
example #409), a new predeclared protocol and complete scientific/work checks.
The archived candidate is an experiment, not a dormant production default.

## References

- Issue #412; complete external comparisons remain in #206.
- [Evidence and reconstruction](../../../benchmarks/results/issue412-split-gram/README.md).
- [Current occupied CUDA contract](../../../docs/df_occupied_cuda.md).

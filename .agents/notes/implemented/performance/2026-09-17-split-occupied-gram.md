# Decision: bounded split occupied Gram qualification

Status: implemented (explicit candidate; no automatic promotion)
Date: 2026-09-17

## Problem

The long reduction in resident occupied RI-K can limit BLAS scheduling. A faster
Gram kernel alone does not establish a complete SCF or force improvement, and
splitting it must preserve raw metric response data and the final-U lease.

## Decision

Add `split4` to the existing `VIBEQC_DF_RESIDENT_EXCHANGE` control. Freeze it in
the native plan and compare it when matching captured SCF policy. Keep `auto`
unchanged until complete endpoint qualification. The candidate borrows four
complete K matrices from the already charged `exchange_intermediate` buffer.
It retains the original U leading dimension, uses one strided-batched GEMM for
equal segments (plus a separate unequal tail), and reuses the existing ordered
FP64 exchange reduction. The borrowed capacity is 18 MiB at 768 AOs; owned
storage does not grow. Full GEMM computes both triangles and therefore roughly
doubles leading Gram FLOPs relative to SYRK.

## Rejected alternatives

Larger split counts offered no material further gain on the measured real 768
inputs and require more scratch. Repacking U would change the tested operation
and its traffic. Borrowing `exchange_contributions` would destroy raw A,
including discarded metric directions required for forces. A new allocation or
handwritten reduction is unnecessary. A new per-class environment switch would
duplicate the existing resident policy control.

## Invariants

- Validate BLAS/index/byte limits and four-matrix capacity before splitting.
- Preserve zero-rank, short-reduction, constrained and nonresident fallbacks.
- Keep raw A immutable and leave U valid for exact-state final projection reuse.
- Preserve factor generation checks and numerical recovery for stale factors.
- Order all scratch users on the same plan stream and retain #203 accounting.
- Distinguish actual K builds from four partial products inside one K build.
- Never add independent derivative and Gram percentages to claim a joint gain.

## Evidence and qualification

Stage A used twelve fixed-U fixtures and six candidates, including four real
768/768 rank-160 snapshots, four real 384 snapshots, an odd real-derived
97/193/rank-37 fixture, a small independent reference, short L and rank zero.
Seven interleaved repeats gave split4/SYRK ratios 0.87568--0.87582 on the four
real 768 inputs. Every candidate passed the independent numerical gates.
Maximum split4 K error was 2.512e-12, RMS 1.629e-14, symmetry error zero and
exchange-energy error 4.048e-11. Memcheck and initcheck reported no errors;
this initial memcheck did not request full leak checking. These are kernel
measurements, not endpoint performance evidence.

The endpoint protocol was declared before measurement against merged #415,
holding its derivative mapping fixed in both arms. It requires seven
interleaved paired complete force replays at 768, median paired ratio <= 0.99
and bootstrap 95% upper bound < 1 (100000 paired resamples, PCG64 seed 412).
All four energy endpoints and 96/192/384 force endpoints have a 3% regression
margin. Energy/force gates remain 1e-9 Eh / 1e-8 Eh/Bohr. Frozen checkpoint D,
SCF updates, J/K builds, eigensolves, final corrections and projection reuse
must match. Diagnostics follow clean timings; all builds finish before timing.

Automatic promotion, if qualified, must stay inside the existing singleton RHF
768/768/rank-160 resident RTX 5090 domain. Explicit selection on other shapes
qualifies correctness only. #206 still requires fresh stock GPU4PySCF timing.

Integrated-candidate validation passes the native DF suite, two CPU policy
tests and 68 Python GPU tests, including discarded metric directions, unequal
tails, policy changes, constrained storage, UHF and final-projection reuse.
Native memcheck with full leak checking and initcheck report zero errors.
An initial UHF trace assertion mistook missing per-product events during an
already captured replay for absent work. The corrected test requires executed
occupied provenance and validated generations; RHF still requires its eager
final Gram records. The failed assertion and passing follow-up are retained in
the [evidence bundle](../../../../benchmarks/results/issue412-split-gram/README.md).

## Revisit when

Complete endpoints qualify a bounded automatic domain, a provider implements
competitive split SYRK, or measured new shapes justify changing the split count.

## References

- Issue #412; ownership/resource contracts in #203.
- [Current occupied CUDA contract](../../../../docs/df_occupied_cuda.md).
- Policy arithmetic: `tests/python/test_df_split_gram_policy.py`.
- Executed split/tail/raw-response/lease checks: native density-fitting suite,
  `test_df_resident_response_cuda.py`, and `test_df_final_projection_cuda.py`.

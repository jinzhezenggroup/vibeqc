# Decision: bound DF triples validation and intermediate lifetimes

Status: implemented
Date: 2026-09-21

## Problem

The factorized (T) path admitted its scratch budget after whole-input finite masks and virtual-pair symmetry subtraction. Thus a rejected one-byte budget still performed numeric scans, and validation allocated virtual-sized scratch outside the occupied-cube bound. The previous virtual triple's Z dictionary also survived until the next dictionary was constructed. Finite inputs could overflow arithmetic and return NaN as an ordinary energy.

## Decision

Perform dtype/shape admission without coercing non-array inputs into unbudgeted storage. Admit the explicit numeric scratch next, then validate finite values and symmetry in chunks derived from that allowance. Retire W/V/Z/denominator owners after each virtual triple. The conservative bound includes 32 occupied cubes plus ten occupied matrices and one occupied vector, covering the shared r3 expression's projection temporaries as well as validation. This is a logical explicit-array bound, not a process-RSS or opaque BLAS-workspace guarantee.

Trap overflow/invalid arithmetic and reject nonfinite corrections or composed total energies. The standard permutation, degeneracy and denominator inventory remains unchanged.

## Rejected alternatives

Whole-factor validation, keeping the old twenty-cube allowance, or clipping nonfinite energies would preserve the bug or alter scientific results. Neither is an acceptable bounded fallback.

## Evidence

Six of seven initial failure-first tests failed on the original source. A separate composed-total overflow case also failed before its repair. After repair all 23 selected admission, same-Hamiltonian triples, H2/H2O solver, factorized residual and reference-identity tests pass. The independent dense oracle and all numerical tolerances are unchanged. No new GPU or performance campaign is claimed.

## References

PR #896; issue #157; `tests/python/test_df_triples_admission.py`.

Agent: ChatGPT
Model: GPT-6 Astra Pro

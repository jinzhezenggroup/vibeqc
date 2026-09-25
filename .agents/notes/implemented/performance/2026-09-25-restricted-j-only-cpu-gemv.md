# Decision: route restricted exact J-only CPU builds through shared GEMV

Status: implemented
Date: 2026-09-25

## Problem

Pure semilocal restricted KS builds repeatedly request exact Coulomb J without exchange. The existing CPU exact provider already materializes the full row-major chemists-order ERI matrix, but the generic scalar J/K contraction still revisits that matrix with handwritten AO-pair loops.

## Decision

For the narrow `coulomb.present && !exchange.present && Restricted` exact-CPU case, reuse the shared CPU linear-algebra boundary and evaluate the raw Coulomb contraction as a dense GEMV over the already-materialized `nbf^2 x nbf^2` ERI layout. Keep coefficient application in the existing Fock assembly.

All exchange-bearing, K-only, unrestricted, range-separated, density-fitted and CUDA paths retain their existing owners. If the external CPU BLAS provider is unavailable, the shared linear-algebra boundary retains its scalar fallback.

## Invariants

- The ERI storage order remains the existing row-major chemists-order pair matrix.
- GEMV returns the same raw J semantics as the scalar contraction; method coefficients are not folded into this owner.
- No exchange or unrestricted request may enter this path.
- Existing invalid-input and pinned numerical tests remain acceptance gates.
- This implementation choice is not a speedup claim without complete CPU endpoint evidence.

## Evidence

The exact-head repository CI, Pre-commit, PR-overlap and CuMetal workflows passed at the reviewed implementation head before this note. Source review confirms the dispatch is restricted to exact restricted J-only requests and preserves the generic scalar owner for all other cases.

The PR's stated performance acceptance is still outstanding: complete CPU DFT endpoint timing/work evidence must demonstrate a real benefit before promotion or merge. Existing skipped/untouched benchmark output is not sufficient evidence by itself.

## Consequences

When admitted, the contraction can use the repository's bounded CPU BLAS implementation instead of a dedicated scalar four-index reduction. The scientific owner and public Fock coefficient semantics remain unchanged.

## Revisit when

Revisit if ERI storage changes, if a matrix-free/direct-J owner replaces materialized ERIs, or if complete endpoint measurements show that GEMV does not improve the targeted CPU DFT workload.

## References

#168, PR #1277.

Agent: ChatGPT
Model: GPT-5.6 Sol

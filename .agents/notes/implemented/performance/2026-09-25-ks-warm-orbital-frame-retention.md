# Decision: retain CUDA KS warm orbital frames in-trajectory

Status: implemented
Date: 2026-09-25

## Problem

The CUDA KS path already retains the accepted warm density, but the ordinary eigensolver's orthonormal-basis orbital frame was discarded after each accepted proposal. Future occupied-subspace warm admission (#991/#996) would therefore need another matrix transfer or recomputation even when the frame was already available on device.

## Decision

Retain one per-spin orthonormal orbital frame beside the accepted proposal for the lifetime of the current geometry trajectory. This slice does not consume the frame to bypass the dense eigensolver and does not project it across changed geometries. New solve epochs, explicit warm-state clearing, submission/completion exceptions, partial publication failures and terminal physical failures invalidate frame eligibility while preserving the separately qualified last-good warm density.

## Invariants

- Scientific equations, eigensolver selection, tolerances, occupations and final-state validation are unchanged.
- The retained frame is implementation-only state and is not part of the public C transport ABI.
- A changed geometry may reuse the existing warm density but cannot reuse the old orbital frame in this slice.
- Partial density/frame publication cannot leave an older frame eligible.
- Arena accounting includes the additional per-spin matrix storage.

## Evidence

At head `540cc62f44e31a9c7e9a23d686ca9920b26d1c8a`, repository CI, CuMetal CUDA, Pre-commit and PR-overlap checks pass. The retained failure/control regression covers submission, completion, physical rejection, partial copy failure, retry, convergence and explicit clearing.

Source-matched RTX 5090 validation on 2026-09-25 exercised native CUDA KS for LDA/PBE/r²SCAN and RKS/UKS. Parent/head complete PBE endpoint comparisons preserved convergence iterations and energies for cold, warm and changed-geometry runs; representative timings were 111.10/110.90 s cold and 13.87/13.85 s warm for RKS water/def2-SVP, and 8.60/8.57 s cold and 2.57/2.57 s warm for UKS OH/STO-3G. This slice makes no speedup claim because the dense eigensolve is still executed.

## Consequences

One additional per-spin matrix remains resident during a trajectory. This intentionally trades bounded device memory for an explicit future warm-subspace admission seam without changing current endpoint work.

## Revisit when

Revisit once #991 consumes the retained frame to skip or reduce dense eigensolver work, or if changed-geometry frame projection is independently qualified.

## References

#991, #996, PR #1162.

Agent: ChatGPT
Model: GPT-5.6 Sol

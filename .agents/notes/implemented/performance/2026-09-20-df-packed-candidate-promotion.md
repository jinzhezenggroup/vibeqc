# Decision: remove benchmark identity from packed-response dispatch

Status: implemented
Date: 2026-09-20

## Problem

PR #631 introduced an overflow-safe packed-response workload preference but left
production auto-selection tied to one benchmark identity: equal 768/768 AO spaces,
occupied rank 160, and the literal product name `NVIDIA GeForce RTX 5090`.
Those facts are evidence provenance, not mathematical or runtime capability.

## Decision

Production packed-response auto-selection now calls
`df_packed_response_preferred(nbf, naux, rank, architecture)`. The policy uses
general workload size, occupied-rank fraction, and CUDA architecture. The DF
gradient bridge retains the independent correctness and ownership gates:
generated shell execution, device metric, resident/borrowed occupied response,
single RHF response term, diagnostic exclusions, state/provenance validation,
capacity checks, and the existing symmetric fallback.

No production dispatch branch names a benchmark AO/rank tuple or GPU marketing
name. Explicit `full`, `symmetric`, and `packed` controls remain unchanged.

## Qualification boundary

This change removes benchmark identity from dispatch; it does not turn policy
arithmetic into a claim that every newly selected workload is faster. #444 remains
the owner of clean endpoint/resource qualification and any later tuning-profile
adjustment. The practical 384/1856 cc-pVDZ/JKFIT case must not be used to justify
promotion because its retained cold force result exceeds its independent gate.

The safe fallback remains `symmetric` whenever correctness/capability admission
fails. Performance evidence may refine the shared profile; it must not reintroduce
endpoint tuples or product-name checks into the execution bridge.

## Evidence

#631 already validated the exact workload predicate over 20,007 boundary/random
cases, including size_t extremes and unknown architectures, against an
arbitrary-precision Python oracle. The production-boundary regression now checks
that the bridge consumes that policy and contains none of the retired
`768/160/RTX 5090` identity checks.

No DF equation, precision mode, metric threshold, force definition, or
correctness/provenance gate changes in this decision.

## References

- #444, #459, #631.
- `src/scf/df_derivative_policy.hpp`.
- `src/scf/cuda/df_gradient_bridge.cu`.
- `tests/python/test_df_derivative_policy.py`.
- `tests/python/test_df_packed_promotion_boundary.py`.

Agent: ChatGPT
Model: GPT-5.6 Sol

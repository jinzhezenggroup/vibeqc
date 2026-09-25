# Decision: compute host DIIS Gram matrices by unique symmetric pairs

Status: implemented
Date: 2026-09-24

## Problem

The method-neutral host DIIS helper stored residual histories as flattened FP64 vectors and rebuilt its dense Gram matrix by evaluating every ordered `(i, j)` pair. The Euclidean Gram matrix is symmetric, so `(i, j)` and `(j, i)` traversed the same residual vectors twice. This work is paid on every DIIS update and scales with both history squared and residual-vector length.

This helper is currently a production consumer for the conventional CPU RCCSD solver. It is separate from the resident CUDA CC DIIS kernel and from the SCF-specific generated/native DIIS implementation.

## Decision

Evaluate only the upper triangle including the diagonal. For each unique pair, preserve the existing `std::inner_product` operand order and FP64 accumulation order, then copy the completed scalar into the transposed dense-Gram slot. The subsequent normalization, augmented solve, coefficient guards, history retirement, extrapolation, and public dense layout are unchanged.

## Rejected alternatives

- Do not change the DIIS linear solve to a packed matrix. The matrix is tiny relative to the residual histories, and changing solve/layout ownership would broaden the numerical surface without reducing the dominant residual-vector traversals.
- Do not fuse this change with the SCF generated DIIS emitter. That path has separate compiler/code-generation ownership and active work in nearby files; keeping this PR on the method-neutral host helper avoids overlap.
- Do not claim a wall-time speedup from pair counts alone. Complete RCCSD timing remains the promotion evidence because equation contractions can dominate the endpoint.

## Invariants

- The same residual vectors define every Gram entry.
- Each retained dot product uses the same element order and FP64 arithmetic as before.
- The full dense symmetric Gram layout presented to normalization and `solve_linear` is preserved.
- Scale selection sees the same set of absolute Gram values; mirrored duplicates cannot change the maximum.
- Singular-history retirement and coefficient guards remain unchanged.

## Evidence

For history `h`, ordered residual traversals fall from `h^2` to `h(h+1)/2`:

| history | before | after | reduction |
| ---: | ---: | ---: | ---: |
| 2 | 4 | 3 | 25.0% |
| 6 | 36 | 21 | 41.7% |
| 8 | 64 | 36 | 43.75% |
| 20 | 400 | 210 | 47.5% |

Each traversal touches `elements` entries from each of two residual vectors and performs `elements` products/accumulations. The new source regression test locks the triangular schedule and representative work counts. Existing `vibeqc_self_consistent_tests` exercise method-neutral DIIS extrapolation, dependent-history retirement, disabled mode, and invalid-shape preservation; RCCSD solver tests remain the endpoint correctness gate.

Performance acceptance should compare a source-matched Release CPU RCCSD run with identical reference/provider/options, reporting complete solve time, iteration count, DIIS history, and residual work. A focused profile may additionally measure DIIS Gram time, but it must not replace complete endpoint timing.

## Consequences

The optimization removes nearly half of residual-history dot work at practical larger histories with no extra allocation and only one scalar mirror store per off-diagonal pair. Benefit depends on how much of the complete CPU RCCSD endpoint is spent in DIIS; large equation contractions may make the wall-time effect small.

## Revisit when

Reconsider the schedule if the host helper changes to a non-symmetric metric, a different accumulation contract, or a solver that can consume packed Gram storage directly with measured endpoint benefit.

## References

- Issue #928 — shared iterative solver / host DIIS ownership.
- `src/solver/diis.hpp` — method-neutral host DIIS implementation.
- `tests/python/test_host_diis_gram_symmetry.py` — schedule/work-count regression.

Agent: ChatGPT
Model: GPT-5.6 Sol

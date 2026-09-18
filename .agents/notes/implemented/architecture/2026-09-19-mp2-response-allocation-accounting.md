# Decision: real response allocation accounting without endpoint promotion

Status: implemented
Date: 2026-09-19

## Problem

PR #426 needs measured simultaneous-live numeric storage rather than a copy of
its admission plan. GMRES scratch, restart copies, and returned solution storage
have different lifetimes. Adding phase maxima or sampling only a returned vector
cannot recover the peak.

## Decision

Use an explicit shared allocation domain, backed by the standard allocator.
Every successful vector allocation and actual deallocation updates a mutex-guarded
counter. Storage growth counts the overlap of old and new allocations. Copy,
move, and swap propagate the domain, whose shared lifetime can outlive a solver
call and whose counter is safe for cross-thread destruction. Numerical work is
not serialized; only allocation events within the same domain take its lock.
GMRES uses this allocator for every owned numerical array, including restart
copies and the returned solution. Budget rejection precedes even counter creation.

The public response-only peak and allocation count are appended to the existing
C diagnostic and propagated through Python. The existing workspace field remains
an admission plan. These fields do not cover input spans, operator-callback
allocations, reference/provider buffers, derivative staging, CUDA ownership, or
allocator overhead. Complete endpoint measurement remains unavailable (zero).

## Rejected alternatives

Do not replace measured endpoint usage with a conservative plan, the sum of
non-overlapping maxima, process RSS, or the new response-only measurement. Avoid
replacing global new/delete in the production shared library, which would
interfere with callers and mix unrelated calculation lifetimes.

## Invariants and evidence

Native allocator regressions exercise overlap versus disjoint lifetimes,
reallocation, copies/moves/swaps, exception unwinding, overflow refusal, concurrent
owners, and foreign-thread destruction. GMRES regressions retain the independent
residual/numerical gates and assert actual counts for zero RHS, output lifetime,
workspace rejection and replay after an operator exception. Public tests preserve
the zero endpoint sentinel while exposing response measurement. Existing full
endpoint qualification must continue to reject the sentinel.

## Consequences and remaining work

There is a small per-allocation bookkeeping cost and a counter lifetime attached
to returned native response storage. No numerical tolerance or acceptance gate is
changed. Extending the accounting to the reference, MO provider, derivative
staging and CUDA owner lifetimes is still required before B2 resource acceptance.
CPU/CUDA exact-head scientific and sanitizer qualification remains separate.

## References

PR #426, issue #193 B2; `src/runtime/tracked_allocator.hpp`,
`src/response/native_gmres.cpp`, and `tools/validate_mp2_public_force.py`.

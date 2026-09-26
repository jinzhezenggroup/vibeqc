# Decision: Bound CUDA Lambda host borrows to each response call

Status: implemented
Date: 2026-09-26

## Problem

The CUDA Lambda owner retained its device allocation and stream longer than the
host buffers borrowed by individual actions. On a copy or synchronization error,
local vectors and stack error flags could unwind before the owner's destructor
drained the stream. Parameter setup also returned while an asynchronous upload
still borrowed its local energy seed, even without an error.

The Hamiltonian response owner's lifetime tests did not cover these Lambda
methods. Pageable-host staging can hide either defect on a particular driver;
it is not a portable lifetime guarantee.

## Decision

Use scope-local transfer fences for both the action's host inputs and the output
detach helper's local error flag. Declare each fence after the buffers it protects
so that unwinding drains before those buffers are destroyed. Dismiss it only
after the normal successful stream synchronization.

Parameter seed setup completes all three uploads before returning. This adds one
counted synchronization per combined Lambda/parameter solve, while retaining one
upload of the accepted cotangents for all ten device parameter programs. The
existing per-action normal synchronization count remains unchanged.

## Rejected alternatives

- Draining only in the owner's destructor is too late for method-local storage.
- An outer input fence cannot protect a stack error flag in an inner copy helper.
- Retaining just the scalar seed would not establish a lifetime contract for the
  two borrowed Lambda spans, particularly during later allocation failures.
- Depending on an eager pageable-memory copy would leave backend-dependent
  behavior instead of an explicit ownership guarantee.

## Evidence and invariants

`tests/python/test_cc_cuda_lambda_lifetime.py` executes the production owner with
delayed-copy CUDA doubles. It covers both Lambda forms, replay, parameter seeds,
parameter outputs, every transfer failure, synchronization errors, generated
action errors, and reuse after failure. The original pre-parameter owner failed
26 of the 37 initial cases. The parameter extension also requires seed uploads
to finish before setup returns.

Host delayed-copy checks protect lifetime ordering; they do not establish CUDA
scientific correctness. CUDA dispatch still requires a build and the public
RCCSD(T) analytic-gradient oracle gate on the final integrated stack.

## Revisit when

A future resident response interface owns the host staging buffers or eliminates
these transfers entirely. Until then, each synchronous host-facing method must
finish its borrows on success and unwind them on failure.

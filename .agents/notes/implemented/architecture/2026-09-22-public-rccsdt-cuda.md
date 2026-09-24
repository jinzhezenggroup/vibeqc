# Decision: compose the native CUDA RCCSD(T) energy owner

Status: implemented
Date: 2026-09-22

## Problem

Native CUDA RCCSD and validated triples science existed, but the public
RCCSD(T) owner and Python admission were CPU-only. A retained implementation
draft supplied a native CUDA lowering of the audited standard triples inventory.

## Decision

Keep the native RCCSD lifecycle and physical-residual acceptance unchanged;
select generated CPU or CUDA triples only after it converges. Both generators
share the audited permutation tables. CUDA uses fixed work assignment and
ordered block/final reductions instead of nondeterministic energy atomics.
It stages the accepted host arrays once and evaluates the triangular domain
without allocating a full occupied/virtual T3 tensor.

## Invariants

No CPU triples/reference fallback, changed approximation, or unconverged triples
publication is allowed. Every finite input and physical denominator is checked.
Triples allocation is admitted against the remaining correlation budget after
retained host CC capacity. CUDA device/stream resources drain and release on
failure, and the prior selected device is restored.

## Rejected alternatives

The internal Python composition would introduce a runtime tools dependency.
A persistent CC-to-triples pointer handoff would require a separate solver
lifetime contract. Atomically summing energy would make repeated runs vary.
A full T3 intermediate violates the bounded memory contract.

## Evidence

`test_rccsdt_cuda_codegen.py` compares unequal `(o,v)` shapes against the
independent full-sum oracle and checks deterministic repeats, exact workspace
admission, nonfinite inputs and denominator failures. Public tests use pinned
molecular references, prepared batch geometry updates and failure isolation.

## Consequences

The existing conventional MO integral preparation and inter-stage host handoff
remain explicit. Recomputing W/Q seeds bounds memory but increases arithmetic;
this change claims interface correctness, not a performance promotion or complete
end-to-end device residency. Public analytic forces remain separately gated.

## Revisit when

A common resident CC state lifetime and independently qualified cooperative
triples lowering support complete-endpoint performance and memory improvements.

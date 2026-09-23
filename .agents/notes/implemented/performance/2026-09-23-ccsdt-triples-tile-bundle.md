# Decision: prewarm exact perturbative-triples response tiles

Status: implemented
Date: 2026-09-23

## Problem

PR #1059 shared native CPU TensorIR bundles across CC/Lambda owners, but the
complete RCCSD(T) response still constructed each requested perturbative-triples
tile VJP on first execution. With a virtual chunk size of one, a three-virtual
case could therefore create three shared libraries per selected source set.

## Decision

At the triples accumulation boundary, generate the exact requested tile VJP
programs once and prewarm them together when the supplied executor supports
prewarm. Execute those same Program objects. Interpreter and other executors
retain their prior per-tile path. The native executor's existing single-program
route remains available for execution without prewarm.

## Rejected alternatives

Do not precompile all possible triples inputs at Lambda construction: the
consumer may request only amplitude, one direct parameter, or denominator
sources. That would compile unnecessary scientific programs and could violate
the requirement to compare matched semantic work.

## Invariants

Preserve the existing tile partition, demand-driven generated VJP, denominator
and input gates, exact program identity, byte/work/node admission, and
independent gradient oracle. A failed prewarm must not publish tile results.

## Evidence

`tests/python/test_cc_triples_response.py` checks that the prewarmed set is
exactly the executed set and that all returned cotangents match the interpreter.
The complete cold/warm endpoint comparator and CI artifact procedure are in
`tools/benchmark_ccsdt_cpu_bundles.py` and
`docs/maintainer/ccsdt_cpu_bundle_qualification.md`.
The comparator supplies one fixed RHF generation ID to all independent child
processes: the production exporter otherwise assigns a fresh UUID, making
cross-process CC/response identity equality impossible despite equal physics.

## Revisit when

If complete-endpoint evidence shows that compile/link work still dominates
after aggregation, consider the separately scoped parallel/incremental AOT
Slice D. Preserve finite compiler-process semantics and the same work gates.

## References

Issue #866; PRs #868, #891, #1059.

# Decision: shared bounded runtime task domains

Status: implemented
Date: 2026-09-21

## Problem

Two successful performance slices had converged on the same execution shape without
sharing an owner. Stationary DFT force submission (#664 / #683) paged rectangular
AO tuples before native primitive traversal, while RCCSD(T) triples (#783 / #790)
paged the monotone virtual domain a >= b >= c before feeding runtime TensorIR
index maps. Each path independently owned enumeration, tail handling and page
counting.

Leaving those copies in place would make later scheduler work (#682), DF
signature buckets (#385), and resource-aware paging learn multiple incompatible
notions of a bounded runtime work domain.

## Decision

Add a pure compiler-common RuntimeTaskDomain plus RuntimeTaskPage. It represents
finite rectangular products and nonincreasing simplex domains, deterministic
identity, exact logical work count, explicit membership guards, and lazy bounded
page iteration.
The common layer owns only domain semantics and bounded page metadata. It does
not own a scientific equation, NumPy control arrays, CUDA allocation, provider
lifetime, convergence policy, or method-specific schedule choice.

RCCSD(T) TileSpec now derives its triple count, iteration order, control pages and
runtime batch count from the nonincreasing domain. Stationary DFT derives its
rank-2/rank-4 AO tuple pages from the rectangular domain while retaining the
existing compact descriptor ABI and native primitive Cartesian traversal.

## Rejected alternatives

Keeping two local batching helpers was rejected because page identity, tail
semantics, work counts and future scheduling metadata would continue to diverge.

Moving primitive expansion or triples equations into the common domain was
rejected because those are scientific/lowering semantics, not generic domain
semantics.

Making the common abstraction depend on NumPy or CUDA was rejected so compiler
planning and identity remain usable from an uninstalled CPU-only checkout.

## Invariants

- Domain enumeration is finite, deterministic, zero-based and exact.
- Pages are capacity-bounded and never require full-domain materialization.
- Nonincreasing domains preserve x0 >= x1 >= ... and an explicit outer window.
- Rectangular domains preserve row-major itertools.product order.
- Domain and page identities contain no device probe, timestamp or method name.
- DFT primitive work accounting remains native and per reset epoch.
- RCCSD(T) still uses generated TensorIR scientific algebra and runtime int64 maps.
- No full T3/denominator tensor or primitive-expanded host table is introduced.
- Scientific legality remains with each consumer; the common domain is execution
  structure only.

## Evidence

Host regression command:

PYTHONPATH=python:. python -m pytest -q tests/python/test_runtime_task_domain.py tests/python/test_cc_triples_tiles.py tests/python/test_stationary_task_work_budget.py

Result before the final batch-count cleanup: 82 passed.

Compiler ownership check:

PYTHONPATH=python:. python tools/check_compiler_structure.py

Result: 262 compiler modules checked, 0 dependency errors.

The prior production slices retain their independent device evidence: #683 for
stationary native task batches and #790 for runtime-indexed CUDA triples. This
refactor makes no new endpoint-speed claim.

## Consequences
Future runtime-domain scheduling can attach resource/profitability metadata to one
compiler-owned domain instead of teaching DFT, DF derivatives and TensorIR separate
paging semantics. Consumer-specific signature/class grouping can remain explicit
until #385 supplies a second qualified use case; it should layer on this domain
rather than create another work-range abstraction.

## Revisit when

Extend the domain kinds only when a real consumer needs a finite shape that cannot
be expressed as rectangular or nonincreasing work. Add scheduler-visible signature
buckets when #385 or another second consumer demonstrates the required legality
and profitability metadata.

## References

- #664, #683
- #783, #790
- #385
- #682

Agent: ChatGPT
Model: GPT-5.6 Sol

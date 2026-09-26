# Decision: batch the existing source-occupied projection

Status: implemented
Date: 2026-09-25

## Problem
The explicit source-occupied control submits one source slice, charge GEMV and
two projection GEMMs for each auxiliary function. It is not a valid proxy for
an efficiently batched low-rank algorithm (#1078).

## Decision
Add `VIBEQC_DF_SOURCE_PROJECTION=auto|batched`; auto preserves previous behavior.
The explicit batched control uses the same generated occupied panel projection
as the retained-B path. The source emits pair-major panels and the existing
gather converts them to auxiliary-major. A second raw panel is charged against
the same exchange scratch; no budget or permanent allocation is increased.

The source callback is synchronous at the bridge boundary, propagates failures,
and cannot authorize a raw cache or future force lease. The derivative-schedule
control from #445 remains separate so producer/consumer ablations are possible.

## Evidence
Host layout and panel-capacity contracts are covered by the shared lowering
tests. Explicit numerical, compile and device performance gates remain. No
production selection, release, merger or measured speedup is implied.

## References
#1078, #445. Agent: ChatGPT. Model: GPT-5.6 Sol.

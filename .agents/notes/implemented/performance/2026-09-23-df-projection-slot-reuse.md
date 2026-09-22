# Decision: retain adjacent streamed DF projections in compiler traversal

Status: implemented; GPU and complete endpoint qualification pending
Date: 2026-09-23

## Problem

Issue #1078 remains open after reducing occupied K from seven raw tensor passes
to 1.5 at 768 AOs, 3712 auxiliaries and rank 160. A bounded 60-second run with
the original 13,685,173,124-byte value allowance still expires during SCF, before
forces. It completes 68 raw generations taking 51.664325 seconds. One uncached
J takes 18.0173 seconds; one uncached occupied K takes 17.35904 seconds. These
are intrusive progress-trace times, not clean endpoint benchmarks.

The actual tile is 579 auxiliaries, and the compiler chooses two blocks of 384
AO rows. The executor loads row blocks 0, 384, then 0 again, although the first
projection could remain live in its existing slot. The four charged buffers
already suffice. The same unnecessary adjacent-row regeneration occurs for
larger block counts.

## Decision

The compiler emits the visit order and two-slot identity in addition to the
capacity/work policy. Native callbacks bind projection/GEMM execution to those
slots. Triangular traversal alternates slots for new output rows, contracts the
diagonal, then visits columns in descending order. The first off-diagonal uses
the preceding output row retained in the other slot. Earlier prefixes overwrite
that other slot only after its last use; the current output row stays live for
the next outer visit. Full-matrix traversal retains its existing order.

For n rows, b balanced blocks and width h, raw generation changes from
`n + h*b*(b-1)/2` to `n + h*(b-1)*max(0,b-2)/2`. One and two blocks both need
one pass; ties prefer fewer matrix products, then the smallest balanced width.
At the original shape, K generates 2,189,426,688 values instead of 3,284,140,032.
J remains two passes, so one eligible uncached J+K build needs three raw tensor
passes instead of 3.5. Seed retries, cache hits and final dense K must still be
counted separately from reported SCF iterations.

## Invariants and rejected alternatives

No additional allocation, coefficient cache across calls, response-budget loan,
precision change, density acceptance relaxation or force-response lease is
introduced. Both retained projections remain disjoint from raw input and metric
scratch. All commands use the same stream, so overwritten projections are dead
before reuse, including graph replay with changed coefficients. Private metric
eigendirection projections are never published as symmetric-C response factors.

Changing only the work estimate would undercount real execution. A native-only
pointer swap would leave compiler admission disconnected from execution. The
emitted visitor and independent lifetime/census tests keep both consistent.
Retaining the full raw tensor exceeds the original value budget; this change
uses only existing bounded storage.

## Evidence

- Host tests execute the emitted C++ visitor over small divisible/nondivisible
  shapes and independently verify slot identities, complete pair coverage,
  capacity, exact source census and immediate callback failure propagation.
  Every legal row width is enumerated to qualify the work/width choice.
- The exact 768/3712/rank-160 fixture reports two blocks, 768 generated rows
  and unchanged dense fallback work of seven tensor passes.
- Native independent physical-integral tests cover four/five AO rows, ranks
  zero/one/two, triangular/full traversal, both coefficient layouts, truncated
  metric fallback and captured replay after changing coefficients.
- GPU measurements and complete cold/warm/changed-geometry gates remain pending.

The retained baseline diagnosis uses integration v11, library SHA256
`9d0dda6deeb41dc83f9a7bac11f83b4573af7fc0a1cce2f2439b3f7765093aad`,
source archive SHA256
`23022825bfbc4b8b827b6ac1c75dcfe6b79e90f9f13be753cc4b8be4398befdc`.
Slurm 11410 artifacts are `hf96-streamed-steps-v11*` in the preserved integration
worktree. Its energy-only resident control (Slurm 11408) changes the resource
regime and cannot establish completion of the original energy-plus-force case.

## Revisit when

Raw-source work still dominates after this bounded reuse. Further improvement
requires source-generation or J/consumer fusion evidence, or a separately
audited live-set strategy. Complete 96-atom energy-plus-force qualification and
README publication remain paused; this pass reduction does not close #1078.

This supersedes the triangular source-work count in
[the streamed value decision](2026-09-23-streamed-df-value-exchange.md).

# Decision: remove benchmark selectors without unmeasured promotion

Status: implemented in PR #706; required CI remains the integration gate
Date: 2026-09-20

## Problem

The previous change removed an exact AO/rank/product-name selector but also
promoted every workload accepted by a heuristic. That heuristic has arithmetic
coverage, not complete numerical/resource/performance qualification. The retained
unequal-auxiliary example cannot qualify it because its cold force gate failed;
this does not establish that packed response caused that historical failure.

## Decision

Remove the benchmark selector and keep packed response explicitly selected by
`VIBEQC_DF_DERIVATIVE_PAIRS=packed`. Automatic execution retains the existing
full/symmetric routes. Keep the shared generalized preference and its independent
arithmetic tests as candidate infrastructure, not active promotion evidence.
All existing packed ownership, capability, lifetime, capacity, and fallback
checks remain in force. No scientific expression or numerical gate changes.

## Tradeoff and rejected alternatives

The previously measured narrow case now also uses the conservative automatic
route. This may cost performance for that case; the explicit packed control
preserves its execution path. This deliberate temporary tradeoff is preferable
to restoring benchmark fingerprints or silently extending an unqualified default.
No new GPU performance result or general speedup is claimed.

## Validation

The production boundary regression requires explicit opt-in and retained
ownership/capability guards, forbids the retired identities, and preserves the
automatic symmetric fallback. The existing shared-policy arithmetic suite remains
unchanged. Final native/CUDA CI remains required.

## Revisit when

A reusable measured workload/target profile qualifies cold, warm, changed-geometry,
unequal-auxiliary and size-boundary complete endpoints under #444. Promote that
profile explicitly without benchmark fingerprints in the execution bridge.

Supersedes the dispatch decision in
`2026-09-20-df-packed-candidate-promotion.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro

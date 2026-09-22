# Bound structural capability investigations, not production arithmetic

Status: implemented
Date: 2026-09-22

## Problem

The broadened target-legal schedule inventory made structural reports expand
large packed symbolic DAGs for every shell/recurrence. Several multi-megabyte
packed candidates dominated the report. Removing one duplicate first-derivative
sweep helped, but did not bound any individual expansion. An unbounded report
prevented the full core tests from reaching later regressions.

## Decision

Report candidates use an explicit deterministic budget of 100,000 symbolic
intern attempts each. Work includes reuse attempts, not just new allocations,
and is charged before graph mutation. Context-local scopes restore their state
on exceptions; nested scopes charge the outer budget as well. Ordinary graph
construction, production lowering, autotuning and emitted scientific arithmetic
have no active budget unless their caller deliberately enters a scope.

The report records the budget unit and limit. A candidate that exhausts it is
not claimed as emitted: the negative reason explicitly says it is unqualified
within this budget. Every target-legal candidate is still attempted. Successful
alternatives and the distinction from production promotion remain intact. The
API accepts a caller-selected positive limit, or None for an explicitly
unbounded investigation; the CLI exposes the positive limit.

## Rejected alternatives

Do not delete the capability test, increase CI timeouts, remove large legal
candidates, infer success from legality without emission, or silently reuse a
smaller shell's result. Wall-clock cutoffs are nondeterministic. The finite
investigation budget is not a hardware occupancy estimate or an error bound.

## Evidence and limits

The independent bounded prototype completed all 55 catalog classes with the
same existing capability count gates: subset-Wick 55, Rys2 4, Rys3 11, Rys4 16,
Rys5 14, first derivative 55 and second derivative 0. Production artifact byte
identity remains separately tested. Small complete reports match unbounded
emission exactly. Tests cover pre-mutation exhaustion, reused-node charging,
exception restoration, nested limits, thread isolation and explicit rejection.

The first full suite now completed and exposed an unrelated whitespace-only
assertion for the unchanged packed derivative-center table. That regression
continues to assert centers 0, 2, 3 in order, without depending on line breaks.

Agent: ChatGPT (Even-PR Review R6)
Model: GPT-6 Astra Pro

# Decision: bind complete production-domain matrix receipts before promotion

Status: implemented
Date: 2026-09-25

## Problem

#1128 defines the exact versioned production-domain profile, #1040 supplies
deterministic boundary probes plus an independent Libxc oracle path, and #1150
reports the resulting capability state. The missing boundary was an
identity-bound receipt that proves an external numerical run covered every
required `spin x case x output` entry before it can become
`production-domain` stage evidence.

Without that bridge, a later numerical runner would need to construct stage
evidence directly and could accidentally omit part of the profile while still
attaching the canonical profile payload.

## Decision

Add `production_domain_evidence.py` as a narrow evidence adapter.

- The required matrix is derived only from the current
  `ProductionDomainProfile`; no second case inventory is maintained.
- A receipt is bound to both the bulk-functional capability identity and the
  exact production-domain profile identity.
- Every profile case must be present exactly once for both spin layouts and
  must declare the exact profile output set.
- The normalized receipt receives a deterministic content identity.
- Any failed or not-run case remains a failed/not-run stage with an actionable
  reason; only an all-pass receipt receives the exact qualification payload
  accepted by `libxc_bulk_capabilities`.
- Structurally blocked ingredient sets cannot receive a receipt.

This module does not evaluate XC mathematics, choose tolerances, or qualify any
Libxc registration by itself. The numerical producer remains responsible for
independent-oracle comparison against the actual production executable.

## Consequences

The next #1120 slice can consume the generated point program from #1276 and the
independent boundary oracle, fill this complete receipt, and feed the resulting
stage evidence into the existing capability catalog without inventing a second
promotion path. Partial, stale, tampered, blocked, and unavailable campaigns
remain fail-closed.

Refs #1120 #1128 #1040 #1150 #1276

Agent: ChatGPT
Model: GPT-5.6 Sol

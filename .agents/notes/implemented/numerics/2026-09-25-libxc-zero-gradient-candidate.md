# Decision: admit zero sigma only in the Libxc qualification candidate

Status: implemented
Date: 2026-09-25

## Problem

The bulk runtime validator inherited the pointwise interior contract and rejected
same-spin sigma <= 0 before evaluating the imported Graph. That makes the
production-domain `sigma/zero` row impossible to qualify generically, even for
functionals whose E/vxc/fxc limits at zero gradient are finite and independently
verifiable.

Simply relaxing the default runtime would silently enlarge an existing execution
domain before evidence exists. Bypassing validation inside the evidence tool
would test a different, unnamed execution path.

## Decision

Add the explicit runtime domain
`libxc-bulk-production-candidate/v1`. It preserves the imported Graph and all
existing structural gates, but changes the sigma admission rule from strictly
positive to nonnegative.

The candidate therefore admits exact physical zero gradients while continuing to
reject:

- negative same-spin/unpolarized sigma;
- non-PSD polarized sigma Gram matrices;
- non-positive densities;
- non-positive tau for tau-dependent meta-GGAs; and
- nonfinite inputs.

The ordinary runtime builder still defaults to `libxc-bulk-interior/v1`.
Only the production-domain evidence campaign explicitly selects the new
candidate. The candidate domain is part of `BulkRuntimeSpec.to_payload()`, so
its expression identity differs from the interior program and is retained by the
v2 execution-bound receipt.

The shared nonfinite control consumes the same bound candidate program used by
the campaign.

## Rejected alternatives

- Relaxing the default interior validator would grant an unevidenced runtime
  domain expansion.
- Hard-coding zero-gradient values or derivatives would create a generic
  mathematical continuation that is not valid for every Libxc functional.
- Skipping zero-gradient rows would weaken the versioned production profile.
- Evaluating the raw Graph behind the validator would break execution-identity
  ownership.

## Invariants

- Zero sigma is an input admission change, not a claim of finite derivatives.
- Actual E/vxc/fxc evaluation and the independent oracle decide pass/fail per
  functional.
- Density, spin-density, and tau endpoint policy remains fail-closed.
- Candidate and interior domains have distinct expression identities.
- No public/runtime capability is promoted from candidate construction alone.

## Evidence

Focused regressions show the default interior domain still rejects zero sigma,
the production candidate accepts it structurally, negative/indefinite sigma
still fails, density/tau endpoints remain rejected, and an unknown runtime
domain fails closed. The candidate and interior expression identities are
distinct.

Repository CI is the executable authority for the stacked branch.

## Consequences

The B1 campaign can now distinguish "zero gradient is mathematically unsupported
or numerically unstable for this functional" from "the old interior validator
never let us test it." Families with safe zero-gradient order-2 behavior may
advance automatically; others remain blocked with exact evidence.

## Revisit when

A generic, independently qualified density/spin/tau endpoint continuation is
introduced. That must use a new production-candidate domain version rather than
silently extending v1.

## References

- #1118
- #1120
- #1315
- #1320
- #1322

Agent: ChatGPT
Model: GPT-5.6 Sol

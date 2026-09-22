# Decision: admit serialized resource phases by peak live set

Status: implemented
Date: 2026-09-22

## Problem

DF automatic budgeting split one device envelope into value and response buckets
even though value setup/SCF work and force-response workspace are sequential. A
resident owner could therefore fit the real peak live set yet be rejected by the
smaller value partition, silently selecting a slower source-backed route.

CUDA KS planning had the same structural risk: one-electron setup excess and
generated-force staging were both added to the retained owner even though those
transient phases do not overlap.

## Decision

Treat the total automatic DF envelope as the admission cap for a complete
resident value/SCF owner. Keep the explicit value/response split for bounded
fallback and response scratch, but do not use the value slice alone to reject a
resident owner whose value-phase peak fits the total envelope.
DIIS admission follows the selected value-phase cap. Explicit positive budgets
retain their existing bounded routing semantics.

For KS, collapse serialized setup/SCF/XC/force transient storage to one maximum
phase excess per memory space. Persistent state remains charged across all
phases.

## Invariants

- Automatic resident DF admission remains single-item dense RHF only.
- The existing DF tile planner must prove the complete owner fits.
- Persistent owners are never treated as dead between phases.
- Sequential transient workspaces are charged by maximum live excess, not sum.
- Numerical equations, thresholds, precision, and force definitions are unchanged.

## Evidence

Regression coverage constructs a DF case where the resident owner exceeds the
legacy value slice but fits the total automatic envelope; resident admission
must be retained. CUDA KS coverage independently makes setup and force
transients large enough that summing them would reject a valid plan, then
requires the phase-peak plan to fit exactly.
The motivating RTX 5090 equal-basis DF investigation found that routing a
768-AO case away from resident response exposed a bounded response path. The
independent #940 charge-contraction fix removed its dominant scalar kernel; this
decision prevents the resource split itself from selecting that slower owner
when the actual phase peak fits.

## Consequences

Resource estimates describe lifetime rather than administrative sub-budgets.
New DFT/response components should either declare explicit non-overlapping
phases through ResourcePlan or collapse serialized scratch into a maximum phase
excess before composition.

## References

- #439
- #890
- #940
- `python/vibeqc_compiler/common/resources.py`

## Retained force-owner correction

The CUDA force lifetime assumption is superseded by
[retained KS force overlap](2026-09-22-retained-ks-force-overlap.md).
The CPU transient-max and separate DF admission decisions remain unchanged.

## Device acceptance and obsolete test assumptions

The integrated scientific source `06aec739` passed independent PySCF cold/warm
energy and complete-force gates at 96, 192, 384 and 768 AO on an allocated RTX
5090 with CUDA 12.9.86. Explicit 24/32/64 MiB energy/force/energy replay also
passed with both one-electron derivative providers. Every traced value peak
and response scratch allocation fit its resolved allowance, and those
allowances summed to the unchanged public budget. At 24 MiB, force replay
actually streamed values while energy replay retained them.

The older property-budget test still assumed a 50/50 split and a 32 MiB
streaming transition. Both the integrated head and base `9d6d4423` fail those
assertions because the accepted resource policy already allocates by workload.
The test now verifies the public total, actual phase allowances, independently
referenced forces, and a real 24 MiB resident/streamed transition. No production
policy or scientific tolerance changed to make these assertions pass.

Response-panel comparison must first prime the occupied owner and freeze its
density. Comparing a first dense response with later occupied responses mixes
different workspace demands. The existing 4/16/4 MiB limits remain checked;
panel counts must be stable when returning to the same allowance and must not
increase with a larger allowance. A strict decrease is inappropriate when one
occupied panel already fits both limits.

Matched performance comparisons must likewise control the warm density.
Repeatedly updating each implementation's own density produced different SCF
iteration counts and apparent warm regressions. Importing the same checkpoint
bytes and freezing updates isolates the owner change without altering SCF
tolerances. Checkpoint physical validation is explicit benchmark preparation
outside endpoint timing; it is not an implicit production CPU reference.
Retain the initial evolving-density samples alongside the controlled replay,
rather than deleting the first observation or treating it as equivalent work.

Detailed numerical, work-count, timing and sampled-memory evidence is attached
to PR #970. Machine-local raw data and runners are retained under
`/home/jzzeng/codes/vibeqc-ready-20260922/evidence/`.

A separate 512 KiB diagnostic response-override probe reported 604,384 scratch
bytes on both `9d6d4423` and `06aec739`. That pre-existing override-accounting
issue is recorded in the acceptance discussion; this PR does not claim to fix
or qualify that smaller diagnostic limit. It is distinct from the public
positive-budget replay gates and the retained 4/16/4 MiB response test.

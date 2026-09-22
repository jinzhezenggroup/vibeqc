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

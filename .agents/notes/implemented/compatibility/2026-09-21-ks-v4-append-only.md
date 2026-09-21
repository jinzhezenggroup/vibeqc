# Decision: append KS execution identity after the complete v3 ABI

Status: implemented
Date: 2026-09-21

## Problem

PR #781 added compiler-resolved spin/family selectors as a v3 suffix while
mainline independently used v3 for the XC execution schedule. Choosing either
side of the conflict would silently reinterpret the other side's ABI bytes.
Mainline also added PBE-D4 and local-profile scheduling, which must survive the
method-ID-to-execution-plan migration.

## Decision

Preserve v1 grid/tile, v2 composition, and the complete v3 schedule prefix. Make
v3's trailing alignment padding explicit before appending the v4 execution-plan
fields. Return 4 from the version query; negotiate v1/v2/v3 for older libraries.
Read the schedule from complete v3 or later descriptors, and read execution
identity only from complete v4 or later descriptors. Reject truncated suffixes.

Keep local-profile-selected active options and their fail-closed downgrade
checks. Carry the existing PBE-D4 correction flag across the preparation-time
legacy selector projection into the snapshotted plan; prepared owners do not
regain method-ID scientific dispatch. This does not add a new dispersion method
or a generic native geometry-correction ABI.

## Rejected alternatives

Reusing v3 for two layouts corrupts ABI interpretation. Discarding upstream
schedule/D4 additions regresses already-landed behavior. Retaining a public
method ID in each prepared owner recreates the scientific dispatch duplication
that #396 is removing.

## Invariants

Legacy descriptor sizes and offsets remain accepted; host-unfused v3 schedules
remain validated; absent v4 identity is not read. Older-library composition and
schedule limitations stay fail-closed. Scientific formulas and tolerances are
unchanged.

## Evidence

`vibeqc_dft_api_tests` covers v1/v2/v3/v4 preparation, poisoned absent v4 identity,
v3 schedule validation, truncated suffix rejection, existing KS snapshots,
PBE0 composition, and the existing PBE-D4 energy contract. Python KS option and
execution-plan tests cover negotiated serialization and MethodIR projection.

## References

PR #781; issue #396;
[original native-plan decision](../architecture/2026-09-21-native-ks-plan-dispatch.md).

Agent: ChatGPT
Model: GPT-6 Astra Pro

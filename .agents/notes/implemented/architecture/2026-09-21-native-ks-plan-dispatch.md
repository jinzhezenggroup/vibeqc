# Decision: native KS consumes compiler-resolved execution selectors

Status: implementation slice
Date: 2026-09-21

## Problem

The MethodIR frontend and the new KsExecutionPlan already own scientific
composition, but `src/methods/dft_method.cpp` still selected RKS/UKS and the
LDA/PBE/r2SCAN runtime family from `VIBEQC_METHOD_*` IDs. That duplicated
scientific dispatch below the compiler boundary and would force every future
same-physics method to add another native branch.

## Decision

Append a version-3 suffix to `vibeqc_ks_options`. Modern callers carry two
compiler-resolved execution selectors derived from MethodIR:

- `spin_channels`: one for RKS, two for UKS;
- `semilocal_family`: the qualified primitive lowerer family
  (LDA=0, PBE=1, r2SCAN=2).

The existing version-2 semilocal scaling/full-range-K coefficients remain the
scientific coefficients for currently qualified global-hybrid execution.

Native preparation snapshots these fields into one `NativeKsExecutionPlan`.
CUDA XC selection, CPU RKS/UKS dispatch, final-state identity, warm-state shape,
and replay validation consume that plan. The prepared KS owners no longer retain
or inspect the public method ID.

Old v1/v2 C callers retain one narrow `legacy_ks_execution_plan` adapter. It is
an ABI compatibility projection, not the modern scientific dispatch path. The
public method ID still validates that a v3 plan has not changed the selector's
spin/family contract; custom compositions inside the selected primitive family
remain supported.

## Consequences

The native runtime no longer contains `is_uks`, `is_pbe_family`,
`is_pbe0`, `is_r2scan`, `functional_code(method)`,
`display_method_name(method)`, or `is_supported_dft` execution helpers.

This slice does not add a new semilocal family or promote RSH/VV10. The next
extension is to carry the remaining KsExecutionPlan contribution records
(range-separated exchange and nonlocal correlation) through the same versioned
descriptor/owner instead of adding named-method entry points.

Refs #396, #781, #167, #491.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Superseding ABI integration

The execution-plan suffix now uses v4, because mainline v3 already owns the XC
execution schedule. The original rationale above is retained as history; see
[the append-only compatibility decision](../compatibility/2026-09-21-ks-v4-append-only.md)
for the current numbering, complete v3 prefix preservation, and D4 integration.

Agent: ChatGPT
Model: GPT-6 Astra Pro

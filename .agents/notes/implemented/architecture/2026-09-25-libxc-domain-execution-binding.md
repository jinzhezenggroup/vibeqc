# Decision: bind Libxc production-domain receipts to execution identity

Status: implemented
Date: 2026-09-25

## Problem

The first exact production-domain receipt bound the functional capability
identity and versioned boundary profile, but it did not record which executable
mathematical program produced the candidate E/vxc/fxc values. That was adequate
for schema plumbing but insufficient before expanding the candidate domain: a
qualification-only Graph or stale execution path could otherwise produce a pass
receipt whose hash did not distinguish it from a different runtime expression.

## Decision

Advance the receipt to `vibeqc.libxc-production-domain-result.v2` and require a
content-addressed `vibeqc.libxc-production-domain-execution/v1` binding.

For both exact spin layouts, record:

- executor kind;
- runtime domain;
- imported source identity;
- expression identity;
- optimization mode;
- feature ABI; and
- the complete order-2 energy/vxc/fxc output contract.

The campaign builds each spin program once, reuses it for every matrix row, and
binds those same programs into the receipt. The current executor is named
`bulk-runtime-array-graph/v1`; this is intentionally not a compiled-CPU or
CUDA claim.

## Rejected alternatives

- Encoding execution details only in a free-form evidence URL would not make
  tampering or stale execution semantics part of the receipt hash.
- Treating the functional capability identity as execution identity would
  conflate imported source provenance with derivative outputs, optimization,
  feature ABI, and runtime-domain policy.
- Calling interpreted Graph evidence `compiled-cpu` would collapse independent
  capability stages and hide missing AOT/runtime qualification.

## Invariants

- A production-domain receipt names the exact mathematical execution used for
  candidate values.
- Both spin layouts use complete order-2 outputs.
- Execution binding and matrix cases are covered by one content-addressed receipt.
- Backend compilation/device execution remain separate evidence stages.
- Changing runtime domain or expression identity invalidates the receipt.

## Evidence

Regression coverage requires both spin programs and complete E/vxc/fxc outputs,
locks the v2 result/execution schemas, and rejects tampering with an expression
identity. Existing partial/tampered matrix gates continue to apply after adding
the execution binding.

Repository CI is the executable authority for the stacked branch.

## Consequences

Subsequent boundary-policy experiments can use a distinct versioned runtime
domain without accidentally transferring their numerical pass to a different
execution program. A later compiled-CPU/CUDA evidence producer can explicitly
cross-check its artifact expression identity against the retained
production-domain execution binding.

## Revisit when

The production-domain campaign switches to a compiled executor or the capability
registry gains a common cross-stage executable-identity object.

## References

- #1118
- #1120
- #1280
- #1315
- #1320

Agent: ChatGPT
Model: GPT-5.6 Sol

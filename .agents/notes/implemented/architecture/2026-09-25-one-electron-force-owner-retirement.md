# Decision: Retire the native one-electron CUDA force owner

Status: implemented
Date: 2026-09-25

## Problem

The generated nucleus-cooperative S/T/V derivative path is already the production
default and has qualification across the intended Direct/DF RHF/UHF domain, while
the retained cooperative native force path and `VIBEQC_ONE_ELECTRON_DERIVATIVES`
selector duplicate scientific ownership. The coordinate-wise Dual derivative
export is a different boundary: it remains useful as an explicit validation
oracle and does not need to remain selectable by production force execution.

## Decision

Remove the native cooperative one-electron force launch/workspace and the
`VIBEQC_ONE_ELECTRON_DERIVATIVES` production selector. Compiler-generated S/T/V
first derivatives plus the generated weighted-force contraction become the sole
CUDA production owner for this response term.

Keep `VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING` only as execution-schedule
diagnostics. In particular, `serial` is a deterministic generated scheduling
baseline, not a second scientific implementation and not an independent
correctness oracle.

Retain coordinate-wise Dual derivative export plus CPU/libcint/PySCF numerical
checks as independent validation boundaries. Reclassify surviving native files
as oracle code only when they are not reachable from normal production force
selection.

## Rejected alternatives

- Retain the native route as a production performance control: rejected because
  it preserves a duplicate scientific owner after the generated owner became
  the qualified default.
- Treat the generated `serial` schedule as an independent numerical reference:
  rejected because it changes scheduling only and shares the same generated
  mathematics.
- Delete the raw Dual derivative export together with the production route:
  rejected because that would remove a genuinely separate validation boundary.

## Invariants

- S/T/V mathematics, signs, density/weighted-density weights, and the generated
  default mapping are unchanged by this retirement.
- Direct and DF production force paths do not silently fall back to the retired
  native implementation.
- CPU/libcint/PySCF and raw derivative checks remain independent of generated
  scheduling.
- Prepared-plan and resource identities continue to encode the active force
  mode/schedule; removing the legacy provider selector does not create a second
  hidden execution policy.
- Generated schedule A/B benchmarks are performance diagnostics only and cannot
  establish scientific correctness by agreement with each other.

## Evidence

- PR #693 qualified the generated nucleus-cooperative path across the retained
  RHF/UHF, Direct/DF, Cartesian/spherical and size holdouts described in the
  existing one-electron schedule note.
- `.agents/notes/implemented/performance/2026-09-22-one-electron-derivative-schedule-contract.md`
  records the generated scheduling contract and the prior retirement criterion.
- PR #1268 keeps independent force and raw-derivative validation while removing
  the legacy production selector and native force consumer.
- Repository CI remains authoritative for the resulting source head; this note
  does not relabel historical measurements as current-head execution evidence.

## Consequences

Production has one scientific CUDA owner for one-electron S/T/V force response.
Historical native performance-control behavior remains reproducible from Git
history rather than through a live provider selector. Schedule diagnostics
remain available through the generated mapping policy.

## Revisit when

If independent evidence finds a scientific failure in a supported generated
domain, repair the generated/scientific owner rather than silently restoring the
legacy fallback. Address future performance regressions through generated
schedule/profile policy and current endpoint evidence.

## References

Issues #357 and #670; PRs #693 and #1268.

Agent: ChatGPT
Model: GPT-5.6 Sol

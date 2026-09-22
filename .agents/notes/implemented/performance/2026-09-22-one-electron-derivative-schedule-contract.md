# Decision: Bind one-electron cooperative derivatives to shared compiler scheduling

Status: implemented
Date: 2026-09-22

## Problem

PR #693 fixed the 768-AO regression and promoted generated nucleus-cooperative
one-electron derivatives, but CooperativeLaneSchedule still exposed only a
domain-local identity. TensorIR and DFT already used the shared #833/#874
schedule/resource/provenance vocabulary, while this integral consumer could not
participate in the same profile diagnostics. The retained native control also
remained reachable for any unrecognized source-selection string.

## Decision

Project CooperativeLaneSchedule into ScheduleContract without moving
scientific legality into generic compiler code. The one-electron policy emits a
shared WorkloadSignature, one workload profile key, and target-specific
schedule contracts for every supported CUDA architecture. No AO count, molecule
name, or GPU product-name threshold is introduced.
Unknown resource/profitability facts stay unknown; target limits are not
misreported as measured schedule costs. Future tuning under #597 can add general
workload/profile evidence through this contract rather than building a
one-electron-local profile database.

The retained native cooperative implementation is an explicit diagnostic
oracle/performance control only. reference, native, or 0 selects it. The
existing tensor control remains an explicit non-generated DF comparison route
(and maps to the native control for Direct HF). Unknown or misspelled
source-selection values remain on the generated compiler-owned path and never
silently fall back to handwritten ownership.

## Rejected alternatives

- Keep the integral schedule identity outside shared ScheduleIR diagnostics:
  rejected because it would preserve a third tuning/provenance vocabulary.
- Introduce an exact 384/768-AO threshold: rejected because #693 measured the
  same direction from 29 through 768 AO and the threshold would encode a
  benchmark identity rather than a reusable workload feature.
- Treat the native implementation as an automatic fallback: rejected because
  generated CUDA errors must remain visible and scientific ownership must not
  change because of an invalid selector string.

## Invariants

- Generated S/T/V mathematics and derivative signs/weights remain unchanged.
- Schedule/profile selection is execution policy, separate from scientific IR
  identity.
- No runtime NVCC/JIT is introduced.
- The native route remains available as an explicit independent oracle/control
  until a separate removal decision retires that diagnostic capability.
- Performance promotion continues to require complete endpoint evidence; the
  shared contract does not itself declare a winner.

## Evidence

- #693: generated nucleus-cooperative endpoint qualification across 29-768 AO,
  RHF/UHF, Cartesian/spherical, Direct/DF and batch holdouts.
- #881: compiler-owned CooperativeLaneSchedule reused by one-electron and DF
  derivative consumers.
- Focused shared-contract tests cover TensorIR, DFT and integral consumers,
  target-specific round trips and stable owner schedule identity.
- Compiler structure and source-policy tests are required by the implementing PR.

## Revisit when

Revisit schedule features when #597 has reproducible target/workload evidence
that the current portable geometry is not best for a compatible device class.
Revisit native-control retirement only when independent validation no longer
requires that oracle and current complete-endpoint evidence covers the intended
production domain.

## References

Issues #670, #669, #682, #459, #597, #833; PRs #693, #874, #881.

Agent: ChatGPT
Model: GPT-5.6 Sol

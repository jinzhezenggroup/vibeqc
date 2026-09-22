# Decision: device algebra under the shared final-state policy

Status: implemented
Date: 2026-09-16

## Problem

CUDA DF returned a retained physical determinant, but its final-state checks and
energy-weighted density W still performed cubic CPU algebra. This affected both
energy-only and force endpoints. Moving only the first candidate's validation
would leave the correction/eigen-provider route with the same hidden CPU work.

## Decision

Keep identity checks, acceptance thresholds and bounded corrections in
`solver/final_state`. Supply device operations for numeric diagnostics, eigen
validation, density projection and W. Empty host F/C arrays represent a borrowed
device frame only when its exact source/model/solve-epoch/determinant identity
matches the existing retained owner. Device status/generation and every retained
D entry are checked before acceptance. Broken provider state aborts selection;
ordinary numerical rejection can enter the existing correction loop.

Physical J/K stays in the existing plan, with F assembled on the validation
stream. Correcting or exporting F downloads it explicitly. C/epsilon is copied
only for reference export, and W only for the existing host force consumer.
Export requests the stronger canonicality gate during selection and reuses its
accepted diagnostics. CPU validation remains independently selectable through
`VIBEQC_DF_REFERENCE_FINAL_VALIDATION=1` and available to intentional CPU callers.

Nine plan-owned matrices, one spectrum and a bounded reduction array serve one
serialized request. Products such as FC and DS are reused within that request.
FDS and SDF are evaluated explicitly because nearly symmetric admitted inputs
do not justify an exact transpose shortcut. Norms use `hypot`; signed sums use
compensated local/final accumulation and pairwise block reduction. All values
remain FP64, and nonfinite/overflowing inputs and products are rejected.

## Rejected alternatives

- A fixed 64 KiB reduction allowance badly restricted tiles in a 16 MiB,
  four-item, 25-AO force test. The observed first test fell from approximately
  82.5 s to 10.5 s after admitting the actual AO-scaled block count, capped at
  128. Reserve a conservative per-packet bound; do not charge the largest
  scratch size to every small plan.
- Downloading C/F to validate, or skipping validation of an already converged
  frame, preserves either the bottleneck or an invalid scientific shortcut.
- A 32-bit diagnostic status left unwritten struct padding in the D2H packet.
  CUDA initcheck detected it even though numerical tests and memcheck passed.
  The full-width status word initializes all packet bytes without adding a
  launch, synchronization or transfer. Final qualification includes initcheck.

## Invariants

Identity mismatch, stale generation/info, nonphysical Fock origin, invalid
occupations, nonfinite products and exhausted correction remain observable.
There is no implicit CPU fallback after a CUDA/provider failure. Energy-only
does not construct W. Forces consume W from the same accepted determinant.
The workspace is charged during tile admission, never borrows graph scratch,
and drains queued work before staging buffers leave scope on exceptional exits.

## Evidence and consequences

The shared analytic/adversarial suite runs against both CPU and CUDA providers.
Retained-frame tests compare all diagnostics and W with the independent CPU
reference; molecular tests compare the actual force consumer's W and complete
forces with PySCF. Resource, spin, representation, correction and direct-SCF
regressions accompany sanitizer checks.

[Reviewed evidence](../../../../benchmarks/results/issue408-device-validation/README.md)
retains five interleaved warm samples per arm at 96/192/384/768 AO for both
endpoints, separate host/device attribution, work counts and sampled memory.
The CPU-reference switch provides the causal ablation within one binary;
results are not cross-engine speed claims or cold-solve measurements.

PR #411 review exposed CuMetal build constraints absent from the NVIDIA build:
`size_t` and `uint64_t` differ on macOS, and shared reduction packets cannot have
implicit member initialization. The follow-up splits identity/index declarations,
checks references before converting to `size_t`, and keeps reduction packets
trivial with explicitly zero-initialized local accumulators. Every lane fills its
shared packet before the first barrier; the full-width status still avoids
unwritten transfer padding. The CPU electron target is multiplied in the trace
accumulator's `long double` precision. These fixes do not change the archived
measurement source patches or claim new timings for the corrected build.

Revisit the nine-matrix workspace if very large AO plans require more tiling,
or if force consumers accept device W. Preserve the shared policy and explicit
error/identity boundaries in either case.

## References

- Issues #408 and #311.
- [Current contract](../../../../docs/developer/fock_build.md#cuda-df-final-state-validation).

# Decision: Bound CUDA RKS host synchronization with device-controlled iteration chunks

Status: implemented; opt-in only
Date: 2026-09-20

## Problem

The native CUDA LDA/PBE RKS/UKS path already kept density, J, XC potential,
Fock, residual, DIIS history, eigensystem and proposed density on device, but
every physical SCF iteration still copied a scalar record to the host and
called \`cudaStreamSynchronize\`. The host then decided convergence before
submitting the next iteration.

## Decision

Qualify a bounded device-control path for direct all-electron RKS only. An
explicitly selected solve may submit at most two physical iterations before the
host reads compact scalar history and control state. The device owns the
unchanged RKS convergence gates and density/warm-state advancement.

The established one-iteration host-controlled route remains the production
default. \`VIBEQC_CUDA_KS_CHUNK=1\` selects that exact baseline;
\`VIBEQC_CUDA_KS_CHUNK=2\` selects bounded device control for eligible
all-electron RKS. UKS and ECP RKS retain the host path; current ECP RKS requires
the strict physical final closure introduced by #586.

## Rejected alternatives

CUDA Graph replay is not a correctness dependency. Unbounded device loops and
chunks wider than two were rejected because direct J/XC cannot yet consume the
iteration-active mask.

Automatic promotion was rejected by complete endpoint timing on RTX 5090 /
CUDA 12.9. LDA was effectively flat, while PBE cold execution was slower with
two-slot submission. There is no reproducible complete-endpoint benefit to
justify changing the default.

Device-controlled UKS was also not promoted. Its prototype did not pass the OH
stationary-occupation qualification. The same OH test also fails on the
unmodified master build on this RTX 5090, so this hardware run cannot establish
a clean UKS equivalence result; retaining the established host path is the
conservative boundary.

## Invariants

- Direct all-electron RKS convergence gates are unchanged.
- \`VIBEQC_CUDA_KS_CHUNK=1\` uses the pre-existing host convergence policy.
- UKS retains host occupation stabilization, DIIS history and bounded final closure.
- ECP RKS retains the #586 host-controlled strict physical final closure.
- Failure and warm-state publication remain per owner/item.
- A masked terminal RKS slot cannot advance physical Fock, DIIS history,
  density, warm state, final-state generation or compact iteration history.
- Final-state canonicalization uses a permanent export mask independent of the
  terminal RKS iteration-control mask.
- CUDA Graphs are optional and not part of correctness.

## Evidence

CUDA 12.9 / sm_120 builds completed for the qualified #370 implementation. The
native qualification block explicitly selects chunk width two for LDA/PBE RKS
and covers CPU endpoint comparison, independent physical-state rebuild,
cold/warm replay, changed geometry, failure recovery, final-state export and
resource accounting. Two physical H2 iterations use one iteration
synchronization on the selected route.

A separate ragged public-batch gate used H2 and water. Both selectors produced
iterations \`[2, 9]\` and identical energies for LDA and PBE. Injecting invalid
H2 coordinates produced per-item statuses \`[failure, success]\`; the water
survivor completed normally in two warm iterations under both selectors.

Complete energy-only batch A/B on RTX 5090 / CUDA 12.9 used two slightly
different STO-3G water geometries and the same compact native-grid
prescription. Chunk 1 versus chunk 2 medians were:

- LDA cold: 3393.1406 vs 3391.8667 ms (1.0004x); warm: 754.2029 vs
  754.2936 ms (0.9999x); changed geometry: 2907.7366 vs 2907.3763 ms
  (1.0001x).
- PBE cold: 3881.5975 vs 4308.7351 ms (0.9009x); warm: 862.1333 vs
  863.5486 ms (0.9984x); changed geometry: 3290.4014 vs 3288.7629 ms
  (1.0005x).

Every A/B endpoint had zero energy difference and identical iteration tuples:
cold \`[9, 9]\`, warm \`[2, 2]\`, changed geometry \`[7, 7]\`. The timing gate
therefore rejects automatic promotion.

The full native suite subsequently reaches the pre-existing OH
stationary-occupation failure on this RTX 5090. A separately rebuilt,
unmodified master binary fails the same OH test, so that platform-specific
baseline failure is not attributed to #370 and is not hidden by weakening the
test.

## Consequences

Explicitly selected direct all-electron RKS workloads can amortize the host
fence across two physical iterations without changing the public result ABI.
The measured endpoint gate does not justify enabling the path by default.
Direct J/XC may execute one bounded unused evaluation when an unexpected
terminal state occurs in the first slot. UKS and ECP RKS receive no
synchronization reduction in this slice.

## Revisit when

Re-evaluate automatic selection only after new complete endpoint measurements
show a reproducible benefit, especially after direct J/XC accept the device
active mask. Reconsider UKS or ECP only when device control models their
final-closure semantics and passes complete physical endpoints unchanged.

## References

- Issue #370
- Issue #168
- Issue #586
- \`src/dft/cuda_ks.cpp\`
- \`src/dft/cuda_ks_kernels.cu\`
- \`tests/native/test_ks_cuda.cpp\`
- \`docs/ks_diagnostics.md\`

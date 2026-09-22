# Decision: CUDA KS convergence also satisfies the export residual norm

Status: implemented
Date: 2026-09-23

## Problem

The prepared ordinary solver changed the convergence trajectory of a 24-AO
r2SCAN RKS water endpoint. Its energy and RMS residual passed, but final-state
export rejected the maximum AO commutator entry: 1.0051304233371638e-10 against
the unchanged 1e-10 density tolerance. Density reconstruction RMS was
1.4621521878477439e-11; trace, idempotency and canonicality were within their
existing gates. The difference was the norm used for stopping, not a failed
cuSOLVER spectrum or a need to relax scientific acceptance.

## Decision

Accumulate the largest absolute AO commutator entry in the existing scalar
reduction. Both ordinary host-controlled iteration and bounded device
submission require this value to pass the existing residual threshold before
publishing convergence and the last-good seed. Keep the public RMS diagnostics
and all final-state validation thresholds unchanged. The scalar lives in the
existing bounded records; allocator-owned shape queries charge its size.

## Evidence and rejected alternatives

The new independent same-grid PySCF r2SCAN RKS test failed at final-state export
before this fix. Afterward, all six 24-AO PBE/r2SCAN endpoint cases pass,
including RKS two-iteration submissions, both spin layouts, warm replay,
changed geometry and final-state export. Seven public CUDA resource cases,
the independent solver spectrum/residual tests through 768 AOs, and the full
native LDA/PBE/r2SCAN suite also pass on the source-matched RTX 5090 build.

Do not weaken the export gate, reinterpret RMS as a maximum norm, discard the
r2SCAN final-state check, or add a host matrix download to each iteration.
Revisit only if the shared final-state contract or residual representation
changes; the stopping and export norms must remain compatible.

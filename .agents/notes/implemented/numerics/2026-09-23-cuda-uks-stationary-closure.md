# Preserve CUDA UKS stationary occupations through final closure

Status: implemented
Date: 2026-09-23

## Problem

The native OH/STO-3G LDA regression failed on both the original #1058 wheel
and its main integration build. GPU iteration 10 satisfied the unchanged
energy, density-change and physical-residual gates after occupation
stabilization. Entering final closure then cleared that policy: iterations
11–14 alternated occupations with density change 0.235702 despite physical
residuals below 2e-14. The independent CPU solve converged in 10 iterations.
This is the previously retained failure in the
[iteration-chunk qualification](../performance/2026-09-20-cuda-ks-iteration-chunks.md),
not a change in the prepared Fock provider's mathematics.

## Decision and invariants

Match CPU UKS by retaining the established virtual-space proposal shift
through bounded final closure. Closure discards DIIS extrapolation, rebuilds
from physical F[D], and keeps the existing energy, density, residual and
iteration limits. The proposal shift never enters physical energies or Fock
matrices. Cold restarts and precision promotion still reset occupation control.

Energy convergence does not certify an unshifted Aufbau export frame. The
existing final-state validator independently checks the unshifted physical
operator and rejects incompatible orbital/density frames; no shifted frame
is relabeled as a physical frame and no export gate is relaxed. This follows
the distinction documented in the
[CPU UKS export decision](2026-09-22-uks-export-projector-closure.md).

## Evidence and revisit conditions

The native OH regression checks independently solved CPU energy, CPU physical
reconstruction, retained occupation policy, cold/warm replay and restart.
Revisit final-state export separately if stationary non-Aufbau determinants
become part of its production contract; do not weaken the existing validator.

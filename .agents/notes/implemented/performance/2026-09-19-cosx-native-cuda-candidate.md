# Decision: Qualify native CUDA COSX algebra before provider promotion

Status: implemented
Date: 2026-09-19

## Problem

GPU COSX introduces numerical AO collocation, point-charge ESP integrals,
point tiling, K assembly, resource ownership, and eventually provider
selection. Duplicating an ESP recurrence or promoting a partially host-backed
path would make correctness and performance claims ambiguous.

## Decision

Reuse the existing CUDA spatial-grid owner for AO values and immutable packed
basis data. Reuse the generated one-electron nuclear-attraction operator for
ESP: its external center C is already independent, so negating the generated
signed unit-charge attraction gives the positive ESP operator required by
COSX v1. No new Coulomb/Boys recurrence is handwritten.

A bounded point tile owns ESP values O(tile_points * NAO^2), projected density
and potential panels O(tile_points * NAO), and O(NAO^2) K buffers. ESP
generation, density projection, ESP application, accumulation, and
symmetrization all execute on the grid owner's stream.

The candidate remains outside FockBuildSpec until grid/approximation identity,
derivative capability, and complete endpoint resource policy are explicit.

## Invariants

- No global Ngrid x NAO^2 ESP tensor.
- No host ESP tensor or ESP H2D transfer in the native candidate.
- The generated unit-charge attraction is the sole GPU ESP scientific
  recurrence; COSX only changes sign and normalized AO contraction.
- Point order, weights, explicit symmetrization, and spin-energy semantics
  match COSX reference v1.
- No AUTO/provider/gradient capability is granted by this diagnostic path.

## Evidence

vibeqc_cosx_cuda_tests compares raw K, symmetric K, and exchange energy against
the independent CPU discrete oracle across multiple point-tile partitions on
sm_120. Resource assertions verify tile-bounded ESP/AO storage.

## Consequences

The first native ESP kernel is correctness-oriented and evaluates AO pairs
directly rather than applying chain-of-spheres screening or tuned pair/task
schedules. Performance promotion requires later screening/scheduling work and
matched comparison against the qualified RI-K baseline.

## Revisit when

The native candidate has provider identity, explicit COSX grid semantics,
screening policy, and benchmark evidence over the large-system/TZ/QZ domain.

## References

- #246
- docs/cosx_reference.md
- tests/native/test_cosx_cuda.cu
- generated_one_electron_values.cuh

Agent: ChatGPT
Model: GPT-5.6 Sol

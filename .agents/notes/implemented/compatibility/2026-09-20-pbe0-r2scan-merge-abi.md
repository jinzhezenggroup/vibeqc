# Decision: preserve merged r2SCAN interfaces when integrating PBE0

Status: implemented
Date: 2026-09-20

## Problem

PR #618 and the merged r2SCAN registration independently allocated public method
IDs 10/11 and incompatible private `vibeqc_xc_point_batch_v2` signatures. Selecting
one side of these conflicts would silently relabel methods or corrupt the ctypes
call layout. The shared UKS evaluator and final-state model also evolved while
PBE0 introduced explicit semilocal and full-range exchange coefficients.

## Decision

Keep r2SCAN IDs 10/11 and append PBE0 IDs 12/13 in the data manifest. Regenerate
all three native/Python registration artifacts. Preserve the original nine-value
v1 point bridge and the merged eleven-value, tau-aware v2 bridge. Add v3 for
explicit semilocal X/C scales; v2 forwards with unit scales. Reject scaled LDA or
r2SCAN until independently qualified.

Thread immutable X/C scales through the shared RKS/UKS evaluator callbacks and
use the numeric functional family in the final-state identity. Preserve PBE0's
MethodIR/exact-exchange provenance and the ECP snapshot versions (CPU/CUDA v4/v5,
CPU hybrid v6, and hybrid+ECP v7).

## Boundaries and rejected alternatives

Do not hand-edit generated registration outputs or repurpose published IDs.
Do not reinterpret the tau-aware v2 symbol as the old branch's scaled-PBE ABI.
r2SCAN remains energy-only at the public stationary-gradient consumer boundary.
The newly merged native CPKS path remains unscaled CPU LDA/PBE only; PBE0 must
fail closed rather than using a pure-PBE kernel or Coulomb-only response.

## Evidence

`tests/python/test_pbe0_r2scan_integration.py` covers both registrations, native
snapshot family IDs, v1/v2 wire equivalence, scaled-PBE component linearity, and
CPKS rejection of PBE0. Existing native DFT/UKS/final-state tests and Python
stationary-gradient/options regressions cover the shared numerical paths.

Refs #618, #620, #649, #165, #164.

Agent: ChatGPT
Model: GPT-6 Astra Pro

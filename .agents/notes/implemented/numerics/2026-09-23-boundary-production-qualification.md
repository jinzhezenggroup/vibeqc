# Decision: qualify the compiled XC production boundary

Status: implemented
Date: 2026-09-23

## Problem

The proposed boundary gate evaluated bare r2SCAN derivative roots and described
the result as the production entry point. The shared work-MGGA wrapper also owns
density/kinetic floors and empty-spin behavior, so those roots continue to fail
at zero minority density even though the compiled entry point was repaired by
#1054. Importing the AOT generator into pytest also replaced compiler packages
with bootstrap stubs, corrupting later imports.

## Decision

Keep deterministic physical probes separate from the independent Libxc oracle.
The compiled CPU FP64 entry point is qualified against the canonical
`tests/data/xc/r2scan-tail-reference.json` fixture: invoke the AOT CLI in an
isolated process, compile the header, and compare the stabilized zero/near-zero
minority tail points plus spin permutations. Its machine-readable status is
`pass`. The acceptance oracle is the original Libxc 7.0.0 Maple E/vxc algebra
evaluated in 113-bit arithmetic; the retained nine-point binary64 Libxc fixture
is diagnostic only because empty-spin derivatives are cancellation-sensitive.
Bare Graph finiteness is not evidence of production behavior. Generation remains
available without Libxc; oracle regeneration is an explicit development tool.

## Evidence and limits

The separately installed Libxc 7.0.0 C API still reproduces the retained
nine-point binary64 diagnostic by summing independent R2SCAN exchange and
correlation evaluations. Production acceptance instead uses the #1108
high-precision Maple fixture so a stable algebraic repair is not rejected for
disagreeing with Libxc's own cancellation-prone binary64 endpoint. The bulk
oracle tool completes all 221 imported registrations in both spins:
273/273 LDA, 740/742 GGA and 532/532 MGGA probes are finite. The two nonfinite GGA
oracle outputs retain `oracle_finite=false` and `expected=null`; they are not
qualified or coerced into passing values. This run generates 1,547 physical
probes and evaluates no VibeQC Graph when producing the expected numbers.

A fresh-process regression protects compiler package identity during pytest
collection. No production algebra, public-method inventory or capability-stage
admission changes. The CPU fixture is bounded evidence at these inputs; it does
not certify the full bulk inventory, CUDA, molecular endpoints or all tails.

## Revisit when

Add identity-bound capability evidence only after a consumer qualifies the full
required domain and backend. If a production wrapper changes, keep the native
entry point under test rather than reverting to unwrapped derivative roots.

Refs #1040, #1028, #1054, #1065, #1108.

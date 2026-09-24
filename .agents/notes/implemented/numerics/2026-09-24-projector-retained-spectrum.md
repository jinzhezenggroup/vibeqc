# Decision: use the prepared spectrum for fixed-rank projector response

Status: implemented
Date: 2026-09-24

## Problem

Recovering f(lambda) by projecting the dense matrix-function value back into its eigenbasis introduces the conditioning of f(A) into an unrelated projector derivative. A rotated five-level pseudoinverse example with retained condition near 1e12 changes an otherwise order-one cross coefficient by about 3.2e-5. For uniformly large spectra, the parent function derivative can underflow to zero even though the projector derivative remains representable; dividing that zero by f(lambda) silently erases response.

## Decision

Retain an immutable copy of the eigenvalues from the existing, sole validated eigendecomposition. The projector uses only retained/discarded gaps, with the parent's unchanged rank and cutoff decision. Internal retained/retained and discarded/discarded entries remain zero. Full-space projectors validate the seed and return exact zero response without rotating a huge finite seed into irrelevant overflow.

The new optional snapshot field preserves old positional construction. The projector rejects legacy/manual snapshots lacking the prepared spectrum rather than recomputing an eigensystem. Parent matrix-function values, derivative rules and mathematical identity remain unchanged; projector identity advances to v2 because its numerical construction changes. Parent response admission charges the retained eigenvalue array. Projector logical-array admission conservatively covers owned arrays and temporaries, excluding parent/caller storage, Python objects, and opaque BLAS/RSS accounting.

## Rejected alternatives

Do not recover spectral values from the dense function matrix, divide underflowed parent coefficients, add another eigendecomposition, widen the cutoff guard, or freeze projector directions. Correct the gauge test's accidentally unresolved cutoff fixture while retaining an explicit rejection test for the original boundary.

## Evidence

The exact modified source and exact parent baseline were tested with retained compiler dependencies on CPU: 18 projector tests passed, including independent eigenvalue-only coefficients, multistep recomputed-projector finite differences, degenerate subspaces, JVP/VJP duality, budgets and immutable state. Before repair, the original fixture and new failure regressions produced nine failures; one is the newly required stored-spectrum contract, not an independent mathematical defect. A separate 18-case parent comparison found unchanged identities, values, eigenvectors, divided differences and JVPs. These are reference CPU checks, not native CUDA or complete local-correlation force qualification.

## Revisit when

A native spectral-state owner replaces this CPU reference, or the projector domain expands beyond locally fixed retained membership. Preserve shared branch decisions and an independent response oracle.

Refs #1205, #185, #466.

Agent: ChatGPT (Odd-PR Review)
Model: GPT-6 Astra Pro

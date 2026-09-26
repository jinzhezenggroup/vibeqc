# Decision: Generate fixed-input standard-(T) response from the audited primal

Status: implemented
Date: 2026-09-20

## Problem

The perturbative-triples correction needs amplitude, integral and orbital-energy
response sources without creating a second handwritten derivative equation set.
A bounded tile depends on a virtual prefix that overlaps other tiles' inputs.
Treating those overlapping inputs as disjoint would lose derivative contributions.

## Decision

Use the shared TensorIR transpose generator on the existing standard closed-shell
(T) energy programs. Retain the primal logical hash in every generated VJP.
`TriplesTileEnumerator` partitions the triangular a>=b>=c energy domain. For
E=sum_t E_t, the input cotangent is sum_t dE_t/dx, even when several E_t consume
the same prefix of x. The scatter therefore adds cotangents into shared input
coordinates; it does not count any energy tile twice.

Orbital energies remain differentiable inputs. Their VJPs include the reciprocal
denominator response; the existing small-denominator rejection still applies.
Common-subexpression elimination and recomputation may change arithmetic work
and storage but never the mathematical multiplicity of the energy partition.

## Rejected alternatives and validation

Do not hand-code an additional triples derivative algebra or freeze denominators.
Do not overwrite overlapping prefix cotangents or confuse bounded intermediates
with bounded full response outputs. The frontend retains dense selected-input
cotangent arrays; it is not a globally memory-bounded molecular gradient.

The tests compare tiled and untiled VJPs and independently perturb each of the
eight primal input families, recomputing the original energy at two step sizes.
The latter check does not compare two implementations of the same new derivative.
Occupied rank two avoids the identically zero rank-one triples case. Separate
checks retain nonzero denominator response and optimized/unoptimized equality.
CUDA plannability is not real-device numerical or complete endpoint qualification.

## Scope and revisit criteria

This slice provides fixed-input response sources only. Corrected Lambda,
orbital response, nuclear integral response, and public CCSD(T) forces remain
follow-up work under #154. Revisit storage and reuse when binding those consumers;
preserve the same generated-primal and independent finite-difference contracts.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Review qualification result

The independent primal finite differences, tiled/untiled derivatives and optimized
response checks pass in the review run. The CUDA-plannability test does not pass:
for its current case the retained arena is 316,928 bytes and the recomputed arena
is 745,728 bytes. Per-output recomputation is only a schedule candidate; duplicated
index tables, pinned outputs and allocator fragmentation can defeat the expected
memory reduction. The failed assertion remains in the tests, and this note does
not relabel that candidate as a memory win or authorize its production selection.

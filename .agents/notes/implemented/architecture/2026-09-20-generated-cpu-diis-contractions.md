# Decision: generate CPU DIIS contractions from canonical TensorIR

Status: implemented in PR #712; integration remains gated by required CI
Date: 2026-09-20

## Problem

The CPU DIIS solver duplicated the residual-history Gram contraction and the
coefficient-weighted Fock-history contraction already expressed by canonical SCF
TensorIR programs. Scientific ownership should not depend on the executor.

## Decision and ownership

`tools/generate_scf_array_native.py` emits checked native specializations of
`diis_gram_program` and `diis_extrapolation_program`. `solver::Diis::update()`
consumes them instead of maintaining independent dot and extrapolation loops.

The generator checks output names, unit einsum coefficients, index labels,
output domains, operand identity/order, and input layouts before emission.
Shape-independent TensorIR template hashes are embedded in the generated header.
This is a constrained lowering of two admitted equations, not a general-purpose
native TensorIR backend.

Compiler-owned science is the residual Gram matrix and the coefficient-weighted
Fock-history extrapolation. Native ownership remains chronological history
insertion/eviction, capacity, optional metric normalization, augmented-system
construction, pivoted solve, singular-history retirement/retry, fallback,
allocation, resource accounting, and lifetime.

CUDA-resident DIIS is unchanged and requires separate qualification before any
future cutover. Iteration-through-SCF differentiation is not introduced.

## Rejected alternatives

Moving the entire DIIS state machine into TensorIR would mix history mutation
and solver policy with two pure contractions. Keeping unchecked C++ templates
with decorative equation hashes would let changed TensorIR silently retain an
obsolete native specialization; fail-closed topology/layout checks prevent this.

## Invariants

Preserve FP64 and historical reduction order: increasing element order in each
Gram dot and increasing chronological history order in each extrapolated Fock
element. Preserve the augmented matrix stride and constraint row/column. No
convergence tolerance, pivot threshold, normalization, history-retirement policy,
or fallback decision changes.

## Evidence

The repair review ran `tests/python/test_scf_array_native_codegen.py`: seven tests
passed, including shape-independent identities, embedded hashes, and rejection
of changed DIIS topology. The separate Krylov lint repair passed all 32 tests in
`tests/python/test_response_krylov.py` on its exact source tree.

Required native build and SCF endpoint CI remain the integration gate. These
Python tests are not claimed as full native numerical or GPU qualification.
No measured performance improvement is claimed.

## Revisit when

A DIIS equation/layout, reduction order, spin/batch admission, or CUDA ownership
changes. Update the checked specialization and independent tests rather than
bypassing the generator checks.

## References

PR #712; issues #181 and #349; `docs/array_api_frontend.md`;
`tests/python/test_scf_array_native_codegen.py`.

Agent: ChatGPT
Model: GPT-6 Astra Pro

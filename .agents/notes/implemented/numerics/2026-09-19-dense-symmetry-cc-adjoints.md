# Decision: dense symmetry projection for generated CC adjoints

Status: implemented
Date: 2026-09-19

## Problem

The existing RCCSD residual DAG declares simultaneous pair symmetry on T2.
Generated AD previously required explicit packed expansion for any symmetric
parameter. Its gather transpose uses a coordinate-incidence matrix, making the
packing boundary quadratic in amplitude count even when the physical VJP is
matrix-free. A direct unweighted packed transpose also loses orbit factors.

## Decision

Add generic dense-symmetry support to the existing TensorIR derivative generator.
Forward seeds obey the declared input symmetry; reverse results average the full
signed permutation group under the dense Frobenius metric. Generate only normal
transpose/add nodes with exact rational factors. Cap the symbolic group at 4096
signed permutations; tensor extent does not determine the number of terms.

The RCCSD Lambda frontend reuses this boundary and the existing physical
energy/residual definitions. The shared/expanded/optimized primal alternatives
remain independently usable. It supplies equation actions only, leaving solved
state validation and the common implicit-solve callback to their own consumer.

## Rejected alternatives

- Dense T2 packing-incidence matrices: unacceptable amplitude-quadratic storage.
- Handwritten CC Lambda equations or differentiated Jacobi/DIIS iterations:
  duplicate scientific mathematics or differentiate the wrong equations.
- Projecting onto generators sequentially: not the orthogonal group projector
  when generators do not commute.
- Unweighted packed dot products: wrong off-diagonal orbit multiplicities.

## Invariants and evidence

The generic group projector agrees with the existing independent orbit-enumerator
adjoint for symmetric, antisymmetric, RCCSD-pair, noncommuting and structural-zero
spaces. CC tests compare generated RHS/JVP/VJP with a determinant-space oracle,
three-step directional differences and a tiny numerical-Jacobian transpose.
A negative assertion detects missing packed weights. Existing packed AD behavior
is retained. Production code constructs neither a CC Jacobian nor an iteration
tape; the numerical matrix appears only in the tests.

The shared GMRES test establishes action interoperability at fixed amplitudes,
not converged molecular Lambda or force support. CUDA planning is not actual
GPU execution evidence. No speedup or complete-process memory claim follows.

## Revisit when

A scalable native signed pack/scatter primitive is available, or the common
implicit-solve consumer requires a resident packed rather than dense state.
Keep metric/transpose tests while changing storage or execution policy.

## References

Refs #151, #152, #465; see `docs/rccsd_lambda.md` and the focused tests.

Agent: ChatGPT
Model: GPT-6 Astra Pro

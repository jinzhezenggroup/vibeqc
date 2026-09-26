# Decision: bind CPU RCCSD Lambda through the shared checked-adjoint boundary

Status: implemented
Date: 2026-09-19

## Problem

#483 provides generated fixed-amplitude RCCSD derivatives but does not prove
that the amplitudes and integral/reference inputs describe a current converged
state. #495 supplies generic implicit differentiation and an opaque solver
contract. Its single independent tensor input cannot accept redundant dense
T2 merely by flattening its storage: that loses the simultaneous-pair metric.
Expressing a complete packing map with current TensorIR gather nodes instead
causes the generated reverse to construct dense gather-incidence matrices.
That scales quadratically with the amplitude dimension, violating the intended
matrix-free response boundary.

## Decision

Keep #483's already audited dense-symmetry derivative programs. Use the
existing PackedLayout maps only at the CPU tooling boundary, with sqrt-orbit
weights converting the full Frobenius metric into independent Euclidean
solver coordinates. Extract the true-residual, solver-contract and failure
acceptance from BoundImplicitState into checked_transpose_solve, which is used
by both BoundImplicitState and BoundCCSDLambda. Reuse ResponseGMRES and #179's
unchanged implementation. This is common adjoint orchestration, not automatic
MethodIR dispatch of CC through ImplicitSolveSpec.

Bind a successful CCSDResult to the exact validated reference, Hamiltonian,
primal equation identities, Fock and integral feed hash, and copied amplitudes.
Reconstruct and check the replay reference; compare Fock blocks to the supplied
reference directly. Recompute physical R1/R2 and energy rather than trust the
result status or history. Freeze retained inputs and outputs with bytes backing.
A live owner callback is optional and explicitly distinguished from detached
immutable-snapshot semantics.

The shared solver gate recomputes its true residual. The CC consumer separately
rechecks stationarity through expanded derivative programs and enforces its
own numerical tolerance. Neither success flags nor an injected loose solver
tolerance can authorize a false successful response. No response is published
on stale state, nonfinite data, nonconvergence or failed admission.

## Rejected alternatives

- A second handwritten CC Lambda/DIIS solver: duplicates both equations and
  #179's numerical/failure machinery and frustrates compiler-first ownership.
- A production dense Jacobian or packed incidence matrix: costs amplitude-
  quadratic storage for a boundary that already has generated matrix-free AD.
- Relaxing ImplicitSolveSpec's independent-coordinate requirement: hides a
  singular redundant state and invalidates the primitive's stated assumptions.
- Claiming a complete #152 B/GPU/force implementation from CPU equations or
  planning alone: native residency, full resource ownership and downstream
  source-weight/nuclear response still require separate acceptance.

## Invariants

Use L = E_corr + <lambda,R> and the physical unpreconditioned CC residual.
Preserve simultaneous T2 pair symmetry and orbit weights. Do not replace the
actual CC result with fixture amplitudes, invoke external QC in the consumer,
record a cross-iteration AD tape, or publish parameter weights/RDMs/forces here.
The extracted _ccsd_programs helper preserves the original primal and expanded
logical hashes; it centralizes the exact state-admission identity rather than
copying those definitions into the response consumer.

## Evidence

`tests/python/test_cc_lambda_solver.py` exercises solved H2, H2O and NH3 roots,
test-only numerical-Jacobian comparisons, three-step stationary Lagrangian
differences, the nontrivial doubles orbit metric, immutable state, stale/corrupt
reference/equation/integral data, false convergence, independent gates and
adversarial/failed solver results. Native HF -> CC -> Lambda tests include a
changed H2 geometry and a determinant-space numerical reference. Existing
implicit-VJP and CC equation/solver regressions protect the shared extraction.
See the PR validation record for exact executed source/library identities and
commands; older native-library results are not final-tree qualification.

## Consequences

This delivers a CPU molecular amplitude-response slice with logical numeric
admission; it does not close #152. Python objects, externally retained replay
lists and opaque NumPy/BLAS workspace are not part of a process-RSS claim.
GPU resident ownership, parameter-weight generation and physical RDM/nuclear
response remain distinct. The #495 dependency is explicit while unmerged.

## Revisit when

The generic method boundary supports native block-state coordinates with a
linear-memory gather/scatter or projected independent map and composed
provider/Krylov resource ownership. Migrate this thin adapter at that point;
retain its identity, residual, convention and failure tests as acceptance gates.

## References

#152, #483, #465/#495, #179, #151; `docs/rccsd_lambda.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro

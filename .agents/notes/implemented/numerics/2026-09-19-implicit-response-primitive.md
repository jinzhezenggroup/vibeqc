# Decision: Generate implicit adjoints around the shared response solver

Status: implemented
Date: 2026-09-19

## Problem

Issue #465 requires a reusable first-order implicit-solve primitive for #193 C2
and other response consumers. TensorIR already differentiates explicit residual
equations; #179 already owns bounded GMRES. Writing another MP2-specific
Z-vector or differentiating SCF/GMRES iteration history would duplicate the wrong
layer. Inspection also showed that #179's Krylov controller is Python tooling,
although its molecular J/K operators have native backends. Calling the complete
controller native would misrepresent the execution boundary.

## Decision

Keep the typed residual/derivative contract in `vibeqc_compiler.method`.
Generate the mathematical stages with #151 `linearize`/`transpose_program` and
the existing graph rebuilder. Runtime tooling binds immutable state and an
opaque solver/executor contract. `ResponseGMRES` invokes #179 unchanged.
`PreparedImplicitCuda` composes the existing native TensorIR resource owners;
it introduces no CUDA kernels or numerical solver implementation.

For state/residual metrics Wx/Wr, use square-root-weighted Euclidean solver
coordinates. The compiler emits the transpose, metric factors, negative RHS
and direct-plus-implicit source terms. Parameter weights remain Euclidean.
The negative-adjoint convention is `Jx* lambda = -gx`; the parameter response
therefore adds `Rq.T Wr lambda` rather than subtracting it a second time.

Recompute the true transpose residual independently after the callback returns.
Failed solves, incorrect success reports, changing live reference identities,
backend changes and resource failures cannot publish derivative weights.

## Rejected alternatives

- Another hand-coded MP2 Z-vector or another GMRES: contradicts the shared
  compiler/solver boundary and repeats already implemented mathematics.
- Differentiating solver history: grants derivatives of an iteration algorithm,
  not the declared converged equation, and grows retained state with iterations.
- Ordinary unweighted packed transposition: loses orbit multiplicities.
- Dense Jacobian assembly in the primitive: unnecessary and unbounded for its
  intended consumers. Dense matrices appear only in small independent tests.
- Calling CUDA planning or host-controlled GMRES fully native: not supported by
  the code. Actual device contraction evidence is labeled separately.

## Invariants

Identity includes the residual, independent layout/gauge, metric conventions,
operator and first-order/solver ABI. Runtime identity additionally includes the
bound reference/data and callback choices. Replay regenerates derivative graphs.
No compiler import reaches into runtime/tooling or probes a GPU. The primitive
assumes a differentiable, locally invertible branch; it is not a general
conditioning certificate. Higher implicit derivatives need their own rule.

Reference admission includes simultaneously live state/source arrays and solver
storage, but excludes opaque host BLAS scratch and Python objects. Native stages
use the unchanged provider allowances and global budget. Neither logical-byte
admission nor these focused tests establish full process or molecular peak memory.

## Evidence

Independent tests cover scalar roots, nonlinear nonsymmetric systems, weighted
dot identities, existing packed orbit weights, re-solved finite differences,
upstream objective cotangents including direct terms, stale-state and adversarial
callback/resource failures. A rank-one residual checks O(n) generated tensor
storage independently of iteration count.

H2 and water reproduce #293's existing MP2 Z-vector with a compiler-generated
transpose of the primal RHF density-response action. Initial use of
`MP2OrbitalRHS.energy_gradient` did not reproduce water: it predates the same-space
canonical orbital-response elimination. The correct reduced seed is
`-MP2OrbitalRHS.response_rhs`. The independent oracle remains unchanged, and the
regression preserves that distinction rather than weakening the comparison.

Allocated RTX 5090 / sm_120 execution uses NVCC 12.9.86 and the existing CUDA 12.4
cuBLAS headers/libraries. The tests disable the CPU interpreter, check the same
mathematical plan, exercise changed-state and failure/recovery, and sum retained
provider measurements under the common admission. The Krylov loop and explicit
staging remain host-controlled; no performance claim is made.

Verification ledger for this implementation:

- The new CPU primitive/physical-response tests pass: **69 passed in 1.80 s**.
- The combined implicit, TensorIR AD, shared response, matrix-function, MethodIR
  and compiler-structure regression passes: **240 passed in 35.06 s**. Existing
  native-operator tests use an unchanged existing CPU library with SHA-256
  `e1cf0db23e86b3b0a6116abb8162fe05874b7c64392f4f5395e477bbf845e1fb`;
  this is not a new native-library build claim. An earlier broad run against an
  older library skipped eight native tests; the final run above executes them.
- **3 real-device tests passed in 20.48 s on the earlier implementation snapshot.**
  All six current generated CUDA sources match those device-tested artifacts.
  Subsequent host-state freezing, solver-contract serialization and test additions
  were checked on CPU. The final four-case device rerun could not obtain the
  occupied GPU; its queued job was cancelled. Final-head device qualification is
  still pending and is not replaced by source equality or the earlier pass.
- Compiler structure: 182 modules, zero errors. SCF structure: 217 modules,
  zero errors. The 183-file CUDA ownership inventory, evidence-retention checks,
  Ruff lint/format and whitespace checks pass.

## Consequences and revisit conditions

The compiler primitive is reusable now, but the runtime adapter remains checkout
tooling. Integrate native molecular state/providers and method custom-rule dispatch
without changing these semantics. Revisit the host callback when #179 provides a
qualified native resident solver; do not implement a parallel private solver in
this primitive. #193 public forces and full C2 resource/gradient acceptance remain
separate work. Arbitrary higher derivatives are not authorized by this first rule.

## References

#465, #193, #179, #151, #293; [current contract](../../../../docs/developer/implicit_response.md).

Agent: ChatGPT
Model: GPT-6 Astra Pro

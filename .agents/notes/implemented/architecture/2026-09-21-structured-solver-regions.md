# Decision: structured bounded solver regions above ProgramIR

Status: implemented
Date: 2026-09-21

## Problem

ProgramIR v2 deliberately models finite serial SSA calls and rejects in-place or
cyclic state. Electronic-structure solvers nevertheless need compiler-visible
bounded iteration state for replay, lifetime planning and eventual capture/device
control. Extending TensorIR with SCF/CC policy or differentiating solver history
would violate the existing scientific ownership boundary.

## Decision

Add `vibeqc_compiler.common.solver_region.SolverRegion` as a data-only wrapper
above an ordinary ProgramIR body instead of changing ProgramIR v2.

The contract has:

- immutable invariant body inputs;
- explicit SSA loop carries from a current body input to a distinct next body
  output;
- a positive finite `max_steps`;
- identified convergence and optional failure predicates;
- scalar or explicit per-item-mask completion semantics;
- named checkpoints whose host visibility is explicit; and
- an explicit `unsupported` or `custom` derivative policy, with custom
  first-order implicit/stationary JVP/VJP rule identities.

The region itself does not execute a loop. Runtime/controller owners keep
convergence thresholds, DIIS/mixing, failure handling and state publication.
Unregistered derivatives fail closed; no solver-history tape is generated.

The first existing consumer is the conventional RCCSD CPU solver. Its
`PreparedCCSD.solver_region` describes the existing equation program, carried
amplitude/DIIS/control state, `max_iterations + 1` evaluation bound, independent
expanded-equation acceptance, nonfinite failure policy and observation points.
The numerical implementation remains the pre-existing Python loop. Solver results
publish the structured-region identity and bound for replay/provenance.

## Rejected alternatives

- **Make ProgramIR v3 cyclic/in-place immediately.** This would weaken the simple
  ownership/lifetime proof used by existing XC consumers and mix two different
  control-flow models before a real execution lowering exists.
- **Put SCF/CC convergence logic in generic compiler code.** Tolerances, DIIS and
  scientific acceptance remain solver/method policy.
- **Differentiate the unrolled iteration history.** Existing stationary/implicit
  derivative machinery owns the correct derivative boundary; iteration count
  must not determine AD tape size.
- **Claim #370-style capture performance in this slice.** The first region is
  descriptive only. Capture/device-controlled execution needs matched endpoint
  evidence and an ordinary fallback before promotion.

## Invariants

- ProgramIR schema v2 and serial SSA validation remain unchanged.
- Every loop-carried dependency is declared explicitly as current-to-next values.
- Every body output has an explicit region role: carry, result, predicate or
  active mask.
- `max_steps`, predicate identities, checkpoints and derivative registrations are
  part of the stable region identity.
- Resource planning reuses one body allocation across steps; it does not multiply
  capacities by the iteration bound.
- Provider-internal scratch/history is excluded unless a consumer exposes it as a
  named boundary owner.
- First-order derivative registration never grants higher-order support.

## Evidence

The focused validation covers strict serialization, identity sensitivity,
state/output role checks, explicit host checkpoints, resource reuse across
different iteration bounds, registered derivative lookup/fail-closed behavior and
the existing RCCSD endpoint with unchanged convergence policy.

Reproduction:

```bash
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_solver_region.py \
  tests/python/test_program_ir.py \
  tests/python/test_cc_solver.py
PYTHONPATH=python:. python tools/check_compiler_structure.py
```

## Consequences

The compiler can now reason about bounded loop identity and boundary lifetimes
without owning solver mathematics. A later executor can lower a qualified region
to host orchestration, CUDA Graph/device control or another backend while retaining
the same logical contract.

This slice intentionally does not prove host-overhead reduction, per-item ragged
execution on a production batch, or a second solver-family integration.

## Revisit when

- #370 or another endpoint is ready to execute a SolverRegion rather than only
  describe it;
- a real batched solver needs `per_item_mask` completion;
- a second solver family is bound and exposes missing generic state semantics; or
- #181/#465 provide an executable derivative-plan identity that a region consumer
  can register directly.

## References

- #835
- #682
- #181
- #465
- #370
- `docs/program_ir.md`

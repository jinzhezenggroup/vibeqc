# Decision: Progressive HF keeps initialization outside target verification

Status: implemented
Date: 2026-09-22

## Problem

A small-basis or loose HF solution can reduce target SCF work, but it also makes
it easy to report the source model, an unrefined mixed state or an uncalibrated
residual as though it satisfied the requested target. Future CC, rank, local and
DFT adapters have different state mathematics and cannot be inferred from the HF
density projection interface.

## Decision

The first #192 controller has an immutable `TargetProblem`, typed `StagePlan`
records and exactly one initialization-to-target edge. Source RHF/UHF is used
only through #189 metric occupied-density projection. Source failure and
projection rejection fall back to the unchanged target calculation. The final
record independently compares scientific identity, prepared-provider identity,
basis provider hashes, solver state, physical residual, requested capabilities,
final arithmetic and work budgets. The immutable target also binds the requested
atom count; force capability requires a complete finite real `(atom_count, 3)`
array rather than trusting the result array to define its own extent.

Because the legacy HF result ABI intentionally leaves its optional physical
residual absent, the controller rebuilds the exact target Fock once at the final
density and evaluates the target overlap commutator. The audit has explicit
operator/density/overlap identities, time and Fock-work accounting; it does not
reuse density-update RMS as a substitute.
For density fitting, an omitted auxiliary basis retains HF's existing resolved
model: the orbital basis is also the auxiliary basis.

Exact target convergence and #173 observable-accuracy evidence are separate
outcomes. The controller may report that the target state is established while
the accuracy status remains `unverified`; only independent observed evidence can
produce overall `verified` status.

## Rejected alternatives

- Treating a converged source or accepted projected density as a target result:
  neither has evaluated the target equations.
- Inferring energy or force accuracy from density change or commutator residual:
  that omits response/conditioning information forbidden by #173.
- Reusing one untyped state-transfer hook for HF and CC: occupied density
  projection does not define virtual orbitals, T1/T2 amplitudes or denominators.
- Allowing an over-budget source stage to consume the target budget silently:
  failed speculation is real endpoint work and remains charged.

## Invariants

- The final stage exactly matches the original model, provider and convergence
  snapshot.
- The target stage produces an FP64 final state; mixed arithmetic must show
  #174 strict refinement.
- A source or transfer failure never changes target physics or tolerances.
- Grid/local/CC/rank adapters require separate typed plans and validation.
- No successful accuracy claim follows from SCF residual alone.
- Force capability must match the target's atom count and real numerical dtype;
  partial, extra-atom or complex arrays are not target observables.
- A missing native HF residual requires an exact fixed-density target audit; the
  extra Fock build and elapsed time remain part of endpoint work.

## Evidence

- `tests/python/test_progressive_controller.py` covers immutable schemas,
  substituted targets, missing strict cleanup, unverified accuracy and budget
  failure, including target-bound force shape and dtype rejection.
- `tests/python/test_progressive_controller_native.py` covers the real HF
  source/projection/target path plus cold fallback after source and projection
  failures.

## Consequences

The first controller is deliberately fixed rather than adaptive. It provides a
reviewable HF baseline and durable final-verification vocabulary, while later
accuracy allocation and method-specific adapters can extend it without changing
the meaning of an already requested target.

## Revisit when

Add another stage type only after its state embedding, invalidation rules,
observable evidence and fallback are independently validated. In particular,
#190 owns CC orbital/amplitude transport and is not implemented here.

## References

- GitHub issues #173, #174, #189, #190 and #192.
- `docs/accuracy.md`
- `docs/basis_projection.md`
- `docs/progressive_hf.md`

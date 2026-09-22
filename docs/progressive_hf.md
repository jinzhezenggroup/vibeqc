# Deterministic progressive HF

VibeQC exposes a bounded two-stage Hartree-Fock controller for using a cheaper
HF calculation only as an initial-state proposal for an explicitly requested
target. The controller never treats the source energy, residual, basis or
provider as the target result.

```python
from vibeqc import (
    Calculator,
    ObservableTarget,
    ProgressiveBudget,
    TargetAccuracy,
    TargetProblem,
    make_deterministic_hf_plan,
    run_progressive_hf,
)

atoms = (("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7)))
accuracy = TargetAccuracy(
    (ObservableTarget("energy", "absolute", "Eh", absolute=1e-8),)
)
source = Calculator(
    basis="sto-3g",
    energy_tolerance=1e-7,
    density_tolerance=1e-6,
    max_iterations=20,
)
target = Calculator(
    basis="def2-svp",
    energy_tolerance=1e-12,
    density_tolerance=1e-10,
    target_accuracy=accuracy,
)

problem = TargetProblem.from_calculator(target, atoms)
plan = make_deterministic_hf_plan(
    problem,
    source,
    target,
    atoms,
    budget=ProgressiveBudget(
        maximum_source_iterations=20,
        maximum_total_iterations=120,
    ),
)
run = run_progressive_hf(plan, source, target, atoms)
print(run.target.energy, run.verification.status)
```

## Immutable target and typed plan

`TargetProblem` snapshots the resolved #173 scientific model, prepared-provider
identity, basis mathematical/source hashes, observable accuracy requirements,
solver tolerances, requested atom count and the final physical-residual gate.
Grid and local-policy identities are explicit fields and must be absent in the
first HF slice. This prevents a later adapter from silently changing those parts
of the target.

`StagePlan` separately records the resolved model, provider, arithmetic policy,
solve tolerance, transfer operation, optional cost estimate, typed #173 error
evidence and allowed successor stages. `DeterministicHFPlan` currently accepts
exactly:

1. an RHF/UHF initialization compatible in geometry, electron count, spin and
   Hamiltonian, followed by #189 metric density projection; and
2. the exact target model/provider with an FP64 final state. `precision="auto"`
   is admitted only when #174 strict refinement is required and later observed.

Different basis or fitted-operator stages are staged warm starts, not a claim of
one continuous homotopy. No orbitals, virtual spaces or correlated amplitudes
are transported by this controller.

## Failure and budget behavior

Stage count, source iterations, total configured iterations, host projection
workspace, optional Fock builds and optional estimated cost have explicit
budgets. A plan whose configured maximum work exceeds its budget is rejected
before execution. Projection and final-verification host work have separate
bounds. Before creating the verification AO/Fock owners, the controller reserves
its dense audit matrices/copies and checks the existing HF provider inventory
against the remaining host capacity. The full HF inventory is a conservative
admission bound for this fixed-density operation; it is not a measured process
or driver-memory peak. An unavailable inventory fails closed. Exhausting either
part returns `budget_exhausted` without starting the verification Fock rebuild.
Failed source work remains in the execution record and total work count.
Diagnostics separately report source/target setup and execution, projection,
final verification and context cleanup, while `total_seconds` covers the complete
endpoint.

A failed source solve or rejected/out-of-budget projection skips the proposal
and executes the unchanged target from its ordinary cold guess. The target stage
is never replaced by a source result. Target failure is returned in
`FinalVerification`; the public runner does not manufacture convergence by
relaxing tolerances.

## Final verification

`FinalVerification` checks the actual resolved target identity, prepared-provider
identity, basis mathematical/source hashes, native success/convergence,
energy/density gates, physical SCF residual, requested energy/force capabilities,
effective final FP64 arithmetic and complete execution budgets. It distinguishes:

A force capability is established only by a finite, real numerical array with
shape exactly `(TargetProblem.atom_count, 3)`. Empty, partial, extra-atom,
complex or nonnumeric arrays fail closed as an omitted requested observable.

Legacy HF results do not export the optional public physical-residual field. The
controller therefore performs one explicit fixed-density rebuild with the exact
target basis, direct/DF operator, screening, metric and arithmetic policy. It
records the Fock execution identity, density/overlap hashes, RMS and maximum
`FDS-SDF`, and fixed-density energy agreement. This audit is included in target
time and counts as one target Fock build; its failure makes target verification
fail closed. It is a state audit, not an observable-error certificate.

- `verified`: exact target gates passed and an independent #173 assessment has
  status `observed_met`;
- `unverified`: the exact target is established, but observable accuracy has no
  independent reference audit;
- `unmet`: a target identity, state, arithmetic, observable or accuracy gate
  failed; and
- `budget_exhausted`: actual work exceeded a bound or a requested work counter
  was unavailable.

`target_established=True` therefore does not imply that an energy or force error
target is certified. A small SCF residual cannot stand in for observable error
evidence. See the [architecture decision](../.agents/notes/implemented/architecture/2026-09-22-progressive-hf-target-boundary.md)
for the rationale and adapter boundary.

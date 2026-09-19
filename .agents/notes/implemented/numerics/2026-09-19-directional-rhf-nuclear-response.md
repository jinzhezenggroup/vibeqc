# Decision: shell-local directional nuclear sources and one shared RHF response

Status: implemented
Date: 2026-09-19

## Problem

#511 supplied the conventional direct-CUDA J/K action for #179, but #449's
Hessian tools path still retained H1/S1 and response vectors for every nuclear
coordinate. Contracting those arrays after construction would not implement
#180's directional source boundary. A nuclear RHS also requires the moving
metric and cannot be obtained from the frozen Fock derivative alone.

## Decision

Generalize the existing generated first-integral source traversal with a
directional reduction mode. Apply each physical atom displacement to all its
mathematical center slots, including the attraction nucleus, and reduce local
shell-component derivatives before global accumulation. The directional mode
allocates only AO-by-AO H1(v)/S1(v), while the unchanged full-coordinate mode
remains available for the tiny dense oracle. No second integral recurrence or
method-specific CUDA scientific kernel is added.

Extract the existing single-perturbation CPHF mathematics into
`solve_rhf_nuclear_perturbation`, shared by dense and directional callers. The
known metric-density Fock response uses the same J/K backend as the solve and
final density response. Preserve the occupied metric block, occupied-energy
response matrix, density derivative and energy-weighted-density derivative.
For occupied coefficients C, their response C1, orbital energies eps and the
occupied-energy response E1, the conventions are

```text
P1 = 2 (C1 C.T + C C1.T)
W1 = 2 (C1 diag(eps) C.T + C diag(eps) C1.T + C E1 C.T).
```

The latter includes off-diagonal occupied-energy response, not only changes
in eigenvalues. Nuclear response performs one shared solve, never AD of SCF or
Krylov iteration history.

`directional_rhf_response` owns the exact backend and result-publication scope.
The CUDA choice reuses #511 and does not substitute DF/CPU actions. All other
work (generated first-integral execution, local reductions, AO/MO transforms
and Krylov) remains explicitly host-side. This is a B1 directional RHS/response
slice, not a molecular HVP, a resident GPU response or a larger public endpoint.

## Rejected alternatives

- Build every coordinate's first-order matrix and then contract: retains the
  coordinate-by-AO-pair storage that the direction was meant to avoid.
- Freeze density or energy-weighted density when differentiating a gradient:
  silently omits electronic relaxation and Pulay-response terms.
- Copy another CPHF reconstruction into the new consumer: permits dense and
  directional gauge/spin/factor conventions to drift.
- Treat a CUDA J/K backend as all-device response/Hessian support: AO/MO,
  Krylov, derivative and final assembly boundaries need separate qualification.
- Add a first-order CUDA component ABI solely for this small consumer: reuse
  the established providers now, record the missing device contraction stage,
  and integrate a shared compiler-owned direction/weight consumer separately.

## Invariants

Native state size/physics admission stays unchanged. A valid direction is real,
finite, shape (atom,xyz), and unnormalized. Source/reference/operator and exact
Hamiltonian identities remain bound. Invalid inputs, failed derivatives,
nonconverged solves and insufficient solver/device workspace fail closed.
Results are immutable and detached; source lifetime is checked before execution
and again before publication. No hidden full-coordinate input or oracle call
is allowed in the directional execution path.

## Validation strategy

- Compare H1(v)/S1(v) with independent native first-integral derivatives.
- Check native reconverged density and energy-weighted-density differences at
  three displacement sizes, as well as frozen Fock/overlap differences.
- Check occupied-orbital metric identities, charge-trace response, raw matrix
  symmetry, uniform translation and signed direction scaling.
- Count the shared solver invocation and forbid all-coordinate cached inputs,
  native dense derivative oracles and an in-consumer SCF rerun.
- Omit the metric-density contribution in a negative test and require the
  independently checked density/energy-weighted-density response to change.
- Preserve existing full-Hessian component, force-difference and external
  analytic-reference tests after extracting the common perturbation helper.
- On a real allocated GPU, forbid CPU J/K fallback during every metric/CPHF
  action and check independent reconverged density differences; run memcheck
  for impossible-allocation and successful replay.

Local qualification of this implementation:

- Directional tests plus shared response/RHS regression: **110 passed, 10
  skipped**; the skips are the explicitly gated existing CUDA tests.
- RTX 5090 direct/DF response plus new directional CUDA-assisted response:
  **14 passed**, including four new directional cases.
- Compute-sanitizer memcheck on impossible-budget and derivative-failure/replay
  cases: **2 passed, 0 errors, 0 bytes leaked** (a subset, not additive coverage).
- Pre-commit hooks passed for the changed files, including scientific compiler,
  SCF structure, CUDA ownership, evidence retention and formatting checks.

This change contains no native/CUDA-source edits. Generated first-component
artifacts compile from the current checkout, while whole-library tests use
identified existing binaries: CPU SHA256
`61abd1253cfbcf729c73bafea92190af2b7cb822bab80a52feaed352cd2fb335`
and CUDA SHA256
`eecfc179ca4ce1098699a395b75fc5f06cffc4c8a14f9ab0c51d973f61c5220c`.
These are not a fresh complete native build of this branch; CI and any broader
Hessian acceptance remain separate. The CUDA environment was RTX 5090
(`sm_120`), driver 580.95.05 under a finite Slurm allocation.

## Consequences and revisit conditions

The live first-order matrix storage no longer grows with the number of nuclear
directions, but traversal, Python orchestration and compilation can remain
expensive. The new diagnostics are not a total peak-memory or speedup claim.
Next compose these density responses with directional second-integral
contractions and explicit nuclear terms, with independent full `Hv` validation.
Use shared compiler/provider ownership for further GPU-resident stages rather
than interpreting this host integration as their completion.

## References

#178, #179, #180, #449, #511; `docs/hessian.md`; `docs/response.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro

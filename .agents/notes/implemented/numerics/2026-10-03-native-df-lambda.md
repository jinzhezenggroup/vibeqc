# Decision: compose native DF Lambda from retained and auxiliary actions

Status: implemented
Date: 2026-10-03

## Problem

Hundreds-AO DF CCSD and occupied-tile triples energy avoid complete ovvv/vvvv,
but native Lambda still rejected that representation. A force consumer cannot
substitute conventional response blocks, ignore virtual terms or publish a
fixed-amplitude derivative. The user explicitly prioritized completing forces.

## Decision

Reuse the existing native CUDA Lambda solve, its packed sqrt-weighted pair
coordinates, general-operator GMRES, right diagonal preconditioner and physical
acceptance gates. Add one resident DF scientific-action owner selected only
when `Problem::naux` explicitly describes a factorized Hamiltonian.

`cc/df_lambda.py` differentiates the existing unpreconditioned shared and
expanded retained-core residual programs with TensorIR AD. Independent virtual
residual inputs are held constant in these core derivatives. Each amplitude
transpose then adds the existing factorized virtual amplitude VJP for every Q
exactly once. Energy RHS needs no virtual correction because the RCCSD energy
depends on retained blocks. The final independent audit repeats the full Q sum
with the separately expanded core. Composed operator identities include both
core and auxiliary derivative hashes and the dense pair-projection convention.

All eight retained parameter VJPs are generated from the shared physical
Lagrangian. Separate Q-major Bov/Bvv cotangents account only for the virtual
residual contribution. The owner does not yet pull back retained Gram blocks;
upstream molecular composition must add those terms before source derivatives.
Explicit T1/T2 energy sources reuse the existing corrected-Lambda interface.
They are not a substitute for generating the complete DF triples response.

## Invariants and rejected alternatives

- Do not differentiate CC/DIIS iterations, Jacobi updates or the preconditioner.
  The equation remains `J^T lambda = -dE/dT` with the original dense Frobenius
  T2 metric and simultaneous pair projection.
- Never reconstruct omitted virtual integrals or a dense amplitude Jacobian.
  Every core/action intermediate has at most two virtual axes.
- Retain the fresh primal energy/residual replay, shared GMRES residual and
  expanded physical Lambda audit. CPU/oracle actions are not production retries.
- Upload the complete input state once. Use one stream and one scratch arena
  sized for the largest action. Copy the retained transpose output into its
  accumulator before any Q action reuses that arena. Each factor output is
  copied to its detached Q slice before the next action reuses scratch.
- Clear arithmetic errors once per complete action. New accumulate entry
  points preserve preceding core/Q errors; existing standalone action APIs
  still clear their own flag. Resetting at each auxiliary slice would hide
  earlier nonfinite arithmetic.
- Host result destinations and scalar seeds outlive exception-path stream
  drains. Parameter results are not published before all action checks pass.
- The DF capacity is a conservative **sum** of resident device storage and
  simultaneous host problem/CC, two packed layouts, Krylov, dense/packed audit,
  optional energy-source and complete detached parameter storage. It does not
  grant a separate full budget to each owner. The conventional route retains
  its existing capacity behavior.

## Qualification

CPU tests compare both core-plus-Q transpose forms, both RHS forms and each
retained VJP with the complete integral equations at `(o,v)=(1,2),(2,3),(3,2)`.
Native CUDA tests cold-solve the same physical Gram Hamiltonian in both DF and
complete-integral form, then compare Lambda and retained response blocks at
3e-10 absolute/relative tolerance. Explicit Gram-chain derivatives independently
check virtual factor cotangents, including Bvv symmetry projection. An
additional corrected-Lambda case uses nonzero amplitude energy sources.

A physical factor direction perturbs **all** Gram blocks. Its analytical
Lagrangian derivative includes both retained parameter and virtual-factor terms,
and is compared with a central difference of separately reconverged native
CCSD energies at step 1e-4 (3e-10 absolute / 3e-6 relative gate). A separate
two-electron case uses the lowest eigenvalue of the independent fermionic
determinant Hamiltonian for the displaced energies: RCCSD is exact in this
case, and the native energies must agree within 3e-11 Eh. This checks response
completeness at fixed Fock/orbitals, not a nuclear displacement.
Exact-budget success, one-byte-short rejection, repeated results and failed
publication are tested after successful primal convergence. The auxiliary
ledger must equal `Q * (GMRES operator actions + 3)`: primal replay, every solver
action, independent audit and factor publication.

The final complete CUDA library passed all ten focused cases (three algebra and
seven native), and all seven native cases passed CUDA memcheck with zero errors.
Sixty-seven CPU/compiler/retention regressions passed with the seven GPU cases
skipped outside their allocation; this includes 47 conventional delayed-copy
failure/lifetime cases. Seven conventional public CUDA force regressions passed:
RCCSD H2/water, RCCSD(T) water against live PySCF, batch H2, two exact/near
degenerate methane cases, and the 14-AO water-cluster warm/changed-geometry case.
Methane and cluster tests also check complete-energy directional differences.
The live PySCF case was rerun successfully after adding its missing optional
threadpoolctl dependency to the test path. All GPU work used finite Slurm jobs.

Qualified library SHA256:
`f3632c694aac0b4c8ff53a315fb32201b7e2a63dfdbe56d8078c627202c6c66e`.
The earlier focused linking against frozen #1785 was only a diagnostic; public
force qualification used this complete rebuilt library. Ignored logs, generated
source and binaries remain in `.artifacts/df-response/` in the isolated worktree.
No benchmark speedup or hundreds-AO force claim is made here.

## Remaining force work

Compose retained Gram cotangents with the full three-index source, its symmetric
projection, orbital transformation and fixed-rank metric pullback. Generate
native occupied-tile DF triples response and its corrected-Lambda sources.
Then combine conventional-reference orbital/Pulay response with contracted
three-center/metric nuclear derivatives on the same Hamiltonian. Auxiliary-g
derivatives, complete nuclear finite differences and independent analytic force
gates must be qualified before extending public descriptors.

The current per-Q VJP is a bounded correctness baseline. Its work ledger includes
every Q/action; bounded storage is not a performance claim. Revisit an AD of the
auxiliary-hoisted residual schedule when complete force profiles establish that
this VJP dominates, retaining the expanded action for independent publication
checks. Mixed precision and extra-stream overlap remain separate #1764/#1765
qualifications; compiler-owned equations remain required by #1763.

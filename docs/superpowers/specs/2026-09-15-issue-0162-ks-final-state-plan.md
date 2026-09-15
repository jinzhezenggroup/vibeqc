# Issue 162 KS final-state handoff implementation plan

## Outcome

Provide an internal, versioned KS final-state snapshot that proves a successful
resident CUDA solve can supply a current and mutually consistent
`D/F/C/epsilon/occupations/W/grid/provider` state to Issue #163 without
changing the public energy-only API.

## Commit 1: Pure KS state contract and negative tests

Files:

- add `src/dft/ks_final_state.hpp`
- add `src/dft/ks_final_state.cpp`
- add `tests/native/test_ks_final_state.cpp`
- update `CMakeLists.txt`

Work:

1. Define `KsModelIdentity`, `KsFinalStateIdentity`, detached candidate and
   verified snapshot types.
2. Reuse `scf::solver::EigenFrameDiagnostic` helpers for finite,
   S-orthogonality, eigen-residual, density reconstruction, commutator,
   electron trace and metric-idempotency checks.
3. Validate the KS energy components supplied by the physical resident state
   rather than applying the quadratic HF energy formula.
4. Construct `W` only after all state checks pass and only when requested.
5. Add deterministic analytic RKS/UKS tests and faults for every identity,
   generation, spin, occupation and numerical invariant.

Verification:

- build and run the new native CPU contract test
- run existing generic final-state and DFT CPU tests
- run formatting and whitespace checks

## Commit 2: Resident CUDA producer and lifecycle

Files:

- add `src/dft/cuda_ks_final_state.hpp`
- update `src/dft/cuda_ks.hpp`
- update `src/dft/cuda_ks.cpp`
- update `src/dft/cuda_ks_kernels.hpp`
- update `src/dft/cuda_ks_kernels.cu` only if a bounded retained-frame copy is
  required
- extend `tests/native/test_ks_cuda.cpp`

Work:

1. Allocate and charge retained physical Fock, accepted coefficient frames,
   orbital energies, generation records and eligibility metadata.
2. Assign immutable plan/source identity at construction. Advance a nonzero
   solve epoch and invalidate eligibility at the start of every solve attempt.
3. On convergence, bind retained D, physical F, C and epsilon to explicit
   density/Fock/orbital generations. Do not authorize pending, failed or
   nonconverged state.
4. Expose a token query and detached snapshot read that require exact current
   identity before transfer and recheck eligibility before publication.
5. Keep energy-only execution free of W construction and final matrix export.
6. Add stale-token, warm-replay, changed-owner, failed-solve, empty-spin and
   transfer-count regressions.

Verification:

- compile CPU and CUDA targets
- run no-device structural/preflight tests
- run allocated-GPU native KS tests and compute-sanitizer for snapshot reads

## Commit 3: Method handoff, diagnostics and closure evidence

Files:

- update `src/methods/dft_method.cpp` only for an internal snapshot access
  boundary if required by prepared ownership
- update `src/methods/method.hpp` or internal diagnostics without changing
  public result layouts
- update `python/vibeqc/resources_ks.py` and resource tests if retained storage
  changes the composed plan
- update `docs/ks_diagnostics.md`
- add a focused source-bound evidence summary under
  `benchmarks/results/ks-final-state-162/`
- update the CUDA ownership ledger only if the repository check requires it

Work:

1. Make the successful prepared owner expose the current internal snapshot to
   a future #163 consumer while preserving serialization and lifetime.
2. Report actual snapshot transfers/synchronizations separately from ordinary
   energy execution.
3. Confirm fixed-geometry replay advances epoch, changed geometry rebuilds the
   owner, and failed neighbors cannot leak an eligible token.
4. Record current source SHA, build/runtime versions, test commands, device,
   numerical checks, resource changes and unsupported scope.
5. Map the final result to each remaining #162 acceptance requirement.

Verification:

- focused native CPU/CUDA and Python diagnostics/resource suites
- existing RKS/UKS cold, warm, changed-geometry and ragged-batch regressions
- pre-commit and repository ownership/structure checks
- final `git diff --check` and staged-file size/evidence audit

## Delivery

1. Rebase or merge current `origin/master` only when needed, preserving this
   branch's reviewed commits.
2. Push `codex/issue-0162-e` to `xshengrui/vibeqc`.
3. Open a PR to `jinzhezenggroup/vibeqc:master` with exact validation and scope
   limits, AI attribution, and `Refs #162, #163`.
4. Resolve CI and review findings with focused follow-up commits.
5. Merge only after required checks pass and review threads are resolved.
6. Confirm the merged code and GitHub issue state; close #162 only when its
   remaining acceptance item is genuinely satisfied.

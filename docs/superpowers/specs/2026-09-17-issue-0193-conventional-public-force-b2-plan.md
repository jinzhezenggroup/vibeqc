# Issue 193 conventional public force B2 implementation plan

Status: approved on 2026-09-17; implementation in progress.

Design source:
`docs/superpowers/specs/2026-09-17-issue-0193-conventional-public-force-b2-design.md`
in this branch's approved design/plan commit.

## Outcome

Ship one reviewable PR that makes conventional canonical RHF-MP2 analytic
forces a genuine bounded native CPU/CUDA public capability, including
transactional C/Python publication, homogeneous prepared-batch isolation,
resource/response diagnostics, current scientific evidence, CI fixes, review
resolution, and merge. RI-MP2 forces remain explicitly unsupported C2 work.

## Fixed decisions and dependencies

- Work only in `D:/Users/Lenovo/Desktop/My_ViveQC/wt-issue-0193-e` on
  `codex/issue-0193-e`; preserve the dirty root checkout.
- Reuse the exact conventional energy/reference/provider semantics already in
  `src/posthf/mp2_energy.*`, `src/posthf/native_provider.*`, and
  `src/scf/types.hpp`.
- Port the validated equations from `tools/vibeqc_mp2/gradient.py` and the
  stopping/resource semantics from `tools/vibeqc_response/krylov.py`; production
  native code must not invoke either Python module.
- CPU and CUDA implement the same force domain. CUDA orchestration reuses
  `src/scf/cuda_one_electron_gradient.hpp` and
  `src/scf/cuda_weighted_eri.hpp`; CPU uses native shell-stream derivative
  evaluation rather than a coordinate derivative tensor.
- Extend ABI structs append-only and honor caller `struct_size`; do not reorder
  or reinterpret existing fields.
- The existing C API remains the final output transaction owner. Method code
  returns unpublished candidates only after the complete force chain succeeds.
- The qz CPU Notebook `general` and later an approved GPU resource provide the
  authoritative Linux builds and numerical runs. Read remote project rules and
  `docs/STATUS.md` before creating the remote `issue-0193-e` worktree.
- Use these exact task locations after the remote rules confirm the selected
  project and shared mount:
  - source: `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-worktrees/issue-0193-e`;
  - CPU build: `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-worktrees/issue-0193-e/build/issue-0193-e-cpu`;
  - CUDA build: `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-worktrees/issue-0193-e/build/issue-0193-e-cuda`;
  - evidence root: `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0193-e`.
- Large raw validation output stays in the approved persistent experiment
  directory. Git receives the smallest reproducible summary, manifest,
  checksums, and scripts actually needed by tests or review.

## Phase 0: establish authoritative baselines

Files changed: none.

1. Read remote
   `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/AGENTS.md`, the
   integration checkout `AGENTS.md`, and `docs/STATUS.md` using
   `inspire --no-env-file --account qz notebook exec general --workspace CPU资源空间`.
2. Create or reuse exactly
   `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-worktrees/issue-0193-e`
   only after verifying integration checkout commit, dirty state, existing
   worktrees, and absence of an active conflicting task directory.
3. Build current `origin/master` in a fresh `build/issue-0193-e-baseline`
   directory with the documented CPU environment. Run the current MP2 contract
   and public tests to record that energy passes and force requests are rejected.
4. Record compiler, Python, BLAS, CUDA availability, exact commit, test commands,
   and results in the task experiment directory; do not commit baseline build
   artifacts.

Gate: baseline must reproduce current energy-only behavior before implementation
results are accepted.

## Commit 2: native adjoint, response, weights, and resource plan

Primary files:

- add `src/response/native_gmres.hpp`
- add `src/response/native_gmres.cpp`
- add `src/posthf/mp2_gradient.hpp`
- add `src/posthf/mp2_gradient.cpp`
- update `CMakeLists.txt`
- update `cmake/VibeQCTests.cmake`
- add `tests/native/test_native_gmres.cpp`
- add `tests/native/test_mp2_gradient.cpp`

Implementation steps:

1. Add failing native tests for checked GMRES workspace sizing, exact small
   linear solves, restarted convergence, recomputed true residual, breakdown,
   nonfinite input/operator output, iteration exhaustion, zero RHS, and budget
   rejection.
2. Implement a real-valued, operator-callback restarted GMRES owner. Keep
   Arnoldi/Hessenberg/Givens state bounded by an immutable plan, return an enum
   status plus iteration/restart counts and true residuals, and allocate no
   solver-history tape.
3. Add failing small-oracle tests for canonical MP2 energy adjoints, denominator
   derivatives, streamed orbital RHS, Z-vector identity, relaxed one-electron/
   overlap/two-electron weights, and gradient component sign. Derive fixtures
   independently from the production implementation; include deliberate term
   omission and sign reversal controls.
4. Implement native MP2 adjoint and relaxed-weight owners against
   `scf::PhysicalReference` and `posthf::NativeBlockProvider`. Bind every payload
   to reference/provider/equation identities and reject stale or inconsistent
   shapes before allocation.
5. Add checked simultaneous-memory planning for MP2/adjoint arrays, response
   vectors, relaxed weights, one shell-local AO cotangent, derivative staging,
   and candidate output. Test every exact boundary and overflow path.
6. Refactor shared formulas from the existing energy owner only when this
   eliminates duplicate denominator/block semantics; keep energy-only behavior
   and diagnostics unchanged.

Focused verification:

```text
cmake --build build/issue-0193-e-cpu --target vibeqc_native_gmres_tests vibeqc_mp2_gradient_tests
ctest --test-dir build/issue-0193-e-cpu -R "vibeqc_(native_gmres|mp2_gradient)_tests" --output-on-failure
python -m pytest tests/python/test_mp2_gradient_fast.py -q
```

Commit title:
`feat(mp2): add bounded native gradient response core`

Gate: independent component oracles pass, every failure status is covered, and
the prepared byte count bounds the measured native allocations.

## Commit 3: conventional force owner and single-system public API

Primary files:

- add `src/posthf/mp2_force.hpp`
- add `src/posthf/mp2_force.cpp`
- add `src/posthf/mp2_derivative_cpu.cpp`
- add or refactor a production CUDA orchestration unit under `src/posthf/`
- update `src/posthf/bridge.cpp` only to call shared production-safe helpers
- update `src/methods/mp2_method.cpp`
- update `src/methods/mp2_method.hpp`
- update `src/methods/registry.cpp`
- update `include/vibeqc/vibeqc.h`
- update `include/vibeqc/vibeqc.hpp`
- update `python/vibeqc/_native.py`
- update `python/vibeqc/calculator.py`
- update `tests/native/test_mp2_contract.cpp`
- update `tests/native/test_mp2_cuda_status.cu`
- update `tests/python/test_mp2_public.py`
- add focused public force tests if the existing file would become unwieldy

Implementation steps:

1. Add failing CPU tests for shell-local one-electron/overlap/nuclear and
   weighted-ERI contraction against independent dense derivative fixtures.
   Prove no full AO rank-four cotangent or coordinate derivative tensor is
   allocated.
2. Implement the CPU derivative stream and the complete
   `ConventionalForcePlan/Result` owner. Assemble unpublished HF + MP2 component
   gradients, validate finiteness and identity, then convert once to public
   forces.
3. Extract the production-safe CUDA one-electron and shell-weighted-ERI
   orchestration currently reached through development bridges. Reuse existing
   kernels and checked budgets; preserve bridge behavior through the shared
   helper.
4. Change `Mp2Prepared` so energy-only requests retain the existing path and
   conventional force requests execute the complete owner. Clear diagnostics
   before all validation/execution. Reject RI force, unsupported backend/model,
   and insufficient force plan budgets explicitly.
5. Extend `vibeqc_correlation_diagnostic` append-only with response iterations,
   true absolute/relative residuals, response/derivative workspace, planned and
   measured endpoint peak, and provenance hashes/flags required by Issue #193.
   Update the C++ and Python mirrors with prefix-copy compatibility tests.
6. Promote MP2 force capability in the registry and remove the Python
   hard-coded force rejection. Keep RI force rejection at prepared execution
   with a specific C/Python error.
7. Add C ABI transaction tests that seed energy, force buffers, and diagnostics
   with sentinels, trigger reference/denominator/response/derivative/CUDA/
   nonfinite failures, and prove no candidate output or prior diagnostic leaks.
8. Add repeated success-failure-success and changed-geometry public tests.

Focused verification:

```text
cmake --build build/issue-0193-e-cpu --target vibeqc_mp2_contract_tests
ctest --test-dir build/issue-0193-e-cpu -R vibeqc_mp2_contract_tests --output-on-failure
python -m pytest tests/python/test_mp2_public.py tests/python/test_mp2_gradient_fast.py -q
```

On CUDA, additionally build and run `vibeqc_mp2_cuda_status_tests` plus the
focused public Python force cases.

Commit title:
`feat(mp2): publish bounded conventional analytic forces`

Gate: a public conventional force succeeds on CPU and CUDA; every injected
failure leaves caller buffers and diagnostics transactional; energy-only and RI
energy regressions remain green.

## Commit 4: homogeneous prepared batch and per-item failure isolation

Primary files:

- update `src/methods/mp2_method.cpp`
- update `src/methods/mp2_method.hpp`
- update `src/methods/registry.cpp`
- update `src/api/c_api_batch.cpp` only if method-neutral transaction handling
  needs a proven correction
- update `python/vibeqc/batch.py` only for MP2-specific diagnostics/errors
- update `tests/native/test_batch.cpp`
- update `tests/python/test_mp2_public.py` or add
  `tests/python/test_mp2_batch.py`

Implementation steps:

1. Add failing tests for two or more conventional MP2 items covering energy and
   force schedules, ragged atom counts, changed coordinates, item-order
   independence, and repeated execution.
2. Implement `Mp2PreparedBatch` as independent item owners sharing immutable
   options only. Preserve per-item status and candidate publication; never
   share mutable response/provider/diagnostic state.
3. Reject all warm-start and HF profiling flags for MP2, RI force batches,
   mixed output/model domains, and invalid coordinates. Do not silently ignore
   flags.
4. Add one-fails-neighbours-succeed tests for reference nonconvergence,
   denominator, response, budget, nonfinite, and output-buffer failures. Verify
   failed item buffers remain sentinel-filled and a later batch call is clean.
5. Add per-item correlation/response diagnostic access if required; otherwise
   document and test the existing supported diagnostic boundary rather than
   returning another item's state.

Focused verification:

```text
cmake --build build/issue-0193-e-cpu --target vibeqc_batch_tests
ctest --test-dir build/issue-0193-e-cpu -R "vibeqc_(batch|mp2_contract)_tests" --output-on-failure
python -m pytest tests/python/test_mp2_batch.py tests/python/test_mp2_public.py -q
```

Commit title:
`feat(mp2): add isolated conventional force batches`

Gate: MP2 advertises batch support, per-item status/output isolation is proven,
and unsupported flags/models fail before mutable execution.

## Commit 5: scientific qualification, documentation, and capability evidence

Primary files:

- add `tools/validate_mp2_public_force.py`
- update `docs/mp2.md`
- update `docs/methods.md`
- update any public API reference describing properties/diagnostics
- add a minimal evidence directory such as
  `benchmarks/results/issue193-conventional-force-b2/`
- update focused Python tests for evidence schema and documentation claims

Implementation steps:

1. Add a parameterized validation driver that writes a fresh output directory,
   records exact commit/environment/hardware/model identities, never overwrites
   prior evidence, and supports CPU or CUDA public single/batch endpoints.
2. Run H2, LiH, H2O, an f-shell case, and at least one system beyond twelve AOs.
   Compare public forces against matching PySCF conventional MP2 analytic
   gradients and fully re-solved central finite differences at at least three
   step sizes. Record absolute/relative/vector component errors.
3. Check translation sum rule, rotational covariance/torque, changed geometry,
   CPU/CUDA agreement, response residuals, and force/energy identity.
4. Exercise denominator, response-iteration, minimum-budget, nonfinite, and
   unsupported-model failures on both backends. Run CUDA memcheck on public
   single and batch force endpoints.
5. Compare planned capacity with measured endpoint peak and record transfers,
   provider/equation identity, tile counts, response iterations, device, CUDA
   runtime/driver, and exact commit.
6. Keep raw outputs under
   `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0193-e/<run-id>`;
   commit only the
   reviewed summary/manifest/checksums and scripts necessary to reproduce it.
7. Update docs to state the exact conventional support boundary, force sign and
   units, RI C2 limitation, batch semantics, failure transaction, diagnostics,
   and evidence location. Remove obsolete "MP2 analytic forces unavailable"
   claims only after both public backends qualify.

Focused verification:

```text
python -m pytest tests/python/test_mp2_public.py tests/python/test_mp2_batch.py \
  tests/python/test_mp2_gradient_fast.py -q
python tools/validate_mp2_public_force.py --backend cpu \
  --cases h2,lih,h2o,water-def2-svp,f-shell --fd-steps 0.004,0.002,0.001 \
  --output /inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0193-e/<run-id>/cpu
python tools/validate_mp2_public_force.py --backend cuda \
  --cases h2,lih,h2o,water-def2-svp,f-shell --fd-steps 0.004,0.002,0.001 \
  --output /inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0193-e/<run-id>/cuda
ctest --test-dir build/issue-0193-e-cuda -R "vibeqc_(mp2|batch)" --output-on-failure
```

Commit title:
`test(mp2): qualify public conventional force endpoints`

Gate: current CPU/CUDA evidence satisfies every design case, committed evidence
is reviewable and nonredundant, and docs match actual capability/failure state.

## PR, CI, review, and merge loop

1. Before each commit, inspect `git status`, working/staged diffs, file counts,
   line/byte totals, largest candidate files, and evidence duplication. Stage
   only the current slice and include the required verified-or-unverified Codex
   signature trailers.
2. Push `codex/issue-0193-e` after the first implementation slice is focused and
   passing, then create one PR against `master`. The PR description leads with
   the public conventional force outcome, exact support boundary, transactional
   and batch semantics, validation commands/results, evidence locations, and
   RI C2 limitation. Link Issue #193 without claiming C2 completion.
3. After every push, inspect PR diff, required checks, review threads, and latest
   head SHA. Diagnose failures from logs; add focused regressions before fixes.
4. Apply review changes in scoped commits, update the PR description when scope
   or evidence changes, reply to/resolve threads only after the fix and test are
   present, and preserve AI signatures on GitHub comments/reviews.
5. Rebase or merge current `origin/master` only when needed and without
   rewriting shared history. Re-run affected focused tests after conflict
   resolution.
6. Before merge, perform a requirement-by-requirement audit against the design,
   plan, Issue #193 B2 checklist, final diff, current CI, CPU/CUDA artifacts,
   resource evidence, and public behavior. Merge only when all required checks
   and review gates are satisfied and no B2 item remains unverified.
7. After merge, verify the PR state and merge commit on `origin/master`, confirm
   Issue #193 reflects B2 accurately without incorrectly closing C2, and report
   retained qz resources and evidence paths. Worktree/resource retirement
   follows existing authorization and retention rules rather than happening
   automatically.

## Plan approval gate

Implementation begins only after the user explicitly approves this plan and
scope. Approval authorizes execution of the already requested development,
commit, push, PR maintenance, review-fix, and merge workflow; it does not expand
authorization to delete unrelated worktrees/data, publish private artifacts, or
perform RI C2 work.

After approval, add this plan to the still-unpushed design commit by amending
commit 1 without changing the approved design text. Then start Phase 0 and
Commit 2. Saving this file before approval does not itself approve execution.

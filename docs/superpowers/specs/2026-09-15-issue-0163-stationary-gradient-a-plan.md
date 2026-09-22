# Issue 163-A stationary DFT derivative implementation plan

Design: `docs/superpowers/specs/2026-09-15-issue-0163-stationary-gradient-a-design.md`

Implementation status: this plan does not establish completion of #163-A.
The native handoff and fixed-density diagnostics are implemented; generated
stationary XC binding still requires a matching native energy/derivative domain.
See [current method behavior](../../user/methods.md).

## Outcome

Add an internal method-level contract that accepts only a consistent successful
KS final state, consumes Issue #236 generated XC geometry partials on a stable
grid branch, and validates AO-centre, grid-point and weight directions against
an independent finite-difference CPU oracle. Keep public DFT forces disabled and
leave complete molecular assembly, physical partition motion and CUDA lowering
to Issue #163 B/C.

## Implementation location

- `python/vibeqc/_dft_gradient.py`: private method/state contract, generated-XC
  partial ownership metadata, stable-grid motion and component contraction.
- `tools/vibeqc_validation/dft_gradient.py`: development/test-only CPU energy
  re-evaluation and multistep directional finite-difference oracle.
- `tests/python/test_dft_stationary_gradient.py`: contract, component,
  directional, identity and negative tests.
- `docs/methods.md`: current-state note that the internal A-slice derivative
  contract exists while public LDA/PBE forces remain unsupported.
- `.agents/notes/implemented/numerics/2026-09-15-stationary-dft-gradient-boundary.md`:
  durable ownership and topology rationale required by repository instructions.

Do not put method/SCF state into `python/vibeqc_compiler/`; that package remains
the owner of local mathematical expressions, lowering and generated identities.
Do not add code to the forwarding-only `tools/vibeqc_dft` compatibility package.

## Step 1: establish the isolated baseline

1. Confirm `codex/issue-0163-a` is clean and based on current `origin/master`.
2. Reuse the repository's available Python environment without reading `.env`.
3. Run the existing tests that own the foundations:

   ```powershell
   python -m pytest `
     tests/python/test_xc_contractions.py `
     tests/python/test_xc_contractions_native.py `
     tests/python/test_ks_diagnostics.py -q
   ```

4. If the baseline fails, diagnose the failure before implementation and keep
   any environment limitation separate from the new slice.

No commit is created for an unchanged baseline.

## Step 2: implement the stationary state contract

Add tests first for an internal immutable state containing:

- exact model/basis/geometry/grid/functional/regularization/provider identity;
- nonzero owner, solve epoch and density/Fock/orbital generations;
- converged/successful/physical flags;
- spin-resolved `D`, physical `F`, `C`, orbital energies, occupations and `W`;
- overlap, physical residual and explicit gradient sign/topology policy.

The implementation validates finite shapes, spin counts, symmetry, electron
density reconstruction, `W = C diag(f epsilon) C^T`, S-orthonormal orbitals,
physical Fock eigen residuals and exact identity equality. It fails before any
derivative evaluation on stale or incomplete state. Keep tolerances named and
scale-aware; do not silently repair arrays or accept dimension-only matches.

Files:

- add `python/vibeqc/_dft_gradient.py`;
- begin `tests/python/test_dft_stationary_gradient.py`.

Verification:

```powershell
python -m pytest tests/python/test_dft_stationary_gradient.py -q
pre-commit run --files `
  python/vibeqc/_dft_gradient.py `
  tests/python/test_dft_stationary_gradient.py
```

Commit:

```text
feat(dft): define stationary KS derivative contract
```

## Step 3: consume generated XC geometry partials

Add an internal generated-XC result wrapper that binds `GeometryPartials` to:

- the `DiscreteEnergyContract` identity;
- basis, geometry, grid and functional identity;
- density generation and stable topology identity;
- exact centre/point/weight shapes.

Add stable-grid motion values with independent AO-centre, grid-point and
quadrature-weight directions. Contract each component exactly once and return a
component record plus total directional gradient. Preserve gradient sign;
force conversion remains unavailable.

Use `vibeqc_compiler.xc.ContractionProgram(..., "geometry")` as the only
scientific derivative owner. If an interface addition is needed, expose an
existing contract/partial type without duplicating formulas or putting method
policy into the compiler.

Extend tests for LDA/PBE and RKS/UKS, asymmetric spin densities, component
isolation, combined directions and rigid translation consistency.

Verification:

```powershell
python -m pytest `
  tests/python/test_dft_stationary_gradient.py `
  tests/python/test_xc_contractions.py `
  tests/python/test_xc_contractions_native.py -q
python tools/check_compiler_structure.py
```

Commit:

```text
feat(dft): bind generated XC geometry to stationary state
```

## Step 4: add the independent multistep CPU oracle

Implement a test/development oracle that:

1. copies the fixed density and exact functional/grid contract;
2. independently displaces AO centres, grid points and weights;
3. rebuilds AO collocation and re-evaluates scalar discrete XC energy;
4. computes central differences over several decreasing step sizes;
5. identifies a stable region and compares it with the generated analytic
   directional component.

The oracle must not call the generated geometry pullback. It may reuse the
audited scalar functional energy because the derivative is obtained by an
independent displaced-input evaluation. Record per-step estimates so one
favourable finite-difference step cannot establish acceptance.

Add negative controls demonstrating that omitted centre/point/weight terms and
reversed signs exceed the acceptance gate. Add deterministic rejection for
invalid AO ownership, nonfinite directions, identity mismatches and requested
topology changes.

Files:

- add `tools/vibeqc_validation/dft_gradient.py`;
- complete `tests/python/test_dft_stationary_gradient.py`.

Verification:

```powershell
python -m pytest tests/python/test_dft_stationary_gradient.py -q
python -m pytest `
  tests/python/test_xc_contractions.py `
  tests/python/test_xc_contractions_native.py `
  tests/python/test_xc_integration.py `
  tests/python/test_dft_scf.py -q
```

Commit:

```text
test(dft): validate stationary XC geometry directions
```

## Step 5: document the implemented boundary

Update `docs/methods.md` without describing historical execution logs. Add an
implemented Agent Note preserving why stationary method state remains outside
the compiler and why topology motion is explicit rather than inferred.

Verify that documentation does not claim complete molecular gradients, moving
partition response, CUDA execution or public forces.

Verification:

```powershell
rg -n "TBD|TODO|complete.*force|public.*force" `
  docs/methods.md `
  .agents/notes/implemented/numerics/2026-09-15-stationary-dft-gradient-boundary.md
pre-commit run --files `
  docs/methods.md `
  .agents/notes/implemented/numerics/2026-09-15-stationary-dft-gradient-boundary.md
```

Commit:

```text
docs(dft): record stationary gradient ownership boundary
```

## Step 6: final local verification and review

1. Inspect branch and staged/unstaged diffs; include only Issue #163-A files.
2. Run all focused tests above plus the repository's practical Python suite or
   the exact CI commands when available.
3. Run `git diff --check` and pre-commit across all changed files.
4. Review the implementation against every design acceptance item and confirm
   public DFT forces still reject requests.
5. Use the requesting-code-review and verification-before-completion workflows
   before publication.

## Step 7: publish and merge

1. Push `codex/issue-0163-a` to the `fork` remote.
2. Create one PR against `jinzhezenggroup/vibeqc:master`, referencing #163 and
   stating that it completes only slice A.
3. Include actual tests, limitations and the required AI signature.
4. Monitor CI and review feedback, diagnose failures, amend with additional
   meaningful commits, and push normally without rewriting shared history.
5. Rebase or merge current master only if required; do not touch either #149
   worktree.
6. Merge only after required checks and review gates pass. Verify the merge
   commit is present on `origin/master`; do not delete worktrees or branches
   without separate cleanup authorization.

## Completion evidence

Authoritative acceptance runs use the dedicated Issue #163-A worktree on qz:
CPU executes the independent oracle and focused regressions, while a separate
#163-specific high-priority GPU workload checks the CUDA build/environment and
unchanged generated-XC path without claiming CUDA gradient execution. Local
Windows runs are fast development checks rather than the final platform gate.

The goal is complete only when all of the following are current and verified:

- the contract and oracle files are merged into `origin/master`;
- LDA/PBE RKS/UKS centre/point/weight and combined directional tests pass;
- stale/incomplete state, topology and component-negative tests pass;
- existing XC and DFT focused regressions pass;
- the PR's required CI checks are green and the PR is merged;
- public DFT forces remain unsupported and #163 B/C limitations are documented.

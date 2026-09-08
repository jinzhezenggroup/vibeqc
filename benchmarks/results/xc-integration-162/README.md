# DFT03 fixed-density CPU XC acceptance, 2026-09-08

This archive accepts only fixed-density LDA/PBE XC energy integration and AO
potential assembly. It does not complete issue #162, register a public DFT
method, establish SCF/quadrature convergence, or validate GPU execution.
The interface and remaining dependencies are in
[the integration contract](../../../docs/xc_integration.md).

## Source and environment

- Upstream baseline: `1e93c3cfa10afdf99ee26c70312bcfe533114331` (PR #211),
  after PR #210. Latest master and #162's complete comments/timeline were
  inspected at task start; no direct #162 implementation PR was linked.
- Validated and independently reviewed code/test/reference/document Git tree:
  `7a265dd0def716edbb659a23e7c7ffb1d9df9b74`. Local and remote `git write-tree`
  matched exactly. This was an uncommitted candidate during validation/review;
  `numerical.json`'s revision is the baseline, with candidate source hashes
  recorded separately. Evidence files and this summary are added afterward.
- Local worktree: `D:/Users/Lenovo/Desktop/My_ViveQC/wt-issue-162`.
- Remote worktree:
  `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-issue-162`.
- Branch: `codex/issue-162-xc-integration`. The original local and remote
  integration checkouts' uncommitted environment files were preserved.
- qz account alias `qz`, Project `原子级化学反应基座模型2.0`, Workspace
  `CPU资源空间`, existing `general` Notebook: 4 CPU, 16 GiB, HIGH priority 9.
  No new Notebook/GPU was allocated, restarted or stopped. The existing CPU
  instance remains available; no new retention window or shutdown was scheduled.
- Independent `.venv` and `build/cpu` were restored using the saved
  `env-assets/vibeqc-cpu-py311-v1/recipe/setup_cpu.sh`, its offline wheelhouse,
  `UV_OFFLINE=true`, `UV_LINK_MODE=copy`, and the saved Python 3.11.16 runtime.
  Full paths follow the supplied qz environment guide. No `.env` was read.

Exact compiler/dependency/library/source versions and SHA-256 values are in
[versions.log](versions.log). They include PySCF 2.14.0, Libxc 7.0.0,
NumPy 2.2.6, SciPy 1.15.3, SymPy 1.14.0, GCC 11.4.0, CMake 3.31.10 and
Ruff 0.16.2. BLAS/OpenMP environment thread limits were one.

## Commands and results

[verification.sh](verification.sh) is the exact executed Bash command sequence.
It uses an explicit worktree and interpreter, `set -euo pipefail`, a final exit
sentinel, full logs and unchanged tolerances. It was uploaded and invoked via:

```powershell
inspire --no-env-file --account qz notebook exec general --workspace CPU资源空间 --cwd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-issue-162 'bash ../vibeqc-issue-162-verify.sh # end'
```

| Check | Measured result | Evidence |
| --- | --- | --- |
| Baseline grid/AO/XC, before implementation | 90 passed | Remote `../vibeqc-issue-162-baseline.log` |
| New + grid/AO/XC focused regression | 129 passed (39 new) | [focused-tests.log](focused-tests.log) |
| Native CPU suite, including HF/integrals/grid | 10/10 passed | [ctest.log](ctest.log) |
| Complete Python regression | 891 passed, 173 skipped | [python-tests.log](python-tests.log) |
| CPU build | Passed, current build up to date | [build.log](build.log) |
| Ruff check/format and Git whitespace | Passed | [format.log](format.log) |
| Reference regeneration | Two independent invocations exactly match each other and saved metadata/array hashes | [reference-generation.log](reference-generation.log) |
| Numerical evidence runner | 24 independent comparisons, all passed | [numerical.json](numerical.json), [numerical.log](numerical.log) |

The complete verifier returned exit 0. Optional CUDA/GPU/PyTorch skips are not
numerical acceptance for those backends. Existing HF tolerances and tests were
not changed. An initial format-only attempt stopped on CRLF bytes transferred
from Windows; those files were normalized to LF and the verifier rerun. The
original failure log remains at remote `build/issue-162-initial-format-failure.log`.
No numerical failure was hidden or tolerance relaxed.

## Numerical scope

H2, asymmetric water, actual Cartesian f and spherical f fixtures use identical
explicit grids, supplied matrices and exact shell/AO conventions on both sides.
PySCF owns the reference AO/features/Libxc/integration/matrix path; VibeQC owns
the evaluated native AO/features/expression/coefficient/assembly path. Source
expression parameters are intentionally shared by functional definition; this
is an independent implementation comparison, not a different-functional test.

- Maximum energy difference: `4.440892098500626e-16` Hartree.
- Maximum potential element difference: `8.881784197001252e-16` Hartree.
- All 24 comparisons retain `atol=1e-11, rtol=1e-10` per element.
- For water's 16 diagonal/symmetric-off-diagonal spin/layout directions, the
  largest errors at density steps `1e-3`, `3e-4`, `1e-4` are respectively
  `1.119500121050e-6`, `1.007542924647e-7`, `1.119334136490e-8` Hartree per unit
  density perturbation. All 48 samples are saved; error decrease is also
  asserted in tests. Additional mixed-spin and synthetic tau checks run in
  pytest. These density steps are not nuclear displacements.
- Deliberately doubled density, weights and potentials are rejected by the
  independent regression gates. Tests also cover total/separate/equal spins,
  alpha/beta swap, partial tiles, symmetry, changed grids and basis coefficients,
  stale molecular grids, empty grids and explicit unsupported-domain failures.

Every nonempty grid point must satisfy DFT02 `interior-v1`; no tail clipping or
weight-based skipping is implemented. The fixtures are controlled quadratures,
not converged molecular DFT calculations. Future CPU RKS requires a versioned
tail/limit policy, a native DFT adapter, common #202 Coulomb strategy, occupations,
DIIS and physical-residual convergence, followed by #203 resource composition.
No second J/K scheduler or global budget API is introduced here.

## Independent review

Two read-only agents, neither involved in implementation, inspect the same
fixed tree above: one for mathematics/independent references/counterexamples,
one for architecture/interfaces/resources/regressions. Their final findings and
the main agent's disposition are recorded in `review.md` before the local commit.

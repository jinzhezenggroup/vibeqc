# RCCSD A validation record — 2026-09-08

Scope: issue #148 A only, real conventional all-electron RCCSD energy and
physical T1 equations at supplied amplitudes. No T2 residual/CCSD convergence,
GPU, triples, Lambda, gradients or performance claim. Refs #148.

## Code and execution identity

- Baseline/default branch: `origin/master`,
  `1e93c3cfa10afdf99ee26c70312bcfe533114331` (checked after fetch).
- Local worktree: `C:/Users/Lenovo/.codex/worktrees/b0dc/vibeqc`;
  branch `codex/148-rccsd-a`.
- Remote worktree:
  `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-148-a`.
  Independent `.venv` and `build/cpu`; remote master business code unchanged.
- `source-snapshot.json` fixes the 16 implementation/test/document/reference
  files tested on that baseline. `source-verification.log` confirms all their
  raw SHA-256 values after upload. Source archive SHA-256:
  `6972ea075d54fba505ceb93060559c4221f2cc1ef5c418d55296910d2b05b79a`.
  The snapshot also records staged Git-blob SHA-256 values. With this host's
  `core.autocrlf=true`, only the generated equation JSON and upstream manifest
  JSON change from CRLF to LF in Git; their normalized contents were checked
  byte-for-byte against the reviewed/tested files. Executable sources match
  without normalization changes.
- Remote: qz account, project 原子级化学反应基座模型2.0,
  CPU资源空间 / general, live configuration 4 CPU / 16 GiB, priority 9 HIGH.
  One build job and one BLAS/OpenMP thread; no GPU allocated or stopped.
- Runtime: Linux 5.15.0-119-generic x86_64, Python 3.11.16, NumPy 2.2.6,
  PySCF 2.14.0, CMake 3.31.10, GCC 11.4.0, Ruff 0.16.2.
  CPU library SHA-256:
  `d576f7b604ea422dcbe1f5e9aaae6bd98f7280e1adaf983392f8c3c783cdff56`.
- Environment recreated using the shared `vibeqc-cpu-py311-v1` offline recipe,
  with `VIBEQC_BUILD_JOBS=1`. No environment files were added to the slice.

## Executed gates

The historical validation used the author's isolated checkout through
`inspire --no-env-file --account qz notebook exec general --workspace CPU资源空间`
with an explicit remote working directory and one BLAS/OpenMP thread.
The machine-specific scratch runner and routine tool logs have been removed;
their measured outcomes are preserved below. Source hashes and scientific
records remain unchanged. Portable commands for the final A/B/C implementation
are in [the C reproduction section](../rccsd-148-c/README.md#reproduction).

| Check | Observed result |
| --- | --- |
| New CC equations, references, provenance | Included all 30 tests; no native-provider skips on qz |
| CC + TensorIR + post-HF + validation Python regression | **204 passed**, 57.70 s, no skips/failures |
| Native CPU CTest, including RHF/UHF/integrals | **10/10 passed**, 15.75 s |
| Ruff check and format | Passed; 10 Python files |
| pip check | No broken requirements |
| Shared CG01 numerical evidence | Five records passed |
| PySCF reference regeneration | Two generations have identical inputs/output case hashes |
| Local Windows Python checks | **27 passed, 3 native-library skips**, 1.73 s |

`numerical.json` reports every energy/residual group available from each oracle
for original, JSON-replayed and conservatively optimized programs. Maximum
absolute errors across these comparisons are:

| Oracle/input | Maximum error |
| --- | --- |
| Explicit determinant algebra, all groups | 2.220446049250313e-16 |
| PySCF, random unconverged amplitudes | 5.551115123125783e-17 |
| PySCF, same-C H2 | 0 |
| PySCF, same-C water | 6.661338147750939e-16 |
| PySCF, same-C LiH | 1.1102230246251565e-16 |

Per-element `atol=1e-11, rtol=1e-10` and absolute energy/residual gates of
1e-8/1e-9 are enforced. No tolerance was relaxed. Native provider tests
independently check same-C fixed amplitudes; fresh native RHF runs are also
executed for H2/water/LiH. Fresh H2 additionally compares with determinant
algebra using independent saved AO integrals transformed by the actual new C.
Fresh water/LiH are finite-value/identity bridge smoke tests, not independent
converged CCSD endpoint evidence.

Reference generation used `python -m tools.generate_cc_references --output
build/cc-generation-1.json`, then the same command with output
`tests/reference_data/cc/rccsd-a.json --compare build/cc-generation-1.json`.
Cases hash:
`708d2a8c329524c11104c1eea9681c56870855dab00f2cd8f6968354751f8b70`.
The committed fixture retains actual update denominators, level shifts, source
hashes, library versions and thread metadata. Program, independent-coordinate
maps and inventory identities are in `tests/reference_data/cc/equations-2o2v.json`.
`determinant-groups.json` contains a complete input/program/reference replay.
Other replay artifacts remain in the remote `build/cc-validation/equations/`.

## Prior interrupted implementation

The user identified `D:/Users/Lenovo/Desktop/My_ViveQC/wt-cc` after this task
had begun. It was inspected read-only and was never resumed or modified.
All 35 coefficients/contractions/operands agree exactly. Its separately
implemented interleaved-spin determinant polynomial oracle agrees groupwise
with the current alpha-before-beta oracle within 1.67e-16. File identities and
errors are in `prior-comparison.json`. Its checks informed the physical versus
coordinate metric test and fresh-reference independent H2 check. No second
public interface or duplicated implementation was imported.

## Resolved early failures and limits

Early tests exposed incorrect test assumptions about the interpreter's budget
exception (`ValueError`), the fixture root directory, and `export_rhf` returning
`(snapshot, diagnostics)`. Each was corrected and rechecked on qz. Windows
pytest's existing shared temp directory was inaccessible; the final local run
used a unique workspace basetemp. The first checksum text file had CRLF line
endings; an LF checksum list was regenerated before running the verified
snapshot. These were not numerical-equation failures.

Neither the full Python suite nor CUDA tests were run in this slice; the
listed related regression and all native CPU suites were run. The CPU CTest
named `cuda_eigensolver_policy` is a host policy test, not CUDA execution.
General was already running for shared work and remains available; this task
did not create/restart it or schedule a shutdown. Its prior uptime exceeds
18 hours; no new 18-hour retention guarantee is claimed.

## Follow-up boundaries

- **B:** define the physical T2 projector in the documented spatial coordinates;
  add traced ladder/ring/permutation terms and shared intermediates to the
  inventory; independently check random R2 and restricted pair symmetry;
  establish full R1/R2 equivalence before/after optimization.
- **C:** add an internal CPU solver consuming the full physical residuals;
  implement MP2-like initialization, denominator diagnostics, damping, separate
  CC DIIS and nonfinite/nonconvergence handling; require both energy and
  independently recomputed residual convergence. Validate two-electron FCI and
  multielectron same-Hamiltonian PySCF endpoints using fresh VibeQC HF. Preserve
  replayable histories and reference/integral identities.
- GPU/#149, (T), Lambda and complete gradients retain their own scope.

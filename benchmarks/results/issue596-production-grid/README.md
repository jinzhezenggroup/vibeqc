# Issue #596 production-grid qualification

This archive retains the independent accuracy/cost and native CPU qualification
for the production `GridPolicy` introduced by #596. The measured source tree is
commit `ad815e87ccfba73a94f2fe080414b209c0e6ec12`, rebased on
`origin/master=ab6938a671205dd612682f75939143d20b51707e`; the qualification
worktree was clean.

## Independent accuracy and point cost

`grid_policy_convergence.json` was produced by
`benchmarks/grid_policy_convergence.py` with PySCF 2.14.0. PySCF independently
solves SCF and full analytic grid-response gradients; it does not execute the
VibeQC native KS solver. The 96脳32脳64 dense oracle was itself checked against
120脳40脳80 before production profiles were evaluated.

| Profile | Grid | max abs dE (Eh) | max abs dF (Eh/Bohr) | Points / dense |
| --- | --- | ---: | ---: | ---: |
| LDA standard | 54脳16脳32 | 6.9671913e-7 | 2.9196122e-5 | 0.140625 |
| LDA tight | 64脳20脳40 | 3.7613606e-7 | 1.1427903e-5 | 0.2604167 |
| PBE standard | 54脳16脳32 | 7.0505067e-8 | 2.0743213e-5 | 0.140625 |
| PBE tight | 72脳24脳48 | 3.0826324e-7 | 4.0551269e-6 | 0.421875 |
| PBE legacy control | 48脳16脳32 | 1.9340074e-6 | 5.6333010e-5 | 0.125 |

The machine gate passed with no failures. PBE standard improves both energy and
force over the historical 48脳16脳32 control while using only 1.125脳 its point
count. Tight LDA/PBE both improve the standard force error. Dense-reference
stability was 9.1078789e-10 Eh / 3.1718549e-7 Eh/Bohr for LDA and
1.2948817e-8 Eh / 6.0336236e-7 Eh/Bohr for PBE.

SHA-256:
- `grid_policy_convergence.json`:
  `5281a3e35ec54fae45f50f0b275cb22732b426e4034a10df58d50e3a38e8b447`
- the shared raw convergence log has the same hash because the driver emits the
  retained JSON to stdout.

The qz shared raw-evidence root is
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0596-qual/ab6938/ad815e87`.

## Native CPU qualification

A clean Release CPU build of the same source completed all **43/43 CTests**.
The focused production-grid native suite then passed **11/11** cases in
341.30 s:

- LDA/PBE RKS 脳 standard/tight against PySCF's independent unpruned
  `(120 radial, 974 Lebedev angular)` energy + grid-response force oracle;
- LDA/PBE UKS 脳 standard/tight on the open-shell doublet against the same
  independent dense-grid oracle;
- light-element production energy/force endpoints; and
- the representative Fe/H transition-metal production energy/force endpoint.

The raw JUnit record remains in the qz shared evidence as
`production-grid-native-cpu.xml`; its SHA-256 is retained below.
The eight pytest warnings only report that `record_property` is not serialized
by xUnit2; no test or scientific gate failed.

Source-bound hashes:

| Artifact | SHA-256 |
| --- | --- |
| Release CPU `libvibeqc.so` | `141ba89c91af238f14ec44eed4ac1c22a192da9fec38625fcd3fe37b7389d993` |
| CPU configure log | `c98e212c29dd3807f6b693b010730397057c2c3dee08c7b622de11e9236df331` |
| CPU build log | `324e8bfc6d2d63af9c2a4bda4c20e3cdd238bb4388665f9647b6c273666d1311` |
| 43-test CTest log | `a55824d88daad28523ccb547d74aa895a780d4d30e7cbe5c846a60c16e6ad84f` |
| Native CPU JUnit | `00df1de8362f718ed6b1379e6b2dbf79330d3760e1e34bfe58ebbf91dba45d51` |
| Native CPU pytest log | `3e39d7fb587ee428335cbaae053ec3c1891ddc598768ab77cb03ed27c1d7a27c` |

## CUDA boundary

The real-NVIDIA stationary CUDA test is an explicit finite-Slurm gate and
requires an RTX 5090 / `sm_120` target. It must bind the **published PR head**,
not the pre-publication measured source commit above. Therefore the final
RTX 5090 result is retained in PR #615's protected review/check record after
this evidence-only commit is published. CuMetal remains an independent Apple-GPU
runtime regression and is not treated as a substitute for that NVIDIA gate.

The evidence commit changes only this retained-results directory; production
code and numerical tests remain the exact measured tree above.


## Final integration bridge

The source-qualified production tree above was measured at `ad815e87` on
master `ab6938a6`. Before publication, upstream master advanced to
`097ee4ac` through #616. That upstream delta is confined to CC triples-response
notes, CI wiring, tests and `tools/vibeqc_cc`; it does not touch any GridPolicy,
KS snapshot/stationary consumer, native DFT grid or #596 qualification file.

The previously generated pre-commit.ci formatting fix was also applied before
the final publish. Against the qualified source, all #596 production files are
byte-identical except `python/vibeqc_compiler/dft/grid.py`, whose sole change is
removal of a blank line; Python AST equality was checked explicitly. The other
production C++/Python files were checked with an exact Git diff.

On integration head `686e27e0`, Ruff passed for the grid/snapshot/CPU/CUDA
qualification set, 12 focused policy/provenance/capability regressions passed,
and `git diff --check origin/master...HEAD` was clean. The commit adding this
section is evidence-only. The final real-NVIDIA gate must still bind the
published PR head, including this evidence-only delta.

### Post-CI master bridge

After that publication, master advanced once more from `097ee4ac` to
`f1b64173` through #651. That delta changes only Array API documentation,
`python/vibeqc_compiler/array_api/{capabilities,namespace}.py`, and the matching
Array API frontend test; it has no overlap with #596 production, snapshot,
stationary-gradient, grid-policy or qualification files. The #596 branch was
rebased cleanly across it.

CI then exposed an evidence-transport-only defect: pytest-xdist/execnet cannot
serialize NumPy scalar subclasses stored by `record_property`. Commit
`ae6743c4` normalizes those retained properties to Python `int`/`float` values.
A two-worker xdist regression using NumPy scalar inputs passes, and this change
alters neither the production grid policy nor any numerical threshold/result.

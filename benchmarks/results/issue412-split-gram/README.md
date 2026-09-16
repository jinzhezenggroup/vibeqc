# Bounded split occupied Gram: negative endpoint result (#412)

The split4 candidate saves about 12.4% of net Gram time on the four real
768/768/rank-160 fixed-U inputs, but it fails complete endpoint qualification.
The first matched-density 768 force pair takes **three SCF updates with SYRK
and five with split4**. The observed complete calls take 2.559862 s and 2.979252 s,
respectively. Both pass independent energy/force gates. This single changed-work
pair is a retained failed work gate, not a seven-pair performance estimate.

The original runner stops at this failure. Automatic SYRK remains unchanged;
the temporary `split4` production control and implementation are removed. No
new density, looser convergence tolerance or wider split search is used to
rescue the experiment. Source and validation patches retain the exact candidate
for reproduction. #404's accepted derivative mapping remains unchanged.

## Fixed-U and integrated qualification

`stage-a-summary.json` records extrema and all four target ratios.
`experiment-v1/` retains all twelve fixtures, six candidates, seven interleaved
samples each, separate component measurements, numerical gates, true work and
resource reservations. `inputs.json` binds binary inputs and executables to
hashes; large U/K binaries remain local. Real capture used source
`796ce85a607dfde85bf6bafd20575d7c538fdad1`, the standalone pre-promotion #404
baseline. Capture timing is intrusive and is not performance evidence.
The first capture shim reported graph-capture errors; the second skips graph
construction and retains successful eager molecular snapshots.

Original capture/trial sources have `.txt` appended to preserve their measured
bytes under formatting hooks. Restore the original names to reproduce them;
`stage-a-retention.json` records names and hashes. Scripts retain historical
workstation paths as provenance. Adapt those paths and regenerate fixtures for
a new campaign. All GPU commands require finite Slurm allocations on `main`
with `--gres=gpu:5090:1`, preserving scheduler device visibility. Stage A used
job 9817. `stage-a-sanitizers.json` records local raw-log hashes and error summaries;
that initial memcheck did not request full leak checking.

Full-GEMM partials compute both triangles, roughly twice SYRK's leading FLOPs.
Stage A reserves capacity for the largest candidate and reports candidate-specific
partial storage separately. The integrated candidate borrows 18 MiB from charged
`exchange_intermediate`, preserves raw A/final U and adds no owned allocation.
Its existing reducer starts at zero, adds four partials, then accumulates into
zero K; its source-level addition count differs from Stage A's custom reducer.

The candidate passes two CPU policy tests, the native DF suite and 68 Python GPU
tests. Native memcheck with full leak checking and initcheck report zero errors.
`validation.json` binds jobs 9823/9824 and actual split/tail/fallback observations.
`validation-harness-failure.json` preserves the initial UHF assertion failure:
an already captured occupied replay emits no new per-product records. Energy
and force checks had passed; the corrected provenance assertion passes without
changing native code or scientific gates.

## Complete endpoint stop

`endpoint-protocol.json` was declared before measurement. Job 9825 holds merged
#415's derivative mapping fixed, restores identical checkpoint D, disables warm
updates and primes every policy selection. It requests seven paired samples,
>=1% complete 768 force saving with bootstrap upper ratio below one, a 3%
regression margin and unchanged independent numerical/operator-work gates.

The complete 384 force medians are 0.697501/0.697988 s (auto/split4); energy
medians are 0.284136/0.284313 s. Both regressions pass. `endpoints/` preserves every
completed sample, including the two 768 rows that trigger the stop. Formatting
only compacts scalar arrays; `endpoint-raw-files.json` binds original run bytes.
The magnitude/stability gates are not evaluated after the work gate fails.
96/192 and qualifying 768-energy runs are not continued after that stop.

`changed-work-protocol.json` declares the separate intrusive follow-up for energy,
forces and sampled process resources. It keeps normal convergence and scientific
gates, observing the changed iteration count instead of requiring three updates.
Its timings are excluded from performance qualification; it cannot replace the
missing seven matched-work pairs. `candidate-analyze.py.txt` preserves the original
complete-campaign analysis rather than silently weakening its assertions.

Job 9826 reproduces the changed branch for **both** complete energy and force
calls. Each policy accepts one final physical Fock with zero final corrections
or candidate rejections. The force projection lease is reused once in both arms.

| Observed work per complete call | SYRK | split4 |
| --- | ---: | ---: |
| SCF updates | 3 | 5 |
| J builds | 4 | 6 |
| K builds | 4 | 6 |
| Eigensolves, including density-seed factorization | 4 | 6 |
| Leading Gram FLOPs over all K builds | 290,287,779,840 | 869,730,877,440 |

Maximum follow-up energy/force errors remain 3.229e-11 Eh / 1.620e-10 Eh/Bohr.
`work.json` records scopes, final-state observations, each occupied Gram call
and hashes of the complete local traces. Final physical work is included once;
four partial products inside a split build do not count as four K builds.

Both observable processes reach a sampled device-residency peak of
24,631,050,240 bytes, including initialization, context/modules and both policies.
This is a sampled lower bound, not an allocator peak or a peak attributable to
either arm. The candidate's 18 MiB partial storage is borrowed and its additional
application-owned allocation is zero; no whole-process memory improvement is
claimed. `changed-work/resources.json` records host RSS/high-water readings,
sampling definition and command identities; all sampled readings are retained.
All intrusive timings remain excluded from the clean comparison.

Recompute the rejection, numerical gates and work/resource summary without a GPU:

```bash
python benchmarks/results/issue412-split-gram/analyze.py
```

## Identity and reproduction

`candidate-build.json` freezes Release CUDA 12.9.1/sm_120, fast compile and AOT
off, generated files, source identity and library hash. The measured implementation
is baseline `25e8efebb069e6067ba3649f220c37e844ecc8f6` plus
`candidate-source.patch`; its native/compiler contents are identical to commit
`258f76b2b9201b4690cda4a48b1c8f9a39f8aedc`. That commit is the attempted candidate,
not an automatic promotion. `candidate-validation.patch` reconstructs its tests.
Apply both patches to the pinned baseline in an isolated checkout and finish
compilation before timing. `run-endpoints.sh` reproduces the original campaign
and deliberately exits nonzero when the fixed-work requirement fails.

Historical checkpoints and large input arrays remain local. Every endpoint
retains checkpoint and density hashes; regenerating a density starts a new
campaign and does not reproduce the same SCF branch by assertion. The intrusive
follow-up pins the frozen candidate library while the checkout retires that
candidate. Its library/source hashes and reconstruction patch identify executed
code; `git_head` also records the launcher's evolving checkout and must not be
mistaken for a relabeled native implementation.

The [decision note](../../../.agents/notes/rejected/2026-09-17-split-occupied-gram.md)
records the rejection. #409 requires a separate packed-representation experiment;
#206 still owns fresh matched stock GPU4PySCF comparisons. No independent savings
are added into a combined or external performance claim.

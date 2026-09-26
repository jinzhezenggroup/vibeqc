# Fix the 768-AO response scaling cliff

With this PR's default selectors, the 768-AO complete warm energy+force endpoint
is **12.084 s, down from 114.804 s (9.50x faster)** with the previous optimized
shell combination.
This is the 96-atom water 32-mer, 768 orbital / 768 auxiliary spherical def2-SVP
AOs, RHF, batch 1 on RTX 5090. All five paired force/energy comparisons pass
unchanged **1e-8 Ha/Bohr / 1e-9 Ha** limits. Maximum candidate errors are
1.540e-10 Ha/Bohr and 4.138e-11 Ha.

| Native version | Cold complete endpoint (s) | Warm median of five (s) |
| --- | ---: | ---: |
| Previous shell implementation | 1052.291 | 114.804 |
| CPU preparation removal only | 208.947 | 113.506 |
| This PR, default selectors | 107.069 | 12.084 |

The warm improvement attributable to the response change is
**9.39x** versus the preparation-only build. Every native
sample in all three records uses the same three-iteration warm branch, with a
fixed engine-local post-cold density and deterministic ABBA ordering.
The fresh GPU4PySCF ordinary median is 2.349 s:
the remaining ordinary latency ratio is **5.14x**.
The iteration branches are retained per sample; the ordinary ratio is **not
an iteration-matched cross-engine speed claim**. The earlier 2.14-s GPU4PySCF result used a different
mixture of iteration branches. Full raw numerical samples, branches, inputs,
versions and compaction provenance are retained beside this report.

## What changes

The bounded panel loop previously recomputed every `R_Q = D^T A_Q D` inside
every auxiliary P panel. At 128 MiB, 768 AOs force 77 panels: each of the 768 Q
columns receives its two dense GEMMs 77 times. The fix borrows the resident
value plan's three existing J/K temporaries, uploads raw A once, evaluates each
Q projection once, and uses GEMM for the complete metric adjoint and exchange
weights. The existing generated coordinate-derivative consumer and spectral
Frechet map are reused, including retained/discarded metric directions.

| Separate 768-AO trace | Panel response | Default borrowed response |
| --- | ---: | ---: |
| Auxiliary consumer panels | 77 | 1 |
| AO projection GEMMs | 118,272 | 1,536 |
| Projection CUDA time (s) | 82.776 | 1.001 |
| Raw H2D bytes | 282,615,349,248 | 3,623,878,656 |
| Raw DMA time (s) | 10.543 | 0.243 |
| Host panel gather elements | 36,282,433,536 | 0 |

Final-build trace counters confirm the same NVIDIA work counts after the
CuMetal compatibility fix. CUDA durations and memory samples in this report
come from the preceding v4 Nsight capture; its exact source identity is retained.

The candidate submits one bulk raw upload. New response scratch remains
**132,236,832 bytes, within 128 MiB**. The three borrowed
J/K buffers total **10.125 GiB**, already allocated and charged to the resident
value plan; this is not a 128-MiB total-memory algorithm. One-second process
memory samples during the separate capture peak at
20,420 MiB; this sampled value is a lower bound,
not an allocator high-water mark. No tensor-sized response allocation or full
nuclear-coordinate derivative tensor is added. Same-stream ordering and bridge
draining return the scratch safely for later SCF/property/geometry replays.

The completed workspace ablation in [panel-scaling.json](panel-scaling.json)
is a causal baseline, not the fix: 128/256/512/1024 MiB gave
113.506/54.394/35.128/26.839 s and 77/32/15/8 panels. The four-point empirical
fit is `T = 15.884 + 1.26009 * panels` seconds, `R^2 = 0.999014`.
Increasing the allowance all the way to 8 GiB gave 17.533 s. The new 128-MiB
additional-scratch route is faster because it also removes repeated raw
traversals and uses complete auxiliary GEMMs. Inclusive host gathers overlap
GPU work and must not be added to CUDA durations.

## Default selection and limits

Automatic selection is deliberately limited to the measured 768/768-AO,
single-system RHF, sm_120 resident shape, with full J/K scratch capacity.
`VIBEQC_DF_RESPONSE_STORAGE=jk-scratch` opts other eligible plans in;
`panel` preserves the earlier algorithm and `auto` is the default.
Explicit incompatible algebra/schedules and attribution probes keep panel
execution in auto mode. Explicit borrowing rejects incompatible controls or
partial/source-backed scratch rather than overrunning it. The standalone
storage selector supplies BLAS unless explicitly overridden. Providers without
`cublasDgeam`, including CuMetal, reuse the existing J/K gather for raw layout
conversion. NVIDIA retains GEAM through a compile-time capability check.

Removing unused CPU metric/transformed-B preparation fixes a separate cold
cost and preserves the CPU oracle. A positive public DF budget retains its
bounded value/response split; the diagnostic response-budget override is
rejected if that public budget is positive.

Parity remains open: this capture spends
4.810 s in generated three-center
derivatives and 3.298 s in SCF RI-K GEMMs. Those are the
largest remaining components. The 192--384-AO default stays on its already
qualified route. This PR does not close #206/#308 or claim a constrained-memory,
occupied-space response algorithm. Changed-geometry correctness is covered
at smaller RHF/UHF sizes; 768-AO changed-geometry wall times remain unmeasured.

## Validation and reproduction

Slurm job 9617: **15 focused GPU tests and the final five-repeat
768-AO endpoint plus complete counter trace**. Before the portability-only
capability guard, job 9616 passed **193 existing GPU regressions and
3 Compute Sanitizer cases with zero errors**. Focused cases cover RHF/UHF,
batch 1/4, dense/occupied SCF, force/energy/force reuse, geometry changes,
finite discarded metric modes against CPU/finite differences, bounded-plan
rejection, and recovery after invalid controls. The benchmark/timeline suite
has **44 passing tests**, and **98 evidence/ownership tests** pass after refreshing
the maintained ownership snapshot. Repository formatting and dependency checks pass.
A supplemental final-library generic control (job 9618, explicit
panel/generic/warp/scalar/pageable) completed cold execution in 1282.190 s,
then was stopped at the user's request during unmeasured priming. No generic
warm sample or complete paired numerical record was produced, and the queued
explicit-combination follow-up did not start. These cold-only observations are
retained in the summary and excluded from acceptance and speedup claims.

No force tolerance is relaxed. Historical batch-four provider tests now use
32 MiB for their successful comparison; their 8-MiB failures were reproduced
on the old library and are explicitly retained as out-of-memory tests.

Build the production library with CUDA 12.9.1 and sm_120 (no GPU is needed to
compile). With a Python environment containing VibeQC, PySCF 2.14.0,
GPU4PySCF and the matching CUDA libraries, run the clean endpoint without any
ambient `VIBEQC_DF_*` selectors:

```bash
cmake --preset cuda-release-sm120
cmake --build --preset cuda-release-sm120 --target vibeqc --parallel 4
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:15:00 env PYTHONPATH=python:. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VIBEQC_LIBRARY="$PWD/build/cuda-release-sm120/libvibeqc.so" \
  python -m benchmarks.compare_gpu4pyscf_batch \
  --case water-32mer-4s4-def2-svp-spherical --batch 1 --repeats 5 \
  --density-fitting cuda --density-fitting-memory-budget-bytes 0 \
  --maximum-energy-error 1e-9 --maximum-force-error 1e-8 \
  --max-iterations 100 --energy-tolerance 1e-12 --density-tolerance 1e-10 \
  --reference-gradient-tolerance 1e-10 --output /tmp/768ao-default.json
```

A separate `srun` allocation can run `benchmarks.issue308_response_timeline`
with `--aos 768 --batch 1 --library ... --reference /tmp/768ao-default.json
--component-trace --nsys --output /tmp/768ao-profile`, under `nsys profile
--trace=cuda,nvtx --cuda-graph-trace=node --cuda-event-trace=false --sample=none
--cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop`.
Export that report with `nsys export --type=sqlite` and reduce it with
`python -m benchmarks.df_response_timeline <capture.sqlite> --output <summary.json>`.
The trace runner verifies source/library identity, complete force parity and
actual work counts. Profiling and memory sampling are excluded from clean timings.

For the older implementations, use the base commits and exact patches in
[provenance.json](provenance.json). Their 768-AO runs explicitly set
`shell / compact / blas / pinned-panels`. The preparation-only staircase sets
`VIBEQC_DF_RESPONSE_BUDGET_BYTES` while keeping the public DF budget zero.
Large raw traces/libraries remain outside Git; their hashes, numerical samples,
compact actual-work summaries and [ownership delta](ownership-delta.json) are retained.
The conservative CUDA ledger counts +177/-12 scientific adapter lines and
+0/-0 runtime CUDA lines; no derivative kernel family or generated capability
is added. The bounded panel route is still needed where full scratch is absent;
the separately recorded serial metric-dot oracle retains its existing retirement gate.

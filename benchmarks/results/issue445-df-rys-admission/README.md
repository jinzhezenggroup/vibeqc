# DF derivative lowering admission: bounded #445 result

This campaign removes the redundant `NAO in {384,768} && Naux == NAO` test
before generated derivative class selection. It does not change mathematical
kernels or the gradient bridge's separate shell/packet/packed-layout admission.
Issues #444/#445/#435 remain open for that separate work.

## Fixed-work result

One optimized baseline library, Slurm job 9993, seven interleaved clean pairs
per case, one frozen density per case, a complete untimed prime before every
measured policy transition, and separate post-timing diagnostics:

| AO / auxiliary | Original auto, ms | Qualified candidate, ms | Interpretation |
| --- | ---: | ---: | --- |
| 96 / 96 | 79.228819 | 80.630187 | Same generic route; startup drift, inconclusive |
| 192 / 192 | 142.749912 | 92.375393 | 35.29% lower full energy-plus-force time |
| 384 / 384 | 333.150380 | 333.041848 | Already selected Rys; unchanged control |
| 768 / 768 | 2141.591773 | 2141.917352 | Already selected Rys; unchanged control |

Every sample passes the independent 1e-9 Hartree / 1e-8 Hartree/Bohr gates.
Across the 56 clean samples, maximum errors are 1.14e-11 Hartree and
3.31e-12 Hartree/Bohr. Both arms take two updates at 96 AO and three at every
other size. All operator/response/transfer and logical shell/primitive/component
work counters match after excluding elapsed timings and owner generations.

At 192 AO the independently instrumented complete force stage decreases from
102.966205 to 56.245295 ms. Both arms visit 451,632 shell triples, 2,783,856
primitive products and 15,424,032 Cartesian component products, consume
7,077,888 public weights, and execute three occupied response projections.
Post-call device residency is 6,786,383,872 bytes in both arms; this is a resident
observation, not a sampled peak. Native charged peaks remain in each raw sample.

These are internal fixed-work comparisons, not fresh GPU4PySCF timings. The
retained independent reference arrays are checked against exact geometry,
basis fingerprints, settings and shapes by the existing endpoint runner.
Do not subtract historical external medians or infer a new 384/768 win. Practical
JKFIT latency, batch-4, changed-geometry timing and constrained-memory throughput
are not qualified by this bounded campaign.

## Build and provenance

Baseline: `e215b30685f8a36ef0cc5772c9166837527f64a2`, with uncommitted test-only
regressions and unchanged compiled source. Library SHA-256:
`1bbb442a5faedb1c5d123c42ad5f3bef19dd48aefc27a3fc82718de2a5f461a8`.
The patch's exact compiled-source change is retained as `candidate.patch`.

Hardware: NVIDIA RTX 5090, 32,607 MiB reported memory, sm_120, driver 580.95.05.
CUDA toolkit 12.9.1, GNU C++ 11.4, Release `-O3 -DNDEBUG`, FP64. Fast compilation
is disabled. The Direct four-center AOT bundle is disabled in both arms; this
neither disables nor substitutes the generated DF derivative/value kernels.
Native DF thresholds and physical-state checks are unchanged. Slurm controls GPU
visibility. OMP/OpenBLAS/MKL each use eight threads. The independent runner
uses the existing Python 3.13 environment with NumPy 2.5.3, PySCF 2.14.0,
CuPy 14.2.0 and GPU4PySCF 1.8.1; no new external timing is attributed to it.

No compiler jobs or numerical test suite overlap the clean timing campaign.
Numeric-only qualification may run concurrently with compilation and is never
reported as a performance sample.

## Reproduction

Use a Python environment with the repository benchmark dependencies and a
supported toolkit. Build both the pinned baseline and this patch with identical
flags, including `VIBEQC_ENABLE_AOT_SHELLS=OFF`, `VIBEQC_CUDA_FAST_COMPILE=OFF`,
`CMAKE_BUILD_TYPE=Release`, `VIBEQC_ENABLE_CUDA=ON` and
`CMAKE_CUDA_ARCHITECTURES=120`. Set `VIBEQC_LIBRARY` to the resulting library,
`PYTHONPATH=python:.`, the CUDA runtime search path, and the thread counts above.
Run through a finite allocation, preserving scheduler-assigned GPU visibility.

On the baseline, run the following for each retained case (96, 192, 384, 768),
substituting the matching name from `benchmarks.df_policy_endpoint.CASES`:

```bash
srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:5090:1 --time=00:20:00 \
  python -m benchmarks.df_policy_endpoint --aos 192 --repeats 7 \
  --control VIBEQC_DF_SHELL_POLICY --policies auto candidate \
  --components-after --warm-checkpoint-out .artifacts/frozen-192.vqckpt \
  --reference benchmarks/results/issue377-379-df/gpu4pyscf/water-octamer-s4-def2-svp-spherical.json \
  --output .artifacts/pilot-192.json
```

On the patched binary, compare `--policies legacy auto` at 192 AO. Here legacy
is exactly the previous automatic lowering, not a weakened external engine.
Restore the saved checkpoint with `--warm-checkpoint-in` and `--skip-cold` plus
the same independent reference, and retain the source patch using
`--source-patch`. The new automatic route must also pass
`tests/python/test_df_derivative_admission_cuda.py` in a finite allocation with
`VIBEQC_RESOURCE_CUDA_TEST=1` and without forcing the candidate policy.

## Final automatic path and tests

The patched library's SHA-256 is
`14da088e24c8ad9707659c889785e3643ba8b006b88e04645476e5274c108659`.
Slurm 9995 restores the byte-identical baseline density and measures seven
interleaved `legacy`/`auto` pairs at 192 AO. Medians are **143.010972 and
92.784583 ms**, respectively: **35.12% less time, 1.5413x throughput**.
All samples take three updates and pass the independent gates. The final
intrusive force-stage observations are 103.660092/56.121460 ms. Generated
production policy and Rys mathematical header SHA-256 values are unchanged:

- `generated_df_production.hpp`:
  `61e2f925637ae73ed8873a638000e9be73c5fd7899cddaba21296e06d107fbe7`.
- `generated_df_rys_shell.cuh`:
  `0916a69446038662dea2acf9006699e048425afe44cdecae6cfd3763a61e83b8`.

The restored density SHA-256 is identical across libraries. Comparing the
baseline candidate with final auto gives zero energy difference and a maximum
force difference of 6.76e-14 Hartree/Bohr across corresponding repeats. Library
size changes from 167,828,272 to 167,819,760 bytes; generated mathematics does
not grow.

| Completed validation | Result | Binary / policy |
| --- | --- | --- |
| Independent host generation, Rys and benchmark-contract tests | 224 passed | Full dependency environment; no GPU needed |
| Full shell, practical-response and automatic exchange suites | 104 passed | Frozen baseline, explicit candidate; Slurm 9994 |
| New panel/group/packet admission regression | 9 passed | Final patched binary, auto -> legacy -> auto; Slurm 9995 |
| CUDA ownership, pairing and publication suites | 51 passed | Current maintained-source snapshot |

The 104 scientific GPU regressions include cold/warm/changed geometry,
RHF/UHF, practical unequal auxiliary bases, original/tighter density requests,
and constrained-budget admission/rejection. They qualify numerical behavior,
not timings. The final nine tests independently verify the new automatic
selection and the auxiliary f fallback; they do not substitute an explicit
candidate for the claimed automatic endpoint.

Retained unsuccessful checks: the new 192-AO regression intentionally fails on
the baseline's missing Rys selection after passing energy/force checks. An
initial host run had 54 skips for missing mpmath; the full 224-test rerun has no
skips. The first ownership test detects the changed source snapshot; regenerating
the same artifact subset from the candidate build makes all 51 tests pass.
None of these is reclassified as a passing original run.

## Retention

Each JSON preserves all clean timing samples, complete energy/force arrays,
all-repeat numerical errors, convergence/residuals, metric/resource diagnostics,
source/library/runner/reference identities, and separate component records.
Repeated identical basis metadata is retained once at the top level. Per-class
trace detail is omitted except actual Rys selections and resource maxima;
semantic work and region timings remain. `retention` records each original
JSON's bytes/hash and the exact omissions. No numerical sample is rounded or
removed. Full logs, checkpoints, binaries and transient traces are not committed.

Agent: ChatGPT
Model: GPT-6 Astra Pro

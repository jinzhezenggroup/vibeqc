# Issue #949: matched 768-AO DF response qualification

This directory contains a bounded protocol for the remaining complete
energy-plus-force comparison. It is not a completed qualification receipt until
the raw output, exit sentinel, device and library identity, numerical gates,
work counts, and both policies have been audited.

The historical RTX 5090 campaign in
`../acceptance-closeout-20260922/df-768-panel-ablation.json.gz` completed its
strict independent PySCF 2.14.0 CPU oracle and one BLAS sample, then reached a
1200-second command limit with zero completed scalar samples. That GPU timing is
not pooled with measurements from another device. The adapter here reuses only
the CPU oracle after checking geometry, charge, spin, bases, AO/auxiliary counts,
and solver thresholds. The active runner checks cold, prime, and measured
energy/force values against the original `1e-9 Eh` and `1e-8 Eh/Bohr` gates.

`run_qz.sh` takes `probe`, `build`, or `measure` as its sole argument. Every Job
sets absolute `VIBEQC_SOURCE_DIR` and `VIBEQC_RUN_DIR`, exact
`VIBEQC_SOURCE_SHA`, and its actual Job name as
`VIBEQC_BENCHMARK_ALLOCATION`. A measurement also declares
`VIBEQC_EXPECTED_ITERATIONS` and a unique `VIBEQC_RUN_LABEL`. Its optional
`VIBEQC_REPEATS` and `VIBEQC_COMPONENTS_AFTER=1` request interleaved clean
pairs followed by separate intrusive counters. Its optional
`VIBEQC_WARM_CHECKPOINT_IN` selects a previously saved, native-validated warm
density checkpoint and a full untimed checkpoint replay. Measurement also
checks `VIBEQC_BUILD_SOURCE_SHA` against the runner source for unchanged
`src/` and `python/`, plus `VIBEQC_LIBRARY_SHA256` against the built binary.
The clean, pinned source is built with Release settings for `sm_90` and the
explicitly supported `portable_cuda` AOT profile. The binary SHA-256 is printed
and the build exit is retained in the run directory. `measure` refuses a
missing or failed build.
Its runner stores a phase marker before each prime and timed endpoint so that
an external time limit leaves a named incomplete stage and a lower bound,
rather than an apparent zero result. The output and frozen density checkpoint
are raw artifacts outside the reviewed `benchmarks/results/` tree.

The matched arms use one prepared calculator, geometry, basis, native library,
GPU allocation, frozen post-cold density, panel storage, FP64 native endpoint,
thresholds, and a declared fixed SCF update count. Each arm is primed before
its clean sample. The only requested arm control is
`VIBEQC_DF_RESPONSE_ALGEBRA=blas|scalar`; the cold solve uses BLAS. This control
selects response algebra beyond the charge dot as well. Any complete-endpoint
ratio is therefore a **response-algebra ablation**, not a charge-only causal
speedup. A completed scalar sample and separate counters are needed before
claiming the #949 performance acceptance item. A timeout remains a timeout
lower bound, not a measured speedup or a reason to close #949.

## Completed H100 evidence

The native library was built from clean `fbd0601a183e02c013cc8f83e087bf7de70f486c`
with CUDA 12.9.86, Release, `sm_90` and `portable_cuda`. Its SHA-256 is
`fbebb4739eec1bca68213151a359c29965fac48f3d69fcf113ade0d3708ca8cd`.
The seven-pair runner was `0719d807e679b363179dc5145da6c24d4e8668f8`;
`src/` and `python/` did not change between those commits. The actual device
was an H100 80 GB HBM3. The selected single-GPU quota cost 1 point/hour. The
highest permitted request was priority 4; the platform reported effective
`NORMAL 20`, which was not inferred from the request. Job
`vibeqc-949-seven-0719d807-20260923` ran from 19:37:05 to 19:56:57 CST,
reported `SUCCEEDED`, and wrote an exit sentinel of 0.

The first H100 pilot correctly rejected the historical six-update expectation:
its one BLAS sample used **three** updates, and scalar was never started. That
failed receipt is retained separately. A subsequent one-pair run and the
seven-pair run used an explicit three-update gate and the same frozen density
checkpoint (SHA-256
`cb2db0cee95ebebe0827cc25a0f4f2226490012d77274dd75beff26331e8c347`).
The archived checkpoint's source geometry hash and the current 768-AO case's
target geometry hash both equal
`4fafd0dc13b446a8eede891ae30815e5d92b010358acbeba74a131aab2ce7378`;
this was checked after the run from the exact checkpoint bytes. The reusable
runner now rejects a different geometry before loading its density.
The native runner's source patch was empty. The exact historical PySCF 2.14.0
CPU result was reused after strict scientific-input checks; PySCF was not run
anew in these H100 Jobs. All cold/replay, prime and measured energy/force
checks use `1e-9 Eh` and `1e-8 Eh/Bohr` gates. The 14 clean samples and both
diagnostic passes converged with exactly three updates; maximum clean errors
were `3.775e-11 Eh` and `7.617e-12 Eh/Bohr`.

| Same-H100 panel response algebra | Seven clean E+F samples / s | Median / s | Range / s |
| --- | --- | ---: | ---: |
| BLAS | 11.792944, 11.789978, 11.832563, 11.750249, 11.769854, 11.832708, 11.803907 | 11.792944 | 11.750249–11.832708 |
| Scalar diagnostic | 42.973720, 43.000329, 43.053864, 43.096414, 43.071148, 42.983018, 42.976655 | 43.000329 | 42.973720–43.096414 |

The policy order alternated `BLAS→scalar` and `scalar→BLAS` by repeat. The
median of the seven same-repeat scalar/BLAS ratios was **3.6440** (range
3.6326–3.6677). The median absolute deviations were 0.0231 s for BLAS and
0.0266 s for scalar. This is a matched complete-endpoint response-algebra
comparison on this source and device. It is not pooled with the historical
RTX 5090 six-update results or used as a GPU4PySCF timing claim.

Post-measurement intrusive passes report the following semantic work. They are
not included in the clean medians.

| Counter | BLAS diagnostic | Scalar diagnostic |
| --- | ---: | ---: |
| Charge BLAS dots | 768 | 0 |
| Charge scalar dots | 0 | 768 |
| Charge dot elements | 452,984,832 | 452,984,832 |
| Density BLAS products | 1,536 | 0 |
| Density scalar products | 0 | 1,536 |
| Metric BLAS dots / GEMMs | 589,824 / 4 | 589,824 / 4 |

Both diagnostic arms report the same 15,069,892,981-byte DF plan device peak,
118,011,491-byte DF plan host peak, and 16,133,390,336-byte process GPU
residency. The algebra switch changes density products as well as charge dots,
so the endpoint ratio cannot isolate a charge-only speedup.

`manifest.json` binds the three compressed raw receipts to their original byte
counts and SHA-256 values. `matched-seven.json.gz` contains every clean sample,
prime, convergence record, independent error, resource peak, and separate
diagnostic counter. The original 555,128-byte JSON and checkpoint remain in
the task's distinct `/inspire/qb-ilm/.../issue-949-matched-768-20260923/`
directory; the checkpoint and native library are not committed to Git. To
verify restoration, decompress each archive and compare its size and SHA-256
with the corresponding `decoded_*` fields in `manifest.json`.

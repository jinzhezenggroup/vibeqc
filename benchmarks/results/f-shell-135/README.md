# Issue #135: f-shell acceptance on RTX 5090

All 34 classes pass release compilation, complete resource reporting, and independent GPU numerical validation for all eight RHF/UHF Fock/force direct/persistent entry points. The matrix contains 692 fixtures and 5,536 kernel executions. Maximum absolute analytic error is **1.12e-15**; the raw maximum relative error on nonzero references is 1.19e-9. Acceptance uses the recorded combined `atol=rtol=2e-10` floor, with separate translation and finite-difference guards.

The [index](index.json) hashes every complete per-class report. Source hashes and cached object hashes were checked against implementation `d20ffbe`. Numerical runs were collected during development and retain their original revision/dirty state; they were not relabeled as clean runs. All 57 generated files in the endpoint library exactly match final generator output.

## Correctness defects found by the matrix

- The ten `ff*` Fock emitters omitted the three-pair Wick matchings needed for six angular quanta. Value and gradient consumers now share complete matching enumeration; a CPU regression independently checks emitted coefficients against Gaussian moments.
- FFFF force multiplicities overflowed a 32-bit `13!` intermediate before division. Only the order-13 path now uses 64-bit intermediates, with an exact-integer regression through all orders 0–13.
- [Before/after error records](regressions.json) retain the failing source hashes and the corrected numerical results.

## Molecular endpoints

The required corpus is water/def2-TZVP and a hydrogen-bonded two-water fragment of the documented WATER27 tetramer. Both loaded basis implementations are inspected explicitly. The full tetramer is an extra stress case. Six unprofiled ABBA repeats use one frozen post-cold density and identical Fock settings; all measured baseline/candidate warm branches take one iteration per system. CPU PySCF supplies independent energies and analytic forces under the recorded strict settings.

| Endpoint | Batch | AOs / f shells | Baseline s | FPPS s | Speedup | Max energy error | Max force error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [water-def2-tzvp](f-water-b1.json) | 1 | 48 / 1 | 0.540281 | 0.537660 | 1.0049x | 3.41e-13 | 6.36e-11 |
| [water-def2-tzvp](f-water-b4.json) | 4 | 48 / 1 | 0.635768 | 0.631325 | 1.0070x | 4.12e-13 | 6.39e-11 |
| [water-dimer-def2-tzvp-spherical](f-dimer-b1.json) | 1 | 86 / 2 | 1.060846 | 1.054019 | 1.0065x | 8.81e-13 | 5.52e-10 |
| [water-dimer-def2-tzvp-spherical](f-dimer-b4.json) | 4 | 86 / 2 | 2.734946 | 2.698916 | 1.0133x | 7.39e-13 | 5.74e-10 |
| [water-tetramer-def2-tzvp-spherical](f-tetramer-b1.json) | 1 | 172 / 4 | 5.880455 | 5.798192 | 1.0142x | 4.55e-13 | 1.35e-10 |

The 2% non-regression budget passes at every required point. Active f-containing primitive work is 46.6% for water, 45.4% for the dimer, and 41.6% for the tetramer. The dimer has 86 spherical AOs and two loaded f shells; the tetramer has 172 AOs and four f shells.

**Unsupported boundary:** [tetramer batch 2](f-tetramer-b2-unsupported.json) and [batch 4](f-tetramer-b4-unsupported.json) fail. Each system has 4,528,320 logical tiles; batching crosses the runtime's 1 GiB descriptor limit and selects bounded execution without complete f-containing consumers. The [batch-4 sanitizer capture](tetramer-b4-sanitizer.txt) reports zero memory-access errors while the endpoint fails. These failures remain explicit and are not counted as acceptance or silently replaced by a generic result.

## Measured device time and selection

The separate Nsight captures contain one warm replay, excluding initialization and the PySCF reference. Generated entries have exact class attribution. Generic angular orders 9–12 necessarily contain f; orders 3–8 mix f with other classes, so their time remains an interval rather than an invented split proportional to primitive counts.

| Candidate capture | Fock f-time fraction | Force f-time fraction | Exact FPPS force |
| --- | ---: | ---: | ---: |
| [f-water-device-time](f-water-device-time.json) | 5.6–33.5% | 3.1–71.9% | 5.713 ms |
| [f-dimer-b4-candidate-device-time](f-dimer-b4-candidate-device-time.json) | 20.5–77.9% | 17.3–91.5% | 17.942 ms |

In the dimer batch-4 [baseline capture](f-dimer-b4-baseline-device-time.json), the generic order-5 force group takes 108.973 ms. With FPPS it takes 54.008 ms plus 17.942 ms for the generated FPPS entry. This demonstrates a class hotspot reduction alongside the 1.0133x unprofiled complete-endpoint result. Profiled wall timings are not used as speed claims.

FPPS ranks first among f-containing primitive work in the [measured ranking](promotion-ranking.json). Its existing force schedule is retained with the same independent numerical/resource/endpoint standard, scoped to the accepted fixed-topology workloads. No additional class is promoted: their numerical evidence is complete, but a generated-class endpoint A/B has not established a production benefit. The bounded f-shell regime remains unsupported. Fresh catalogs start with provisional acceptance until matching evidence is attached.

## Release costs and per-class states

CUDA 12.9.86, sm_120, release `-O3`, and bounded two-worker compilation are recorded in the index. All entries have legal measured occupancy. Nine classes are spill-free across all eight wrappers; spilling is reported separately from compile/launch validity. CUOBJDump includes an extra 1024 shared bytes on this target beyond PTXAS/runtime user shared allocation, and both measurements are preserved.

Current objects total 154.31 MiB and extracted cubins 141.14 MiB. Recorded per-class compiler durations sum to 3,643.06 seconds; cache verification performs no recompilation. The [release endpoint build](f-shell-build-cost.json) took 1,479.525 seconds, with a 99.37 MiB [library and exact source inventory](release-library.json). These are measured costs, not clean-build comparisons under controlled load.

| Class | Compile s | Object MiB | Max registers | Max spill load/store B | Max absolute error | Production / endpoint |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [fsss](fsss.json) | 4.97 | 0.97 | 148 | 0 / 0 | 2.10e-16 | Unselected; endpoint A/B not run |
| [fsps](fsps.json) | 5.86 | 1.18 | 158 | 0 / 0 | 1.33e-16 | Unselected; endpoint A/B not run |
| [fspp](fspp.json) | 8.26 | 1.56 | 162 | 0 / 0 | 1.80e-16 | Unselected; endpoint A/B not run |
| [fsds](fsds.json) | 7.92 | 1.56 | 156 | 0 / 0 | 2.18e-16 | Unselected; endpoint A/B not run |
| [fsdp](fsdp.json) | 13.98 | 2.26 | 168 | 160 / 144 | 8.20e-17 | Unselected; endpoint A/B not run |
| [fsdd](fsdd.json) | 40.93 | 4.30 | 168 | 2348 / 944 | 2.78e-16 | Unselected; endpoint A/B not run |
| [fsfs](fsfs.json) | 13.51 | 2.26 | 168 | 148 / 136 | 1.92e-16 | Unselected; endpoint A/B not run |
| [fpss](fpss.json) | 10.79 | 1.26 | 158 | 0 / 0 | 2.57e-16 | Unselected; endpoint A/B not run |
| [fpps](fpps.json) | 12.31 | 1.67 | 156 | 0 / 0 | 2.44e-16 | Force retained; endpoint passed |
| [fppp](fppp.json) | 16.64 | 2.47 | 96 | 1096 / 620 | 6.97e-16 | Unselected; endpoint A/B not run |
| [fpds](fpds.json) | 15.33 | 2.36 | 165 | 0 / 0 | 4.41e-16 | Unselected; endpoint A/B not run |
| [fpdp](fpdp.json) | 30.55 | 4.42 | 96 | 4028 / 1508 | 2.71e-16 | Unselected; endpoint A/B not run |
| [fpdd](fpdd.json) | 72.65 | 7.45 | 168 | 4980 / 1404 | 1.83e-16 | Unselected; endpoint A/B not run |
| [fpfs](fpfs.json) | 30.16 | 4.40 | 96 | 3968 / 1512 | 4.42e-16 | Unselected; endpoint A/B not run |
| [fpfp](fpfp.json) | 71.69 | 8.18 | 64 | 18548 / 10064 | 1.37e-16 | Unselected; endpoint A/B not run |
| [fdss](fdss.json) | 50.36 | 2.00 | 167 | 0 / 0 | 6.82e-16 | Unselected; endpoint A/B not run |
| [fdps](fdps.json) | 54.04 | 2.83 | 168 | 8 / 8 | 2.26e-16 | Unselected; endpoint A/B not run |
| [fdpp](fdpp.json) | 70.39 | 5.02 | 96 | 5712 / 3652 | 1.01e-16 | Unselected; endpoint A/B not run |
| [fdds](fdds.json) | 69.48 | 4.73 | 168 | 204 / 184 | 4.24e-16 | Unselected; endpoint A/B not run |
| [fddp](fddp.json) | 102.95 | 7.75 | 168 | 1116 / 892 | 1.80e-16 | Unselected; endpoint A/B not run |
| [fddd](fddd.json) | 78.95 | 3.35 | 168 | 404 / 404 | 3.66e-16 | Unselected; endpoint A/B not run |
| [fdfs](fdfs.json) | 99.13 | 8.21 | 96 | 11784 / 5392 | 2.53e-16 | Unselected; endpoint A/B not run |
| [fdfp](fdfp.json) | 79.68 | 3.35 | 168 | 388 / 388 | 2.91e-16 | Unselected; endpoint A/B not run |
| [fdfd](fdfd.json) | 222.39 | 5.46 | 168 | 1484 / 940 | 1.04e-16 | Unselected; endpoint A/B not run |
| [ffss](ffss.json) | 270.93 | 9.33 | 168 | 0 / 0 | 5.07e-16 | Unselected; endpoint A/B not run |
| [ffps](ffps.json) | 307.30 | 5.17 | 96 | 4236 / 2240 | 6.64e-16 | Unselected; endpoint A/B not run |
| [ffpp](ffpp.json) | 341.81 | 6.84 | 64 | 26148 / 17736 | 3.29e-16 | Unselected; endpoint A/B not run |
| [ffds](ffds.json) | 340.36 | 6.48 | 96 | 10124 / 5072 | 3.11e-16 | Unselected; endpoint A/B not run |
| [ffdp](ffdp.json) | 159.27 | 4.46 | 168 | 108 / 100 | 2.98e-16 | Unselected; endpoint A/B not run |
| [ffdd](ffdd.json) | 242.20 | 6.08 | 168 | 460 / 436 | 2.41e-16 | Unselected; endpoint A/B not run |
| [fffs](fffs.json) | 157.59 | 5.17 | 64 | 3484 / 680 | 4.86e-16 | Unselected; endpoint A/B not run |
| [fffp](fffp.json) | 241.03 | 6.08 | 168 | 452 / 420 | 1.46e-16 | Unselected; endpoint A/B not run |
| [fffd](fffd.json) | 69.96 | 3.51 | 168 | 956 / 956 | 4.23e-16 | Unselected; endpoint A/B not run |
| [ffff](ffff.json) | 329.69 | 12.20 | 168 | 2776 / 2588 | 1.12e-15 | Unselected; endpoint A/B not run |

Validation: **617 Python tests passed, 75 skipped; 9/9 native CPU suites passed; pre-commit passed.** All GPU numerical tests, endpoint runs, profiling, and sanitizer execution used finite Slurm allocations on `main` with `gpu:5090:1`, preserving assigned device visibility. Normal CUDA PR CI compiles only the five representative smoke classes. See the [runner protocol](../../../docs/f_shell_validation.md) for reproduction commands.

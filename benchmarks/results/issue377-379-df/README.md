# Occupied DF execution and compact source qualification (#377–379)

This bundle records current-master comparisons on NVIDIA GeForce RTX 5090
(sm_120), production Release CUDA 12.9.1, spherical def2-SVP RHF water clusters,
Naux=NAO and ranks 20/40/80/160. All real-device work uses finite Slurm jobs on
`main` with `--gres=gpu:5090:1`. Energy/force gates remain 1e-9 Ha and 1e-8
Ha/Bohr; the metric relative threshold remains 1e-10. No approximation changes.

Independent energy/force gates pass at every size. At 768 AOs, the five-pair
current-master exchange comparison improves the warm endpoint from about
12.05 to 10.81 s. Occupied response then reduces it to 9.81 s. The fresh
automatic-policy GPU4PySCF matrix measures 9.848 s; all five native samples
take three iterations, with maximum energy/force errors of 4.05e-11 Ha and
1.31e-10 Ha/Bohr. Only one GPU4PySCF sample shares the three-iteration branch
(2.537 s); its other four samples take one iteration. The 96/192/384 matrix
has no matching branches, so it establishes numerical parity and scoped raw
times, without a qualified iteration-matched speed claim.

| Full raw generation, including readback | Master median (s) | Compact median (s) | Speedup |
| --- | ---: | ---: | ---: |
| 192 AO | 0.72749 | 0.43598 | 1.67x |
| 384 AO | 7.98692 | 3.16615 | 2.52x |
| 768 AO | 103.90351 | 24.24465 | 4.29x |

All three complete raw arrays are bit-identical to master. Source-backed
768-AO plan setup measures 104.325 / 24.588 s. Fixed-density dense/occupied
K medians are 0.82841 / 0.20960 s, with maximum K difference 1.78e-14.

| 768 response diagnostic | Dense | Occupied |
| --- | ---: | ---: |
| AO/projection/expansion products | 1536 full-AO | 1536 projection + 1536 expansion |
| Product device time, including metric/weight GEMMs | 1.933 s | 0.449 s |
| Generated three-center derivative device time | 4.812 s | 5.252 s |
| Auxiliary derivative panels | 1 | 12 |
| Peak response-weight elements | 452,984,832 | 37,748,736 |
| Retained projected elements | — | 19,660,800 |
| Additional response device scratch | 132,236,872 B | 37,865,032 B |
| Borrowed J/K capacity | 10,871,635,968 B | 10,871,635,968 B |
| Response H2D traffic | 3,628,698,912 B | 3,628,698,912 B |
| Additional factor-validation D2H | 0 | 4,718,596 B |

The occupied projection and expansion each perform 175,154,135,040 FLOPs.
The factor has rank 160, occupies 983,048 bytes with its generation controls,
and produces density generation four after one dense seed and two occupied
iterations. The observed final commutator is 1.38e-12, density RMS 3.07e-18,
and idempotency error 2.31e-14. The 64-slice schedule slightly increases shell
primitive work (348,913,664 to 350,896,128); its matrix-product savings still
improve the complete clean endpoint. Profiles at 192/384 retain their own
actual branch and counters; their intrusive wall times are not clean samples.

The unchanged resident cold endpoint is 104.46 s in the independent matrix.
Its largest measured cold device component is one-electron/nuclear derivative
generation, 57.84 s. The cold final determinant needs strict correction and
correctly falls back to dense response. `cold-components.json` keeps this
whole-endpoint ledger separate from source-backed raw generation.

Validation includes 91 CUDA regressions, 30 CPU native tests, 48 CPU Python
tests (one CUDA-only skip), six final occupied-response replay cases, 36
independent source configurations, native adversarial lifecycle checks, and
Compute Sanitizer with zero errors. The UHF tests cover two occupied spin
channels and zero beta rank. `independent-response-parity.json` also checks
every clean 192/384/768 response sample against the fresh GPU4PySCF results.
The final build also passes seven focused CUDA checks and the native lifecycle
test after preserving explicit occupied precedence over automatic backend
filtering. Its independent 768-AO confirmation records 9.818 s occupied and
9.829 s automatic execution, both with three iterations and unchanged gates
(`final-policy-confirmation.json`). This guard changes no executed route in
the main RTX 5090 qualification.

## Interpretation

- `master-768-exchange.json` contains five interleaved dense/occupied K endpoint
  pairs against current master f23e041, with one frozen post-cold warm density.
- `qualified-768-response.json` contains the corresponding five response-space
  pairs with K held occupied. `rejected-small-panel-768.json` preserves the
  earlier slower schedule. `rejected-resident-export-768.json` preserves the
  later cold-setup regression; its resident-export integration is excluded
  from the final code. Its warm response measurements remain valid evidence.
- `96/192/384-exchange.json` and `192/384-response.json` retain every clean
  sample, priming cost, iteration branch, energy and force. The 384 response
  comparison changes both algebra and borrowed resident storage: it is a
  combined-route crossover, not an isolated low-rank GEMM speedup. Automatic
  selection remains limited to the qualified 768/768 RHF rank-160 domain.
- `*-raw-source.json` records complete generation, including readback, at
  192/384/768 AOs. Counter arithmetic follows the exact launched ranges and
  mapping. Coefficient reads are logical lane operations, not measured DRAM
  transactions. Both old and compact traversal counts are reported for the
  same nonzero expansion and primitive work. No intrusive atomics affect timing.
- `*-fixed-k.json` retains fixed-density calls including upload and readback;
  `*-fixed-k-components.json` separates actual stream timings from capture.
- `profiles/` contains intrusive component/work/transfer/scratch and final-state
  diagnostics, never clean performance samples. Capture counts describe graph
  construction; executed iteration provenance and fixed-K traces are separate.
- `gpu4pyscf/` contains fresh matched complete energy+force comparisons using
  each engine's frozen post-cold seed. Basis metadata and scientific settings
  are retained with the unchanged independent gates.

The source probe is not the unbudgeted resident Cartesian exporter. Its
approximately 103.9-to-24.3-second improvement must not be attributed to that
separate preparation path. The rejected integration demonstrates this boundary.
Whole-plan capacity, borrowed scratch and new response allocations are reported
separately; capacity is not a measured global GPU peak.

## Reproduction and identity

Build with `cmake --preset cuda-release-sm120` using CUDA 12.9.1 and production
compiler settings, then build `vibeqc`, `vibeqc_df_source_probe`,
`vibeqc_df_occupied_probe`, `vibeqc_df_value_probe`, and
`vibeqc_df_occupied_response_tests`. The retained native libraries used
`-lineinfo`. Use a fresh artifact directory for every run.

Run `python -m benchmarks.df_policy_endpoint --aos 768 --repeats 5 --output ...`
with `PYTHONPATH=python:.` and `VIBEQC_LIBRARY` pinned to the desired library.
Change `--control VIBEQC_DF_RESPONSE_SPACE` with `VIBEQC_DF_EXCHANGE=occupied`
for response comparisons; `--trace --journal --repeats 1` is a separate
intrusive pass. If a library directory contains `source.patch`, the runner
retains that frozen source rather than a later dirty checkout. An explicit
`--source-patch` is also supported. Every command must run within finite `srun`.

`vibeqc_df_source_probe inputs/768.txt 3 report.jsonl raw.bin` generates the
full raw source. The inputs contain unnormalized physical shell primitives;
the native molecule owner applies its usual normalization. For fixed K,
append the row-major `coefficients` array from the **Git-tracked**
[`inputs/768-occupied-coefficients.npz`](inputs/768-occupied-coefficients.npz)
to `inputs/768.txt` (for example with `numpy.savetxt`, using `%.17g`)
before invoking `vibeqc_df_occupied_probe INPUT 5 0 0 arrays.bin`. The NPZ
is available in a clean checkout and has SHA-256
`9eca543c2e88db47b30f11dd29ca6c20fcd691a836a2593a82da04ade69da8a2`.
The retained input metadata identifies the independent converged checkpoint
and exact orthogonality check. Large raw tensors and **output** density/K
arrays, logs, binaries and detailed traces
remain outside Git; `local-artifacts.json` pins their paths, sizes and SHA-256.

`source-versions.json` pins each cited library, patch and reconstruction base.
The main implementation patches use master f23e041; the precedence guard
is an incremental patch against 4be1c3a. The review-v11 patch against 4f3f2dc
adds a bounds check for malformed automatic-response tokens; valid-token
execution and the cited performance samples are unchanged. Its focused CUDA,
sanitizer, reference-contract and independent endpoint checks are recorded in
`validation.json`. The runner's `git_head` records the checkout at invocation;
a pinned library can have an older reconstruction base.
Compact endpoint files omit only repeated basis metadata, which is retained
once, and record the full original JSON hash. Historical provisional probe
results with an incorrect lane label are excluded from final causal counts.

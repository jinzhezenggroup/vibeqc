# Device response weights for retained DF values (#206)

The default retained-value plan now reuses its forward device metric factors and the existing CUDA force-response contraction. Raw auxiliary slices are uploaded into bounded scratch. The full one-electron/Pulay, three-center, auxiliary, metric/subspace and nuclear response is retained. `VIBEQC_DF_HOST_RESPONSE_WEIGHTS=1` selects the former host-weight adapter under the same conservative resource reservation.

Five interleaved warm repeats per engine cover RHF at 96/192/384 AOs and UHF at 19 AOs, each at batch one/four, for energy and complete forces. The external basis data, actual factor ranks, software stack and all raw results are retained. Ordinary timing ratios remain distinct from a demonstrated common SCF iteration branch.

| AOs / batch | VibeQC energy | GPU4 energy | VibeQC forces | GPU4 forces |
| --- | ---: | ---: | ---: | ---: |
| 19 / 1 | 0.002920 s | 0.063839 s | 0.007707 s | 0.242597 s |
| 19 / 4 | 0.005049 s | 0.256100 s | 0.024590 s | 0.973833 s |
| 96 / 1 | 0.012038 s | 0.068955 s | 0.165541 s | 0.268036 s |
| 96 / 4 | 0.034931 s | 0.275535 s | 0.633124 s | 1.055516 s |
| 192 / 1 | 0.065769 s | 0.080993 s | 1.472930 s | 0.348317 s |
| 192 / 4 | 0.237512 s | 0.320358 s | 5.799279 s | 1.397593 s |
| 384 / 1 | 0.476370 s | 0.137929 s | 45.951228 s | 0.613716 s |
| 384 / 4 | 2.374277 s | 0.548398 s | 183.995242 s | 2.464323 s |

The 192-AO performance gate remains unmet. All repeated energy errors are below 1e-9 Ha and all force-component errors below 1e-8 Ha/Bohr. The maximum observed paired errors are 4.934009e-11 Ha and 9.628634e-11 Ha/Bohr. RMS errors, net forces, per-item convergence and metric diagnostics are also retained.

| Same-library force ablation | Host weights | Device weights | Host / device |
| --- | ---: | ---: | ---: |
| 192 AOs / batch 1 | 3.383580 s | 1.472930 s | 2.297 |
| 192 AOs / batch 4 | 13.560719 s | 5.799279 s | 2.338 |
| 96 AOs / batch 1 | 0.270609 s | 0.165541 s | 1.635 |
| 96 AOs / batch 4 | 1.062020 s | 0.633124 s | 1.677 |

Each host/device ablation has identical VibeQC iteration rows. The external engine retains its own fixed post-cold density policy, and its branch differences remain explicit. No native compilation, profiling, memory sampler or reference preflight overlaps the clean endpoints.

The separate 96/192-AO profiles expose raw upload bytes, device response scratch, scalar-density transfers and final-output transfers. A 384-AO cold/warm profile additionally retains sampled GPU process memory and cumulative host high-water measurements. These are observations for the recorded calls, not whole-process budget qualification. Native metric byte diagnostics describe plan estimates; global qualification remains ≤16 orbital /128 auxiliary AOs.

The instrumented 384-AO warm call uses 8 response panels and uploads 4,015,521,792 raw bytes with 132,174,096 device scratch bytes. Its exchange-weight/metric region takes 39.669 s; raw uploads occupy 8.213 s of host time. Host and device intervals overlap and must not be added. The maximum sampled process GPU usage is 8260 MiB; the maximum observed host high-water value is 2,576,867,328 bytes.

Validation covers 71 host checks, 105 passing GPU checks in the first run plus two corrected exact-byte expectations verified in a 34-check complementary run, native DF with default/generated one-electron response, capture recovery, and occupied UHF batch-four memcheck with zero errors. Actual W/full-force exports, retained/discarded metric subspaces, invalid-value transactions, warm reuse and route/transfer counters are tested. The initial resource-expectation failures and frozen-library loader failure are retained.

The unchanged direct-SCF matrix was also rerun. Its outcome and the frozen pre-change 96-AO control are reported below; tolerances were not changed.

| Direct gate | Candidate | Pre-change control |
| --- | --- | --- |
| 96 AOs / batch 1 | fail | fail |
| 96 AOs / batch 4 | fail | fail |
| 192 AOs / batch 1 | pass | not repeated |
| 192 AOs / batch 4 | pass | not repeated |

The candidate and pre-change native force arrays differ by at most 5.676258e-13 Ha/Bohr, with identical native iteration rows. The direct failures remain acceptance failures even if reproduced on the pre-change control. This slice does not close #206, #308, #309, #310 or #311. The complete changed-geometry/budget/ablation integration matrix and final acceptance audit remain open.

The initial 30-minute allocation was stopped only after all batch-four energy samples had been written, because the remaining 384-AO force case could not fit its remaining window and Slurm rejected an extension. That completed result and the cutover journal are preserved. The sole remaining force case ran in a fresh finite allocation; no partial force timings were reused.

Qualified source: `899a6ab8ffe3c84f1be7d89a6d61354afd14cbf2`. Native source identity: `b4eb37622c473091a8f0435e3bd8900effcda4a2da9520de8002f151b9ff04ce`. Frozen library SHA-256: `a428e9c276a28fd9276e9148ba6d7dd04ace848f20a4786516be9407e90e1152`.

`evidence.zip` contains exact runners, raw samples, input qualification, traces, state exports, failed attempts, build/validation logs and reconstruction. Every member was restored and compared byte-for-byte. Archive SHA-256: `2440dafc718bc58f2981dddb4dd2262dfc2ffa59b3b7389e3d36ee4898c388e3`.

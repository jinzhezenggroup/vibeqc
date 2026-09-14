# FP64 cuBLAS metric response dots (#206)

The exchange metric response now contracts the existing bounded raw panel with `cublasDgemv` on the plan-owned handle and stream. Strided accumulation preserves Coulomb and both spin contributions. The spectral/auxiliary response and FP64 model are unchanged. `VIBEQC_DF_SERIAL_RESPONSE_DOT=1` selects the former serial AO-pair dot with the same explicit scratch bound; provider failures propagate through cleanup.

Five interleaved warm repeats per engine cover RHF at 96/192/384 AOs and UHF at 19 AOs, batch one/four, energy and complete forces. Exact orbital/auxiliary bases and actual external factor ranks were independently checked for every geometry before timing. External SCF branch differences remain explicit; these ordinary ratios do not establish iteration-matched parity.

| AOs / batch | VibeQC energy | GPU4 energy | VibeQC forces | GPU4 forces |
| --- | ---: | ---: | ---: | ---: |
| 19 / 1 | 0.002906 s | 0.063777 s | 0.007202 s | 0.243178 s |
| 19 / 4 | 0.005049 s | 0.256993 s | 0.022437 s | 0.970578 s |
| 96 / 1 | 0.012056 s | 0.068579 s | 0.123377 s | 0.264523 s |
| 96 / 4 | 0.034908 s | 0.273966 s | 0.473905 s | 1.061039 s |
| 192 / 1 | 0.064891 s | 0.080687 s | 0.830929 s | 0.350182 s |
| 192 / 4 | 0.233961 s | 0.320424 s | 3.253369 s | 1.404536 s |
| 384 / 1 | 0.472125 s | 0.137290 s | 11.965691 s | 0.615462 s |
| 384 / 4 | 2.361985 s | 0.548515 s | 47.460322 s | 2.466163 s |

All 100 paired comparisons pass 1e-9 Ha /1e-8 Ha/Bohr. Maximum paired errors are 4.934009e-11 Ha and 1.086335e-10 Ha/Bohr. RMS errors, net forces, raw per-item convergence, metric diagnostics and all samples are retained. The 192-AO performance target remains unmet.

| Same-library force ablation | Serial dot | GEMV dot | Serial / GEMV |
| --- | ---: | ---: | ---: |
| 192 AOs / batch 1 | 1.469688 s | 0.830929 s | 1.769 |
| 192 AOs / batch 4 | 5.838267 s | 3.253369 s | 1.795 |
| 96 AOs / batch 1 | 0.165675 s | 0.123377 s | 1.343 |
| 96 AOs / batch 4 | 0.638934 s | 0.473905 s | 1.348 |

Each serial/GEMV ablation has identical native iteration rows. No native compilation, profiling, reference preflight or memory sampling overlaps clean endpoint measurements. Separate 96/192-AO cold/warm profiles retain actual dot counters and unchanged scratch/transfer accounting.

The instrumented 384-AO warm call uses 8 panels, uploads 4,015,521,792 raw bytes, and reports 132,174,096 bytes of response device scratch. The metric-dot device interval is 0.170 s, three-center derivative contraction 3.138 s, and raw-upload host time 8.376 s. These host/device intervals overlap and must not be added. Upload host intervals may include waits for preceding stream work; a separate charge-dot ablation is needed before inferring CPU packing cost.

The maximum sampled GPU process usage is 8258 MiB; the maximum observed cumulative host high-water value is 2,576,830,464 bytes. Every large-profile force component passes against the retained external reference. Samples and native plan estimates do not extend whole-process qualification: the global qualified range remains ≤16 orbital /128 auxiliary AOs.

Validation: 71 host checks, 115 GPU checks, native DF with default/generated/serial-dot variants, capture recovery, and occupied UHF batch-four memcheck with zero errors. The frozen library embeds the independently checked source hash. The first compile hit its finite 20-minute limit; its log remains in the historical full-run bundle, and the exact incremental build completed in another finite allocation before any GPU qualification.

The existing direct-SCF acceptance matrix was rerun without changing its tolerances. Historical pre-change control arrays from #343 are retained for the 96-AO comparison; no new control timing is claimed.

| Direct gate | Candidate | Historical pre-change control |
| --- | --- | --- |
| 96 AOs / batch 1 | fail | fail |
| 96 AOs / batch 4 | fail | fail |
| 192 AOs / batch 1 | pass | not repeated |
| 192 AOs / batch 4 | pass | not repeated |

The candidate and historical pre-change native forces differ by at most 6.436830e-13 Ha/Bohr, with identical native iteration rows. Strict direct acceptance failures remain open. This slice does not close #206, #308, #309, #310 or #311; complete changed-geometry/budget/ablation integration and final acceptance remain outstanding.

Qualified source: `3430c0bb1071685f65072d9941a33e4f3e2d1e95`. Native source identity: `cb60b50bac01691a81ac6bd46919c840bdf1f609d5651e637adef31e7f01e9ca`. Frozen library SHA-256: `5b841ed1dfbfd34eaa73b20a1c4f28da4cf1d798bd8e502093679c9a1f7aaa06`. The exact measured patch and rebase identity record are retained under `reproduction/`.

`measurements/` retains every value in the 20 clean comparison records, including
all raw timing samples, arrays, convergence and errors, plus independent input
qualification. `direct-gate/` and `historical-prechange-direct/` retain the strict
gate failures and controls. `reproduction/` retains the measured source patch,
runner commands and identities. All JSON values and ordering match the original
records; `summary.json` pins both original and selected file hashes.

The former 7.45 MB `evidence.zip` also contained routine test outputs and detailed
profiler traces. It has been removed from this PR's current tree. An exact local
copy remains under ignored `.artifacts/`; its historical Git identity and SHA-256
are recorded in `summary.json`. Full-run inspection can recover it without
putting it back into the reviewed result directory:

```bash
mkdir -p .artifacts/metric-gemv-history
git show 5a721350fc7ba83133e062980dff0757c2c1f3ad:benchmarks/results/issue206-metric-gemv/evidence.zip > .artifacts/metric-gemv-history/evidence.zip
```

This storage correction does not rerun benchmarks or change any reported gate.

Retained Python runners have repository lint formatting/import cleanup; the
profile sampler makes its original `check=False` default explicit. Original
measured runner hashes remain separate from the retained file hashes. These
files describe historical local paths; use fresh output paths when reproducing.

## CUDA ownership disclosure

```text
generated capability: existing generated DF values/coordinate response reused unchanged
handwritten scientific CUDA LOC: +33 / -5
runtime CUDA LOC: +0 / -0
legacy production path removed: no
retained duplicate reason: oracle
```

[ownership-delta.json](ownership-delta.json) compares the physical source with
base `12e46d4e6994e8015bf4c84d83d0786db5bf2d91` using the ownership reporter's
nonblank/noncomment lexer. The net +28 lines are handle/status/layout and GEMV
composition within method-specific functions; they remain conservatively
scientific. Ten unchanged serial-kernel lines move from `scientific` to
`oracle`, so the aggregate maintained scientific count still grows by 28.
This classification does not claim physical deletion or generated migration.

The semantic `df_response_weights` ledger entry owns the default cuBLAS path
and the opt-in `VIBEQC_DF_SERIAL_RESPONSE_DOT=1` control. Its different serial
reduction order provides numerical/regression and timing controls alongside
independent NumPy and external references; it is never an automatic error
fallback. The ledger requires removing the kernel and selector after complete
#206 numerical, resource and endpoint acceptance, once independent gates and
retained samples cover its regression role. Those closure gates remain open.

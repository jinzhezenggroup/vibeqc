# 384-AO DF response attribution (#308)

The original progress-fenced upload interval reproduces at **8.303 s**. Within
that host scope, 2.054 s overlaps preceding same-stream kernels, 1.526 s overlaps
actual H2D copies, and 4.724 s is outside observed device activity. These are
clock intersections, not exclusive causal CPU/wait measurements. In particular,
the residual is not labeled as pure packing. The original charge reduction
accounts for the preceding kernel activity in this fenced capture.

Separate steady warm captures disable progress fences, freeze the post-cold
density, and perform an untimed replay before measurement. Every selection
executes three SCF iterations, eight response panels, 56,623,104 three-center
weights and 147,456 metric weights. All upload **4,015,521,792 raw bytes in 3,404
copies**. Raw values already reside on the host; warm response performs no
integral regeneration in this resident-provider experiment.

| Diagnostic | Complete force wall | Raw-copy API host | Actual raw H2D | Explicit pre-copy drain | Explicit CPU gather |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 11.461 s | 10.467 s | 1.507 s | — | — |
| Drain before original strided copy | 13.227 s | 5.620 s | 1.513 s | 6.572 s | — |
| Drain, pinned gather, contiguous copy | 9.985 s | 0.028 s | 0.211 s | 6.035 s | 2.821 s |
| 128 gradient destinations | 11.917 s | see retained timeline | see retained timeline | — | — |

All durations in this table come from intrusive profiler runs; they are not
clean endpoint promotion evidence and the columns must not be summed. The
packed loop's wall time is 2.820 s and its thread CPU time is 2.821 s. The
drained strided route has **zero kernel overlap inside its raw-copy API calls**,
so the remaining host interval cannot be assigned to earlier derivative work.
Its device DMA cost and runtime staging/control remain distinct from the
explicit gather measured by the packed control.

Actual default device execution includes 3.141 s of three-center derivatives,
about 2.06 s of charge reduction, 0.50 s of left/right density products, and
4.327 ms of metric derivatives. The detailed GEMV, exchange-weight, spectral
metric, SCF and transfer activities are retained in the timeline summaries.
Kernel durations exclude idle gaps between submissions, which CUDA event
intervals can contain.

The distributed-sink control keeps the same derivative arithmetic, subgroup
reduction and atomic operation count. Its three-center interval is 3.139 s,
versus 3.141 s for default, and its final cuBLAS gradient reduction takes
2.944 microseconds. This single capture shows no large saving from changing
atomic destinations; it does not isolate total atomic instruction cost or
establish a speedup. The probe adds 148,480 device bytes without changing the
132,174,096-byte base response workspace, auxiliary tile, or raw traffic.
Nsight Compute counters were unavailable (`ERR_NVGPUCTRPERM`).

Every complete energy and force array passes against the retained independent
GPU4PySCF reference at unchanged 1e-9 Ha / 1e-8 Ha/Bohr gates; maximum force
error is below 1.1e-10 Ha/Bohr. All four current warm host traces contain zero
CPU-reference eigensolve calls. RHF/UHF and Cartesian/spherical probe transitions
pass numerical checks and memcheck with zero errors. Independent metric,
auxiliary-only center, finite-difference, rank-crossing and geometry replay
checks are retained, together with the 96-AO 32/64-MiB property-budget checks.

Sixteen old 1–4-MiB derivative-test budgets failed on the unchanged master
control as well as the instrumented library: current metric, compact SCF and
ordinary AO providers reserve fixed workspace floors. The same numerical cases
now use feasible 8/16-MiB budgets (including the batch size); former tiny
requests remain explicit OOM tests. No force/energy tolerance or reference case
was removed. Validation covers 59 response/derivative cases, four property-budget
cases, 34 CPU accounting cases, and four memcheck cases. The summary records
which runs establish these counts, including failed attempts.

`measurements/` retains complete arrays, iteration branches, numerical errors,
component ledgers and measured source/runner/library identities. `timelines/`
retains actual Nsight kernel/API/copy summaries and clock intersections.
`summary.json` pins selected files and the locally retained full profiler
artifacts. Logs, binaries and databases remain in ignored `.artifacts/`.

Reproduce with the commands in [the timeline guide](../../../docs/developer/df_response_timeline.md)
and `benchmarks/issue308_response_timeline.py`. Select `drain` or `packed` with
`VIBEQC_DF_RESPONSE_UPLOAD_PROBE`; select the sink control with
`VIBEQC_DF_RESPONSE_SCATTER_PROBE=sharded`. The original progress control has a
separate retained runner. Its measured runner hash remains in the sample;
the retained copy has formatting/import cleanup, with its own file hash.

This evidence supports proceeding with generated shell-block response work.
It does not promote an optimization or
close #308/#206. The next implementation should share primitive geometry,
Boys values and moment intermediates across a shell block, while separately
addressing the now-measured charge reduction and raw staging costs.

## CUDA ownership disclosure

```text
generated capability: existing generated scalar DF derivatives reused by all probes
handwritten scientific CUDA LOC: +77 / -4
runtime CUDA LOC: +18 / -6
legacy production path removed: no
retained duplicate reason: none
```

The physical delta is in `ownership-delta.json`. No lines were reclassified.
The bridge's timing, allocation and output-layout adapters remain conservatively
counted as scientific method glue. Both destination specializations invoke the
same generated derivative policy; they are not separate scientific derivative
implementations. These diagnostic controls are not automatic runtime fallbacks.

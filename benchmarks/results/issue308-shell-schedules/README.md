# Generated s/p/d/f response scheduling (#308)

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

Clean complete-force endpoints select generated compact shell scheduling with
BLAS response weights and bounded pinned panels. Automatic selection is scoped
to resident 192--384-AO sm_120 device-metric execution. Other sizes, backends,
source-backed values and diagnostic/serial mappings retain their former
defaults. All four explicit controls remain available as independent overrides.
The full original oracle is `generic` / `warp` / `scalar` / `pageable`.

The compiler supplies all 64 s/p/d/f shell classes and three bounded subgroup
schedules. Shared Gaussian geometry, Boys values and moment polynomials feed
immediate weighted contraction. Metric, auxiliary and retained/discarded
subspace response remain complete; no coordinate derivative tensor is formed.

| v4 clean case | Generic mean (range), s | Compact combination mean (range), s | Warp combination mean, s |
| --- | ---: | ---: | ---: |
| 192 AO, batch 1 | 0.803615 (0.801797--0.806909) | 0.268689 (0.266731--0.270345) | 0.327366 |
| 192 AO, batch 4 | 3.221813 (3.211565--3.228307) | 1.043454 (1.042503--1.044251) | 1.282262 |
| 384 AO, batch 1 | 12.006982 (12.002828--12.009850) | 3.261368 (3.257585--3.263579) | 3.590240 |
| 384 AO, batch 4 | 47.905333 (47.739115--47.998993) | 13.497533 (13.490439--13.507008) | 14.851119 |
| 19 AO UHF, batch 1 | 0.007007 (0.006992--0.007021) | 0.029385 (0.029256--0.029638) | 0.026917 |
| 19 AO UHF, batch 4 | 0.021603 (0.021578--0.021637) | 0.110642 (0.110581--0.110696) | 0.100685 |

Each row retains three uninstrumented repeats on one fixed post-cold density,
with untimed priming per candidate. Compact improves the complete 384-AO
endpoint by 3.682x/3.549x for batch one/four. The 192-to-384 ratios fall from
14.94/14.87 to 12.14/12.94. This is a material endpoint improvement; residual
scaling remains steep. Small UHF rejects both combined candidates, so its
automatic route remains generic.

All candidates preserve each case's SCF branch: `[3]` at 192/384 batch one,
`[3,2,2,3]` at 192 batch four, `[3,3,3,3]` at 384 batch four, and two iterations
per UHF item. All independent energy/full-force comparisons pass the unchanged
1e-9 Ha / 1e-8 Ha/Bohr gates. Across v4 clean samples, maximum errors are below
4.889e-11 Ha and 1.087e-10 Ha/Bohr. Full arrays, sample ranges, per-item metric
resources and exact source/library/runner identities remain in the JSON files.

The v3 clean sweep is also retained. Its compact means are 4.022703/16.451007 s
at 384 AO, versus v4's 3.261368/13.497533 s. V4 changes host gathering to bounded
128-row groups with contiguous column stores, adding no copy buffer. Both
versions were measured in separate finite Slurm allocations with their frozen
sources and libraries; neither was measured during compilation or profiling.

The earlier v3 nine-candidate intrusive sweep remains attribution evidence.
It separately tests shell schedules, response BLAS and pinned staging; it is
not used in place of the clean endpoint selection above. The actual v4 compact
Nsight capture reports:

| Activity | Measured time |
| --- | ---: |
| Generated three-center derivative kernels (CUPTI) | 0.977928 s |
| Exchange response matrix-product kernels (CUPTI) | 1.079606 s |
| Exchange metric-dot kernels (CUPTI) | 0.156156 s |
| Metric-center derivative kernels (CUPTI) | 0.004202 s |
| Raw-response H2D DMA (CUPTI) | 0.158380 s |
| Raw-copy API calls, host | 0.021902 s |
| Explicit host gathers, inclusive NVTX | 1.233679 s |
| Host waits for panel reuse, inclusive NVTX | 1.173585 s |

Host and device intervals overlap and must not be added. CUDA event intervals
also include stream gaps and instrumentation effects; only the CUPTI activity
rows above are kernel/DMA execution. The old inclusive raw-upload interval was
never pure transfer or CPU packing time.

The full shell consumer visits 7,225,344 nonzero shell triples, consumes
56,623,104 nonzero public weights, evaluates 44,605,440 primitive/Boys products
and serves 243,749,376 Cartesian component contributions at 384 AO. It uploads
4,015,521,792 raw-response bytes over eight weight panels. Two 16-column pinned
panels occupy 37,750,784 host bytes; last-read events protect their reuse and
the arena drains before freeing them, including exceptional exits.

V4 qualification in Slurm job 9605 passes 67 GPU tests, four additional f-basis
cases for each of warp and packed scheduling, 32 memcheck cases with zero errors,
and seven focused racecheck cases with zero errors or warnings. Independent
emitted-arithmetic qualification covers all 64 classes at asymmetric/coincident
centers (128 cases against Libcint). All 192 compiled kernel variants fit their
generated bounds: maximum shared memory 48,624 bytes; register counts 94--255.

The promoted v5 library changes selection only. Its 67 GPU tests and separate
19/192/384 default-consumer audits pass, with independent complete-force parity
and zero CPU-reference eigensolves. The matched #206 rerun is complete: 16 cells,
five interleaved repeats per engine, with energy and complete-force properties.
All 80 paired comparisons pass 1e-9 Ha / 1e-8 Ha/Bohr; maximum paired errors are
4.935e-11 Ha and 8.257e-11 Ha/Bohr. Independent orbital/auxiliary bases and actual
external metric ranks are qualified for all 20 input geometries before timing.

| AOs / batch | VibeQC energy | GPU4 energy | VibeQC complete force | GPU4 complete force |
| --- | ---: | ---: | ---: | ---: |
| 19 / 1 | 0.002906 s | 0.064356 s | 0.007235 s | 0.243225 s |
| 19 / 4 | 0.005034 s | 0.255533 s | 0.022251 s | 0.970007 s |
| 96 / 1 | 0.012090 s | 0.068979 s | 0.123545 s | 0.263308 s |
| 96 / 4 | 0.034973 s | 0.275450 s | 0.478453 s | 1.056581 s |
| 192 / 1 | 0.065235 s | 0.080027 s | 0.271016 s | 0.351015 s |
| 192 / 4 | 0.235746 s | 0.320794 s | 1.056175 s | 1.400153 s |
| 384 / 1 | 0.476119 s | 0.137592 s | 3.277437 s | 0.614871 s |
| 384 / 4 | 2.373977 s | 0.549652 s | 13.503923 s | 2.467058 s |

These are ordinary medians. VibeQC uses two or three warm iterations per item;
GPU4PySCF usually uses one, with occasional higher branches at 384 batch four.
Every engine's raw branches and final residuals are retained; no cell provides
an iteration-matched external speedup. At 192 AO the ordinary force endpoint
now favors VibeQC, while the 384-AO external gap remains about 5.3--5.5x.

The unchanged direct-versus-direct audit passes both 192-AO cells. The 96-AO
batch-one/four cells reproduce the previously retained force-threshold failures:
5.234e-11/5.492e-11 Ha/Bohr against 3e-11. The prior audit reports
5.226e-11/5.492e-11; full current and prior gate records remain linked from
`summary.json`. No direct consumer or tolerance was changed by this DF slice.
Those direct acceptance failures remain open, along with #206's residual
384-AO gap and larger/constrained/changed-geometry closure evidence. #308 and its
child-specific repeated closure requirements are not declared complete.

`summary.json` pins retained measurements and qualification logs. Reproduction
files retain the exact v3/v4/v5 measured source patches and parent Git heads.
The v5 patch applies to `92140cb`; v4 scientific code is the pre-promotion source.
The measured native identity matches promotion commit `5ab5449`. Individual
benchmark Git metadata reflects the enclosing checkout at measurement time;
the frozen source patch, native identity and library hash identify the measured
scientific code throughout the run.
The v5 helper `.py` files are formatter-normalized; `v5-measured-*.py.txt`
retain the exact executed bytes and match the provenance hashes. Complete
profiler databases, build products and routine logs remain under the
ignored `.artifacts/issue308-shell-block/` directory. Reproduction scripts show
historical local paths: use fresh output directories and finite Slurm `main`
allocations with `--gres=gpu:5090:1`, preserving assigned device visibility.

## CUDA ownership disclosure

```text
generated capability: shared s/p/d/f shell derivatives with three bounded schedules
handwritten scientific CUDA LOC: +152 / -18
runtime CUDA LOC: +59 / -33
legacy production path removed: no
retained duplicate reason: oracle
```

`ownership-delta.json` compares this slice with PR B at `672ec47` using physical
nonblank/noncomment line accounting. Net scientific growth is conservatively
counted method/provider/layout, bounded host staging and scoped selection glue.
Gaussian derivative mathematics remains generated. No source lines are
reclassified to hide growth. The explicit generic route remains an oracle, and
the serial metric-dot oracle retains its existing #206 retirement gate.

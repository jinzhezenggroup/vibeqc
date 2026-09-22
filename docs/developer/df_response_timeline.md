# DF response timeline and causal upload probes

The 384-AO response sentinel uses the same orbital/auxiliary def2-SVP basis and
full analytic-force reference as the matched #206 matrix. Its resident raw
values are prepared before the warm force call. A warm raw-value read therefore
uploads an existing column; it does not generate new integrals. Source-backed
execution is a distinct provider and must be reported separately.

`benchmarks/issue308_response_timeline.py` checks the loaded library's embedded
scientific source identity, exact reference coordinates and settings, every
energy and force component, and actual SCF iteration branches. It freezes the
post-cold density and performs an untimed replay before measurement. Clean and
instrumented runs use separate output directories. Raw profiler databases and
logs belong in ignored `.artifacts/`; reviewed results retain compact summaries,
raw endpoint samples, source/library identities and reproduction commands.

For #206 closure rows that declare a resident response, pass
`--expected-response-policy resident`. The runner then fails closed if the
executed trace uses a streamed/host-panel fallback, performs a host raw-panel
gather, uploads more than one full `[AO,AO,Naux]` raw tensor, or places its
auxiliary consumer count outside the declared tile bounds. The lower bound is
`ceil(Naux/tile)`; shell-aligned consumers may legitimately exceed it, up to
`Naux`. It records response storage separately from the value provider, along
with whitening and occupied/dense policy, in each response row. Optional
`--max-h2d-bytes`, `--max-d2h-bytes`, and
`--max-transformed-tile-productions` arguments make per-case resource ceilings
explicit. The response-work gate accepts either exact dense AO-pair work
(`Naux*N^2`) or exact symmetric packed-pair work
(`Naux*N*(N+1)/2`), as selected by the trace's `response_packed_pairs`
counter; the chosen representation and observed weight count are retained in
each response row. The hardware-free validator is also available directly:

```bash
python benchmarks/issue206_resident_sentinel.py \
  .artifacts/response-default-run/warm-0.cuda.jsonl \
  --expected-policy resident \
  --output .artifacts/response-default-run/resident-sentinel.json
```

Use `auto` for exploratory traces; it still publishes the selected route but
does not turn a fallback control into a resident acceptance result. This
structural sentinel complements, and does not replace, clean endpoint timing,
numerical parity, and the complete resource ledger.

## Capture

With an up-to-date Release library, run a single warm replay under Nsight:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /group/software/cuda-12.9.1/bin/nsys profile \
  --trace=cuda,nvtx --cuda-graph-trace=node --cuda-event-trace=false \
  --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output=.artifacts/response-default \
  /path/to/python benchmarks/issue308_response_timeline.py \
  --library build/cuda-release-sm120/libvibeqc.so \
  --output .artifacts/response-default-run --component-trace --nsys
```

The component trace supplies NVTX origins and exact mathematical work/byte
counts. Its CUDA event intervals can include stream idle gaps; Nsight's actual
kernel and copy activities own device execution durations. The companion host
trace requires zero actual CPU-reference eigensolve calls in the warm replay.
`VIBEQC_DF_PROGRESS_TRACE` is prohibited by this runner: that diagnostic fences
each region and changes the pageable-copy/stream interactions being measured.

Export and summarize outside the GPU run:

```bash
nsys export --type=sqlite --output=.artifacts/response-default.sqlite \
  .artifacts/response-default.nsys-rep
python benchmarks/df_response_timeline.py .artifacts/response-default.sqlite \
  --output .artifacts/response-default-summary.json
```

The summary correlates raw-copy API calls with their actual H2D activities. It
also intersects the host API intervals with same-stream kernels and copies.
These are overlapping views: never sum API, event and device durations to obtain
an endpoint. Time outside observed device activity remains explicitly
unassigned runtime/staging/descheduling time; subtraction alone cannot identify
CPU packing. Untraced captures can identify the original strided-copy route;
the pinned-copy probe requires NVTX origins to distinguish raw and density
uploads that share `cudaMemcpyAsync`.

## Causal controls

`VIBEQC_DF_RESPONSE_UPLOAD_PROBE` has two diagnostic selections. Leave it unset
for production execution. Both controls retain complete response work and
transfer counts; neither is an automatic optimization or fallback.

| Selection | Prior stream work | Host gather | H2D submission |
| --- | --- | --- | --- |
| unset | Original dependencies | CUDA-managed pageable staging | Original strided 2D copy |
| `drain` | Explicitly timed stream drain before each raw read | CUDA-managed pageable staging | Same strided 2D copy |
| `packed` | Same explicit drain | Separately timed loop into one pinned AO slice | Contiguous pinned copy |

The drain mode tests how much host waiting moves out of the original copy call
without changing its source, destination, shape or bytes. The packed mode
exposes a measured host gather and real contiguous transfer. It is not a clean
speed comparison: it deliberately serializes the stream. Its one-slice pinned
allocation is charged to the bridge's host bound and survives until the owning
arena drains. The probes reject source-backed execution, where no host raw
copy exists. Actual drain/gather counters distinguish requested from executed
work, and returning to the unset route releases the diagnostic staging.

The `coulomb_response_charge_dot` component identifies the existing charge
reduction independently of the following upload's inclusive host time. The
three-center and metric derivative regions still contain arithmetic, subgroup
reduction and final gradient atomics in the same kernel. Their aggregate
duration is not an independently measured atomic or arithmetic duration.
Hardware counter availability and any additional sink ablation must be stated
before attributing a fraction of that fused interval.

`VIBEQC_DF_RESPONSE_SCATTER_PROBE=sharded` tests sensitivity to atomic destination
contention. It maps blocks round-robin onto 128 complete atom-gradient buffers,
retaining the same generated scalar arithmetic, subgroup reduction and number
of atomic sums. A final cuBLAS reduction returns the full physical gradient.
The extra buffers use unused bridge budget headroom; the auxiliary tile and raw
traffic stay fixed, and insufficient headroom fails rather than silently
changing the experiment. The default kernel specialization has no remapping.
Report the measured derivative-interval change and separate final reduction;
their difference measures sink-layout sensitivity, not pure arithmetic time.
This is a diagnostic control, not a promoted derivative schedule.

For clean endpoint measurements omit both tracing arguments and use at least
five repeats. Preserve 192/384 AO, batch 1/4, and the 19-AO UHF sanity case when
qualifying a performance candidate. Run compilation, profiling and reference
preflight separately from those samples. This attribution tooling alone does
not close #308 or establish promotion of a derivative schedule.

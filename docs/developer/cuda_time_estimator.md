# Experimental CUDA kernel timing

`generativeqc_compiler.common.cuda_time_estimator` turns compiler-visible work
counts into an engineering time estimate using **caller-supplied measured rates**.
It is an offline tool: it does not import the runtime, compile code, or probe CUDA.
It does not select production defaults. Measured
[RTX 5090 FP64 family profiles](../../benchmarks/results/cuda-timing-rtx5090-20261004/README.md)
are available for explicit use within their recorded workload and launch regime.
The original v1 calibration and formula remain supported.

The existing `common.cuda_cost_model` and `tools/analyze_cuda_cost.py` screening
reports keep their relative, non-timing contract. Timing is an optional layer.

## Calibration contract

`CudaTimingCalibration` requires all of the following:

| Field | Meaning |
| --- | --- |
| `device`, `architecture`, `sm_count` | Concrete device/SKU identity, canonical `sm_XX` target, and measured device topology |
| `workload` | Kernel family, arithmetic precision, operation-count convention, and traffic/cache regime |
| `provenance` | Retained calibration measurements and their software, clock/power, and timing conditions |
| `effective_compute_ops_per_second` | Achieved whole-device operation rate in the stated workload regime |
| `effective_memory_bytes_per_second` | Achieved bandwidth using the same byte-count convention as the candidate |
| `launch_seconds` | Positive incremental latency per serial launch, measured separately from body work |
| `saturation_occupancy` | Explicit occupancy threshold in `(0, 1]` for compute and, without a separate curve, memory |
| `uncertainty_fraction` | Explicit engineering fraction in `[0, 1]`; not a statistical confidence level |

Advanced terms require `model="roofline-calibrated-overlap.v2"`:

| Field | Meaning |
| --- | --- |
| `memory_throughput_curve` | Piecewise linear occupancy/achieved-rate fractions, monotone from `(0,0)` to `(1,1)`; empty uses linear scaling |
| `crossover_penalty_curve` | Piecewise linear occupancy/penalty pairs spanning occupancy 0 to 1, coefficients in `[0,1]`; empty means full overlap |
| `compute_wave_correction` | Correct compute throughput for uneven final SM waves, assuming equal block work |
| `batch_seconds` | Nonnegative fixed overhead charged once per nonempty batch |

The default model is `roofline-linear-occupancy.v1`. It rejects non-default
advanced terms, and its calibration payload retains the original v1 schema.

Peak specification rates are not calibration. FP32, FP64, tensor-core, and
special-function operation counts are not interchangeable. Likewise, semantic
bytes and observed DRAM bytes must not be mixed unless the achieved bandwidth
was calibrated with that same convention. Calibration workloads must avoid
counting launch latency twice in the body rates.

The estimator rejects architecture and known SM-count mismatches. Matching these
fields cannot prove the GPU SKU, precision, cache regime, or kernel family matches;
that remains the caller's responsibility. `workload` and `provenance` are required
labels, not automatic verification of their contents.

## Work and formula

One `StaticCudaCost` must describe one homogeneous kernel, possibly repeated with
the **same grid and resource shape**. Arithmetic operations, semantic bytes, and
explicit dynamic spill bytes are totals across all those launches. `grid_blocks`
is the grid size **per launch**. Work is not multiplied by `launch_count` again.

For v1:

```text
parallel_scale = min(1, device_occupancy_upper_bound / saturation_occupancy)
compute_seconds = total_operations / achieved_compute_rate / parallel_scale
memory_seconds = (total_semantic_bytes + total_dynamic_spill_bytes)
                 / achieved_memory_rate / parallel_scale
launch_seconds = launch_count * calibrated_launch_seconds
estimated_seconds = max(compute_seconds, memory_seconds) + launch_seconds
interval = estimated_seconds * (1 +/- uncertainty_fraction)
```

V2 starts with that compute scale, optionally corrects it for uniform-block SM
imbalance, and interpolates memory scale independently:

```text
q = ceil(grid_blocks / sm_count)
busy_occupancy = min(q, resident_blocks_per_sm) * per_sm_occupancy / resident_blocks_per_sm
compute_scale = grid_blocks / (sm_count * q) * min(1, busy_occupancy / saturation_occupancy)
memory_scale = interpolate(memory_throughput_curve, device_occupancy_upper_bound)
C = total_operations / achieved_compute_rate / compute_scale
M = total_bytes / achieved_memory_rate / memory_scale
k = interpolate(crossover_penalty_curve, device_occupancy_upper_bound)
body = max(C,M) + k * min(C,M)^2 / max(C,M)
estimated_seconds = body + launch_count * calibrated_launch_seconds + batch_seconds
```

The body is zero when both resource times are zero. The crossover correction
lies between roofline `max(C,M)` and serialized `C+M`, peaks near balanced work,
and decays when one resource dominates. `parallel_scale` reports compute scaling,
`memory_parallel_scale` reports memory scaling, and `overlap_seconds` is time
saved relative to serialized resource times. The wave correction requires known
grid/SM topology and assumes uniform work across blocks; it is unsuitable for
irregular block durations without further qualification.

The static occupancy bound is optimistic and can omit unknown resource limits;
all source diagnostics remain attached to the timing report. These formulas do
not resolve arbitrary instruction dependencies, cache effects, latency hiding,
or concurrent kernels. The engineering interval remains
`estimated_seconds * (1 +/- uncertainty_fraction)`, not a guaranteed bound on
execution. `bottleneck` can be `compute`, `memory`, `balanced`, `launch`, or `none`
for a known no-op; unknown estimates have no bottleneck classification.

PTXAS spill bytes are static resource evidence. They do not count executions
across threads, loop iterations, or launches and cannot be added to total semantic
traffic. If the compiler reports zero spills, the estimator assumes zero dynamic
spill traffic. Otherwise the caller must pass total `spill_traffic_bytes`, possibly
an explicit zero when those bytes are already included in semantic traffic.
Other traffic, such as non-spill local-memory accesses, belongs in semantic work.

Missing operations, bytes, launches, dynamic spill evidence, or whole-device
parallelism yield `None` for the total and interval. Independent known components
remain available. `allow_per_sm_fallback=True` explicitly permits an optimistic
estimate without global underfill evidence and is recorded in the report.
A zero grid or impossible resident-block count never becomes executable through
that fallback. A known zero-launch, zero-work request returns zero, including fixed
overhead; zero launches with nonzero work are rejected. Invalid numeric inputs raise, and overflowing
computed seconds remain unknown instead of emitting infinity or NaN.

## Offline CLI

Serialize a measured calibration with `CudaTimingCalibration.to_payload()` to
obtain a `generativeqc.compiler.cuda-timing-calibration.v1` or `.v2` JSON object,
then run:

```bash
PYTHONPATH=python python tools/analyze_cuda_cost.py \
  --arch sm_120 --block-threads 256 --grid-blocks 680 \
  --ptxas retained-single-kernel.ptxas.log \
  --operations 1000000000 --traffic-bytes 100000000 --launches 2 \
  --calibration measured-device-calibration.json
```

The counts above illustrate the interface, not measurements. Calibration supplies
SM count when `--sm-count` is absent. `--spill-traffic-bytes` supplies dynamic spill
work; `--allow-per-sm-fallback` opts into the fallback. Both require calibration.
Without `--calibration`, the CLI emits the existing static screening report.
With it, `time_estimate` adds the model identifier, calibration, static evidence,
parallelism basis/scales, component times, engineering band, and diagnostics.
Calibration schema and model versions must agree.
A timing report rejects multi-kernel PTXAS logs: resource maxima across different
kernels are only useful for screening. Headerless single-kernel logs retain an
explicit unverified-architecture diagnostic.

## Collect and reproduce calibration

`tools/calibrate_cuda_time.py` builds a standalone CUDA probe with `ccache` and
fits achieved rates from actual synchronized batch wall time. On the local RTX
5090 host, run collection through Slurm, preserving its device visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc 'PYTHONPATH=python python tools/calibrate_cuda_time.py \
    --nvcc /group/software/cuda-12.9.1/bin/nvcc --arch sm_120 --samples 9 \
    --suite refined --output .artifacts/cuda-timing-calibration'
```

The refined probe measures empty launches, eight independent FP64 FMA chains,
streaming copies, and dependent mixed FMA/streaming transforms. Input/output
arrays are each at least four times L2 capacity. Training varies grid size from
a quarter of the SM count to eight blocks per SM, with several intensities per
occupancy. Separate held-out grid sizes, iteration counts, array sizes, and launch
counts never participate in fitting. A host `std::fma` oracle checks 17 spread
indices per case with a `2e-12` absolute-error gate. Calibration requires complete,
spill-free PTXAS evidence. Add `--training-only` during model development to
collect no holdouts; those reports can never qualify. Freeze model choices before
collecting fresh validation data.

All raw samples are retained. The model uses median synchronized batch wall
time; CUDA event samples are retained separately and are not substituted for
wall time. The timed interval includes host submission, event recording, and
final event synchronization, and excludes allocation, initialization, transfers,
warmup, and numerical validation. This is a warm kernel-batch calibration, not
an isolated CUDA-event latency or a chemical endpoint measurement.

Training empty-launch batches determine a robust Theil-Sen incremental cost and
nonnegative fixed batch intercept. Pure compute rates account for SM imbalance;
copy throughput uses geometric means and monotone isotonic pooling. The streaming
fit searches a fixed compute-rate grid within +/-10% of its high-intensity
training anchor and bounded memory/crossover parameters at measured occupancy
knots. Arithmetic, copy and streaming profiles remain separate because their
instruction dependencies differ. No family is automatically inferred from a
production kernel. Choose `calibration-arithmetic.json`, `calibration-copy.json`
or `calibration-streaming.json` explicitly. The engineering band is the maximum
training residual relative to prediction, rounded up to 0.01 with a 0.05 minimum.
Held-out data do not change any parameter or band.

Refined qualification requires median/P95/maximum absolute relative prediction
error at most **5%/12%/20%**, both overall and for every held-out family; missing
families cannot qualify. Band coverage is reported separately, without a
confidence claim. Failed gates retain their reports and cause a nonzero process
exit. These gates qualify this probe domain;
they do not qualify register-pressure, spill-heavy, cache-resident, tensor-core,
other-precision, concurrent-stream, or complete chemistry workloads.

The retained [measurement and qualification bundle](../../benchmarks/results/cuda-timing-rtx5090-20261004/README.md)
contains source/compiler/binary hashes, PTXAS resources, Slurm allocation identity,
device clock/power snapshots, all work counts, raw wall/event samples, and scored
holdouts. It can be refitted on a CPU without CUDA or Slurm:

```bash
PYTHONPATH=python python tools/calibrate_cuda_time.py \
  --measurement benchmarks/results/cuda-timing-rtx5090-20261004/measurement.json.gz \
  --output .artifacts/cuda-timing-replay
```

Gzip replay preserves decoded sample bytes and their original SHA-256 identity.
The qualification also refits a v1 comparator using its original training recipe
(pure arithmetic/copy and launch batches of at least 256), then predicts every
same fresh holdout. V2 additionally trains on mixed kernels and small batches;
this comparison measures the complete calibration methods, not just the formulas.

`--suite legacy` (the default) retains v1 collection and fitting: a common
occupancy threshold from a fixed 0.001 search, geometric-mean achieved rates,
and a training-derived band rounded up to 0.05 with a 0.10 minimum. Its original
20%/35%/50% gates and
[retained calibration](../../benchmarks/results/cuda-timing-rtx5090-20261003/README.md)
remain replayable. Architecture and SM count alone do not establish compatibility
for either version's workload or software/clock regime.

## Endpoint boundary and qualification

The API does not compose an SCF or force endpoint. Estimate heterogeneous serial
kernels separately and add their times: `sum(max(compute_i, memory_i))` differs
from `max(sum(compute_i), sum(memory_i))`. Concurrent work needs a separate overlap
model; summing component times does not establish wall time in that case.

An endpoint consumer must explicitly account for compilation, host setup, plan
construction, grid/geometry rebuild, transfers/synchronization, iterations, and
force/response work. Cold, warm, and changed-geometry scenarios must record which
of those stages execute or reuse state. Iteration counts and cache invalidation
are method/runtime policy, not device calibration. Unknown stages cannot be
silently treated as zero.

Before relying on estimates for decisions, retain held-out measurements spanning
kernel families, sizes, underfilled grids, occupancy/resource pressure, and cache
regimes; record prediction residuals separately from the engineering band. Any
production promotion still follows the complete-endpoint, work-count, and
independent numerical gates in
[Performance engineering](../maintainer/performance_engineering.md).
Broader calibration, residual quantiles and learned residual models require fresh
evidence before becoming automatic behavior.

The scope and rejected alternatives are preserved in the
[calibrated timing decision](../../.agents/notes/implemented/performance/2026-10-03-calibrated-cuda-kernel-timing.md)
and [refined model decision](../../.agents/notes/implemented/performance/2026-10-04-refined-cuda-timing-model.md).

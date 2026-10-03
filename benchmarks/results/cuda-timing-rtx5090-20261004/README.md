# RTX 5090 refined CUDA timing qualification

Explicit FP64 family profiles for warm, serial, same-stream kernel batches.
These qualify the recorded probe domain; they do not select production defaults
or predict a complete chemistry endpoint.

Measured through Slurm job **12183** at 2026-10-03 16:38 UTC (2026-10-04 local),
on NVIDIA GeForce RTX 5090, 170 SMs, `sm_120`, driver 580.95.05, CUDA 12.9.86,
575 W power limit. Before/after clock and temperature snapshots, compiler and
ccache commands/statistics, PTXAS resources, source hashes and binary identity
are in `measurement.json.gz`.

## Model and independently held-out results

The v2 model separates compute and memory occupancy response, corrects uniform
compute blocks for uneven final SM waves, fits bounded compute/memory crossover
cost, and separates fixed batch overhead from incremental launch latency.
The three profiles must be selected explicitly for matching workloads:

| File | Scope | Achieved FP64 TOP/s | Achieved semantic TB/s | Engineering band |
| --- | --- | ---: | ---: | ---: |
| `calibration-arithmetic.json` | Independent register FMA chains | 1.634856 | 1.492980 | +/-7% |
| `calibration-copy.json` | Streaming copy and empty launch batches | 1.634856 | 1.492980 | +/-5% |
| `calibration-streaming.json` | Dependent FMA grid-stride transform | 1.980976 | 1.492980 | +/-5% |

FMA counts as two operations. Nonempty batches add **7.320674 microseconds** once
and **2.889326 microseconds** per launch. Ordinary kernels use 256-thread uniform
blocks; each streaming array is at least four times the 96 MiB L2 capacity.
All operations and bytes include every launch; no work count is extrapolated
from static spill bytes.

There are **55 training cases and 27 fresh held-out cases**, each with nine wall
samples and nine event samples: **738 of each**, with none removed. Training
includes grids 42/85/170/340/680/1360 and mixed intensities 2/8/32/64. Fresh
holdouts use grids 56/255/850, mixed intensities 3/6/12/24/48, different pure-FMA
iterations, copy array sizes and launch batch sizes. An exploratory training-only
collection preceded this formal run. Holdout timings never enter fitting or
engineering-band selection, and the model was not refitted after evaluation.

The baseline refits the original v1 recipe to this run's pure arithmetic/copy
training and launch batches of at least 256, then predicts **all the same 27
holdouts**. V2 also trains on mixed kernels and small launch batches; this is a
comparison of the complete calibration methods, not an equal-training ablation.

| Same fresh holdouts | Median relative error | P95 relative error | Maximum relative error |
| --- | ---: | ---: | ---: |
| Refitted v1 baseline | 7.39% | 23.01% | 53.67% |
| Refined v2 | **1.64%** | **10.36%** | **14.40%** |

| V2 family | Cases | Median | P95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Copy | 3 | 1.61% | 3.39% | 3.59% |
| Independent FMA | 3 | 1.36% | 9.61% | 10.52% |
| Empty launches | 6 | 0.31% | 3.95% | 4.68% |
| Mixed streaming | 15 | 2.10% | 11.30% | 14.40% |

V2 passes the stricter, predefined **5%/12%/20%** median/P95/maximum gates both
overall and within every held-out family. The training-derived engineering bands
cover **22/27** held-out medians (81.48%); these are not confidence intervals.
The worst error is `fresh-mixed-255-6`; all misses remain in the report.
Independent host `std::fma`/copy checks use 17 spread indices per case with a
2e-12 absolute-error gate. All cases pass. Compute Sanitizer reports zero memory
errors on the exact measured binary across all four families and non-divisible
grid-stride tails; `sanitizer.json` records Slurm job **12186**.

## Retention and source identity

`measurement.json.gz` and `qualification.json.gz` are lossless gzip archives of
the original JSON bytes. The latter retains every prediction, work count, gate,
band decision and the full matched baseline report. Profiles remain plain JSON
for direct CLI use. SHA-256 identities refer to decoded bytes; the measurement
digest is `ab78f6b3461fb3d02bfbe56588cd33d65bd0c5d4ae963aa241765825c8936fea`.

Collection ran on parent `60e70eff4` with local changes. Commit **`124055b86`**
preserves the exact four source files listed by hash in the measurement. Later
docstrings and stricter invalid-evidence validation leave this run's fit and
predictions unchanged, as verified by CPU replay. This source snapshot, rather
than the dirty parent alone, reconstructs the measured implementation. Build
objects and verbose logs remain ignored local artifacts.

## Reproduction

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc 'PYTHONPATH=python python tools/calibrate_cuda_time.py \
    --nvcc /group/software/cuda-12.9.1/bin/nvcc --arch sm_120 --samples 9 \
    --suite refined --output .artifacts/cuda-timing-refined-recollection'

PYTHONPATH=python python tools/calibrate_cuda_time.py \
  --measurement benchmarks/results/cuda-timing-rtx5090-20261004/measurement.json.gz \
  --output .artifacts/cuda-timing-refined-replay
```

The second command needs only CPU/Python and reproduces all profiles and reports.
Pass the matching family profile to `tools/analyze_cuda_cost.py --calibration`.
Architecture and SM count alone cannot establish workload compatibility.

Wall timing includes host submission, event recording and final synchronization;
allocation, transfers, warmup and numerical checks are excluded. Register/spill
pressure, cache-resident arrays, other precisions, block costs, concurrent streams,
software/clock regimes and chemistry setup/SCF/force stages need new qualification.
See the [current contract](../../../docs/developer/cuda_time_estimator.md) and
[decision note](../../../.agents/notes/implemented/performance/2026-10-04-refined-cuda-timing-model.md).

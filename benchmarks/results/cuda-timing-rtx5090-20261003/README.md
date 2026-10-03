# RTX 5090 CUDA timing calibration

This is a measured **FP64 arithmetic/streaming, warm serial kernel-batch** profile
for `common.cuda_time_estimator`. It is explicitly selected by the caller and does
not change production scheduling or qualify a chemistry endpoint.

Measured on NVIDIA GeForce RTX 5090, 170 SMs (`sm_120`), driver 580.95.05,
CUDA 12.9.86, 575 W power limit, through Slurm job **12164** on 2026-10-03 UTC.
The device was warmed before sampling; clock/power snapshots and exact source,
compiler, PTXAS and binary identities are retained in `measurement.json.gz`.

| Parameter | Measured/fitted value |
| --- | ---: |
| Achieved FP64 compute rate (FMA = 2 operations) | 1.587633 TOP/s |
| Achieved semantic streaming bandwidth | 1.484881 TB/s |
| Incremental launch cost | 2.864730 microseconds |
| Common saturation occupancy threshold | 0.144 |
| Training-derived engineering band | +/-15% |

The fit uses **23 training cases**, each with **9 raw samples**. The **17 held-out
cases** also have 9 samples each and never affect the parameters or error band.
Both input and output streaming arrays exceed L2 capacity by at least a factor
of four (L2 is 96 MiB). FP64 seed, initialization and reduction work is included;
semantic bytes and operations cover every launch in the timed batch.

| Held-out family | Cases | Median relative error | P95 relative error | Maximum relative error |
| --- | ---: | ---: | ---: | ---: |
| Streaming copy | 3 | 1.38% | 1.57% | 1.60% |
| Independent FP64 FMA chains | 3 | 4.35% | 8.97% | 9.48% |
| Empty launch batches | 2 | 0.30% | 0.44% | 0.46% |
| Mixed FMA/streaming, excluded from fitting | 9 | 7.97% | 20.60% | 21.79% |
| All held-out cases | 17 | 1.60% | 19.41% | 21.79% |

All families pass the predefined median/P95/maximum gates of 20%/35%/50%.
The +/-15% engineering band covers **15/17** held-out medians; it is not a
statistical confidence interval. The two misses remain in the report. The largest
absolute error against the independent host numerical oracle is
**8.881784197001252e-16**, below the 2e-12 gate. Compute Sanitizer memcheck reports
**zero errors** across all four kernel families and non-divisible tail cases.

`measurement.json.gz` retains all 360 wall samples and 360 event samples, work counts,
resource evidence and provenance. `calibration.json` is directly consumable by
`tools/analyze_cuda_cost.py --calibration`. `qualification.json.gz` retains every
prediction, residual, band decision and gate. `sanitizer.json` is the compact
sanitizer receipt for the exact measured binary. Gzip storage preserves the
original decoded bytes and SHA-256 identities. Build objects and verbose logs
remain ignored local artifacts.

## Reproduction

Collection uses `ccache`; this host exposes the GPU through Slurm:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc 'PYTHONPATH=python python tools/calibrate_cuda_time.py \
    --nvcc /group/software/cuda-12.9.1/bin/nvcc --arch sm_120 --samples 9 \
    --output .artifacts/cuda-timing-calibration'
```

CPU-only replay must reproduce the profile and qualification without CUDA:

```bash
PYTHONPATH=python python tools/calibrate_cuda_time.py \
  --measurement benchmarks/results/cuda-timing-rtx5090-20261003/measurement.json.gz \
  --output .artifacts/cuda-timing-replay
```

Use the measured profile for matching candidate work (this example supplies
caller-estimated resources and explicitly declares zero additional spill work):

```bash
PYTHONPATH=python python tools/analyze_cuda_cost.py \
  --arch sm_120 --block-threads 256 --grid-blocks 680 \
  --estimated-registers 22 --operations 1000000000 \
  --traffic-bytes 100000000 --launches 2 --spill-traffic-bytes 0 \
  --calibration benchmarks/results/cuda-timing-rtx5090-20261003/calibration.json
```

The measured endpoint starts before batch submission and ends after final event
synchronization. Allocation, transfers, warmup, and numerical validation are
excluded. Other precisions, dependency structures, register pressure, spills,
cache-resident arrays, concurrent streams and chemistry setup/SCF/force stages
need separate qualification. See the
[current contract](../../../docs/developer/cuda_time_estimator.md) and
[decision note](../../../.agents/notes/implemented/performance/2026-10-03-rtx5090-cuda-timing-calibration.md).

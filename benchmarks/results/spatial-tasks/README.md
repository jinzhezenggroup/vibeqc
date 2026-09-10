# Bounded spatial AO task evidence

These bundles accept the numerical behavior of the explicit local candidate
from #234. CPU measurements include complete fixed-density polarized PBE
energy and potential. CUDA measurements include AO/features and the native
consumer/scatter boundary; the scatter matrix is diagnostic host input, not
GPU XC output. No complete GPU XC integration or production promotion is
claimed. Dense execution remains available.

## Inventory and numerical scope

Each backend retains 24 cases with five preparation/execution samples each:
4/16 separated atoms, contracted s/p/f shells with signed coefficients,
Cartesian/spherical AOs, tiles 16/64, and dense/local-unscreened/local-screened
modes. Tile 16 uses 32 MiB host and 160 MiB device budgets; tile 64 uses 64 MiB
host and 256 MiB device budgets. The local mode composes these through the
shared resource planner. The legacy dense mode records its aggregate numeric
budget and does not claim shared host/device replacement accounting.

Every arithmetic gate uses `atol=1e-11`, `rtol=1e-10` against bounded CPU
collocation with the **identical AO mask**. This checks local maps, complete
density submatrices, spin features and symmetric matrix assembly. Independent
CPU AO and existing external XC fixtures remain covered by regression tests.
The screening cutoff is 1e-9 for every ordinary AO derivative through order
one on deterministic 32-point regions. Differences from unscreened
collocation are separate `approximation_difference` diagnostics. They are not
rigorous density, energy or force error bounds. Changing the hardware tile
preserves the region/mask definition.

`samples.json` retains every timing, quantitative error, input identity,
resource plan and native observation. It does not retain full AO or potential
arrays. `summary.json` can be reconstructed from those samples; the publisher
validates inventories, fixed tolerances, source/build consistency, serialized
resource checksums/derived accounting and observed native capacities.
`publication.json` binds the files by checksum. Integrity tests also reject
corrupt inventories, timing intervals, gates, plans and dense comparison
sources. Numerical acceptance leaves performance/production stages `not-run`.

## Timing and memory interpretation

All top-level times are synchronized host wall time. Construction includes
partition/envelope/map preparation and native owner creation, and excludes
the separately built CPU library and cold CUDA compilation. Feature timing
includes density upload, tile work, diagnostic feature downloads and assembly
of the bounded diagnostic result. CPU XC timing executes collocation,
features, functional evaluation and complete E/V assembly again. Add
construction to XC for its complete cold endpoint; do not add the preceding
diagnostic feature sweep a second time.

CUDA `device_consumer_scatter_seconds` is a second traversal with borrowed
device views, diagnostic local matrix uploads, scatter and one final matrix
download. `scatter_calls_seconds` is its scatter-call subset. Native metrics
are cumulative across both traversals; `consumer_metric_delta` isolates the
second traversal. Event fields separate input/output copies, AO kernels,
library contractions and packing. CPU `cpu_sections` similarly accumulates
AO/density work across feature and XC calls and is not a separate endpoint.

For 16 atoms, screened Cartesian tasks use 22–38 of 224 AOs; spherical tasks
use 20–33 of 176. Screening construction costs about 1.0–1.4 seconds. For
example, Cartesian tile-64 CPU XC takes about 0.035 seconds after screened
construction versus 0.068 seconds dense, but construction dominates one
evaluation. Cartesian CUDA feature times are about 0.0085 seconds screened
versus 0.0114 seconds dense at tile 64; again, these exclude construction.
These workloads show when reuse may help, not a universal speedup.

Local resource plans include numeric metadata, retained quadrature,
host/device panels, gather/scatter storage, full global D and V (still
O(NAO²)), and a declared 96 MiB cuBLAS allowance. The native arena and
observed cuBLAS retention are checked against those capacities. Resource
peaks are capacity bounds, not measured process peaks. Python object headers,
allocator rounding, caller-retained detached tiles and CUDA context/modules/
stacks beyond the allowance are explicit exclusions. Replacement peak and
lease-lifetime behavior are exercised by tests, separately from these fresh
construction timings.

A later review fix makes each density upload automatically clear the local
plan's global potential. Repeated device-task executions therefore use default
scatter safely. The original measured workers explicitly reset their first
scatter; archived timings retain those exact historical sources and do not
measure the later automatic reset. The updated path has separate repeated-
execution numerical and sanitizer coverage, without a new performance claim.

The optimized CUDA AO kernel uses 50 registers, down from 122 in the first
local implementation. The feature/gather/scatter kernels use 56/32/28
registers. All report zero spills, stack and shared memory. Cold compilation
of the retained CUDA artifact took 3.9194 seconds. Full CPU build wall time
and the CPU processor model were not captured; the bundle says so explicitly.

## Historical dense comparison

The CUDA bundle additionally retains three process blocks per source in the
actual order baseline/candidate/candidate/baseline/baseline/candidate. Each
block contains five samples of all eight dense cases. These are three process
blocks, not fifteen independent interleaved trials and not a shared
significant-speedup gate. Median candidate/baseline feature ratios are
0.997–1.007; construction ratios are 1.044–1.089, exposing added preparation
cost. No negative sample or slower case was removed.

Measured final source is `93ac3fadbfdddd5cac70dfb3581d52630018aa44`.
The historical optimized candidate is its ancestor
`70f0c43f4bc9b5e5423504bdcd207d537dd810e8`. Both are retained in
[`njzjz-bot/vibeqc: evidence/issue-234-measured`](https://github.com/njzjz-bot/vibeqc/tree/evidence/issue-234-measured).
The dense baseline `1ba6f175caed656729e0863880251cc315232b1c` is retained in
[`evidence/issue-231-baseline`](https://github.com/njzjz-bot/vibeqc/tree/evidence/issue-231-baseline).
Later integration commits do not replace these measured identities.

## Reproduction

Fetch the source branches above and use isolated clean checkouts at the exact
recorded SHAs. Use Python 3.11 with the repository's test/reference
dependencies, one OpenMP/OpenBLAS thread, GCC 11.4 and CUDA 12.9.1 targeting
sm_120. The measured GPU was an RTX 5090 with driver 580.95.05. Build a matching
CPU library for each checkout (it supplies normalized AO records and the CPU
reference even for a CUDA worker):

```bash
cmake -S <measured-checkout> -B <measured-checkout>/.artifacts/spatial-cpu-build \
  -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=OFF \
  -DVIBEQC_ENABLE_AOT_SHELLS=OFF -DVIBEQC_BUILD_TESTS=OFF
cmake --build <measured-checkout>/.artifacts/spatial-cpu-build -j 8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/benchmark_spatial_tasks.py \
  --root <measured-checkout> \
  --library <measured-checkout>/.artifacts/spatial-cpu-build/libvibeqc.so \
  --cache .artifacts/spatial-cuda-cache --backend cpu --samples 5 \
  --output .artifacts/reproduction/234-cpu-final.json
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python tools/benchmark_spatial_tasks.py \
  --root <measured-checkout> \
  --library <measured-checkout>/.artifacts/spatial-cpu-build/libvibeqc.so \
  --cache .artifacts/spatial-cuda-cache --backend cuda --samples 5 \
  --output .artifacts/reproduction/234-cuda-final.json
```

The reproduction driver selects `/group/software/cuda-12.9.1/bin/nvcc`
explicitly; adjust that compiler path for an equivalent local installation.
Preserve Slurm's assigned device visibility. The existing content-addressed
CUDA compiler captures toolchain, flags, source/header hashes, binary hash,
compilation time and kernel resources; the worker checks CPU native source
identity. The six historical workers must also use **CUDA through Slurm**:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python tools/benchmark_spatial_tasks.py \
  --root <historical-checkout> \
  --library <historical-checkout>/.artifacts/spatial-cpu-build/libvibeqc.so \
  --cache .artifacts/spatial-cuda-cache --backend cuda --dense-only --samples 5 \
  --output .artifacts/reproduction/234-optimized-dense-<side>-<index>.json
```

Select the exact baseline/candidate revisions listed above and run in order
`baseline-0`, `candidate-0`, `candidate-1`, `baseline-1`, `baseline-2`,
`candidate-2`. Each worker is a separate process with a matching CPU library
for its independent reference. CPU-only historical workers are not accepted
by this CUDA comparison. Then publish into a new directory:

```bash
python tools/publish_spatial_tasks.py --artifacts .artifacts/reproduction \
  --destination benchmarks/results/spatial-tasks/new-reproduction
```

The publisher never overwrites a bundle. For an archival integrity replay,
the two final `samples.json` files and expanded `runs` from
`cuda/dense-comparison-samples.json` restore all eight worker JSON inputs
without requiring a GPU. A new measurement requires actual native execution.

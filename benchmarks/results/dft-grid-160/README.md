# DFT01: atom-centered quadrature and bounded AO/density jets

These records validate issue #160 at clean scientific revision
`2c4ecf01710a81e66b6c5fd0b9109b677071f767`. The final CPU/CUDA endpoint job was
Slurm **9006**, using `main` and `--gres=gpu:5090:1` with a finite 15-minute
limit. `manifest.json` preserves SHA-256 hashes of the raw evidence files.
The subsequent review adjustment only isolates nonnumeric density input types
in ragged batches; CPU/GPU regression logs cover it separately. It changes no
numerical kernel, grid prescription or measured successful execution path.

## Numerical and ownership gates

| Gate | Result |
| --- | --- |
| Complete clean-checkout Python suite | 792 passed, 129 optional/GPU skips |
| Focused CPU grid/state tests | 30 passed |
| Native CPU suites | 10 passed |
| Real CUDA tests, including shared post-HF cache regression | 16 passed |
| Compute Sanitizer on all 16 tests | 0 errors; 0 bytes leaked |
| Independent fixture regeneration | All six array/hash sets identical |
| Endpoint fixtures | H2, asymmetric H2O, actual Cartesian/spherical f, diffuse, tight |
| Complete individual executions | 240: cold, warm, changed geometry and changed replay |
| Interleaved individual timing samples | 180 |
| Interleaved complete ragged-batch samples | 10 (five per backend) |
| Maximum identical-fixture scaled error | `0.00014615920258552207` |
| Maximum full-grid/changed-grid scaled error | `0.0006092295259252939` |
| Element gates | `atol=1e-11`, `rtol=1e-10`; scaled error must be ≤1 |
| Largest finest-grid electron-count error against Tr(DS) | `3.492227040879925e-8` electrons |

The full local Python run recorded 791 passes, 129 optional/GPU skips and one
source-identity mismatch because a cache metadata edit followed library load.
Rebuilding and rerunning that exact check passed (`identity-recheck.log`). The
[clean-checkout implementation CI](https://github.com/jinzhezenggroup/vibeqc/actions/runs/34205348188)
passed all 792 tests, with 129 optional/GPU skips (`ci-python-summary.log`).

The fixture gate compares every AO derivative through third order and every
spin density invariant against pinned **PySCF 2.14.0 / NumPy 2.4.6**. Identical
unpartitioned atomic grids also compare PySCF's native Becke weights. Fixtures
include arbitrary non-SCF density matrices and occupied-orbital checks. The
largest raw fixture error is `1.430511474609375e-6` in a very large tight-basis
sigma value; it passes the same scale-aware gate, without a relaxed boundary
tolerance. These tests are distinct from quadrature convergence.

Each case records raw density-integral errors at `(nradial,npolar,nazimuth)`
`(16,8,16)`, `(32,12,24)` and `(64,20,40)`. The finest value must improve over
the coarse grid and approach the independent overlap trace within `1e-5`
electrons. Actual errors are retained, with no post-hoc renormalization or
claim of monotone convergence for all possible grids. Separate CPU tests
integrate Gaussian and Slater radial functions with known analytic answers.

## Complete workflow costs

Median times in milliseconds, measured with identical inputs and interleaved
backend order across five trials:

| Case | CPU cold | CUDA cold | CPU replay | CUDA replay | CPU changed | CUDA changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2 | 44.059 | 31.716 | 42.554 | 27.714 | 44.039 | 31.749 |
| H2O | 104.881 | 53.639 | 103.290 | 49.468 | 104.925 | 53.694 |
| f Cartesian | 91.966 | 36.702 | 90.319 | 32.605 | 91.748 | 36.772 |
| f spherical | 95.830 | 33.199 | 93.727 | 29.253 | 95.524 | 33.270 |
| Diffuse | 43.608 | 31.806 | 42.174 | 27.703 | 43.598 | 31.776 |
| Tight | 53.707 | 31.953 | 52.292 | 27.939 | 53.683 | 31.999 |

CUDA replay is **1.52–3.20×** faster than this VibeQC native CPU route. The
six-item ragged batch takes **425.426 ms CPU / 195.093 ms CUDA** (2.18×).
This is a measured comparison against VibeQC's CPU implementation, not a
GPU4PySCF comparison or a claim of performance leadership. No replacement
schedule or public DFT method is promoted.

All timing paths include host grid generation/partition, AO evaluation,
density validation/contraction, transfers, host publication, and integration
of rho/tau. Cold times include basis/grid/device preparation; changed times
include transactional geometry rebuild and execution. The last atom moves
by `(0.05,-0.03,0.02)` Bohr while D stays fixed. Independent CPU/GPU field
comparisons precede timing for both geometries. Native section timings are
synchronized and instrumentation remains enabled in both correctness and
timing runs. Setup, grid, AO, BLAS, feature reduction and transfer costs appear
separately in the per-execution diagnostics. There is no SCF or XC calculation.

Host-built points upload to the GPU; native CUDA AO jets, cuBLAS contractions
and sigma evaluation execute on device; feature outputs explicitly download.
Ordinary private streams are used. The ragged wrapper executes independent
plans sequentially, without a fused batch-launch claim.

## Resources and capacity scope

Hardware: NVIDIA GeForce RTX 5090, UUID
`GPU-8e9c9e1a-e183-258c-0b3a-03a5ddebb2f8`. NVCC/PTXAS are **12.9.86**, CUDA
runtime **12090**, CUDA driver API **13000** (NVIDIA driver **580.95.05**).
The actually loaded **cuBLAS version is 120402**, recorded independently of the
compiler toolkit. The native runtime enforces the compiled `sm_120` target;
other architectures have no numerical validation in this archive.

Runtime compilation took **3.267174249 seconds**. AO and feature kernels use
**74 and 56 registers** respectively, with **zero stack/spills**. The shared
validation kernel uses 26 registers. These PTXAS records do not describe
cuBLAS's internal kernels. Native-library, source, compiler and artifact hashes
are retained in the JSON records.

| Largest capacity | Bytes |
| --- | ---: |
| One CPU plan | 1,388,736 |
| One CUDA plan | 106,769,856 |
| One native device arena | 4,717,824 |
| Old + replacement CUDA plans during geometry update | 213,539,712 |
| Six retained CPU plans | 5,091,744 |
| Six retained CUDA plans | 635,802,528 |
| Both validation fleets simultaneously live | 640,894,272 |

Native arena sizes equal their plans. Observed retained-provider deltas were
**67,108,864 or 69,206,016 bytes**, within the charged 96 MiB allowance per
handle. Explicit workspace is charged separately. The capacity includes
numeric metadata, density matrices, point/jet/work/output tiles, partition
scratch and quadrature setup scratch. Object headers, allocator rounding,
internal BLAS host workspace, CUDA context/modules/stacks and additional
caller-retained exports/iterators are outside this scope. These are not total
process/VRAM upper bounds. See [the interface contract](../../../docs/dft_grid.md)
for host setup and transactional replacement details.

## Reproduction

The recorded Python is 3.11.14 at
`/home/jzzeng/codes/qc/build/gpu4pyscf-venv/bin/python`. Build the native CPU
library in `build/cpu`; it supplies basis normalization for both backends.
The shared finite compiler adapter builds the separate CUDA artifact.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  env PYTHONPATH=.:python VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so \
  OMP_NUM_THREADS=1 python tools/validate_grid.py --cuda \
  --nvcc /group/software/cuda-12.9.1/bin/nvcc \
  --cache /tmp/dft160-cuda-cache --output /tmp/dft160-final-evidence
```

GPU tests use the same finite Slurm form with `VIBEQC_GRID_CUDA_TEST=1` and
`python -m pytest tests/python/test_grid_cuda.py -q`. Sanitizer prepends
`compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99` to
the Python command. All raw reports are included. No numerical GPU command
overrides the visibility assigned by Slurm.

Routine log/XML files named in this historical account are now represented in
[the retention audit](../retention-238/migration.json), with extracted measurements,
diagnostic conclusions, and exact original Git/checksum identities.

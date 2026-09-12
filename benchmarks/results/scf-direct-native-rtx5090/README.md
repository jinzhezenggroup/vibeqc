# Direct-native extraction validation

`validation.json` records complete Release build samples and matched prepared
endpoints for the exact parent and extracted source identities. It retains each
numerical comparison, geometry, timing repeat, iteration count and library hash.
The original endpoint worker is `tools/validate_weighted_eri_endpoints.py`.

These build timings, binary sizes and parent/candidate endpoint samples predate
the upstream MP2 integration at `b69ec53`. They describe the recorded source
identities; integration validation is recorded separately and does not change
the scope of these historical measurements.

`integration.json` records the merge with `b69ec53` at source `c5301f3`. Both
the development and optimized Release libraries pass 22 native suites, 214
Python cases without skips, and four weighted-integral runs (3,349 records /
141 tiles per library), under Slurm allocations 9302 and 9303. It includes
physical HF reference export, public CPU/CUDA MP2, identical-orbital component
and permutation checks, typed allocation failures, bounded replay, and the
independent 14-AO reference with a partial final virtual block. The record
retains library hashes, explicit opt-ins, test commands and weighted results.
Five current CMake graph checks cover real-only, virtual-only, combined,
standalone and global-RDC configurations.

Reproduce a fixed RHF endpoint for each library with:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VIBEQC_LIBRARY=/absolute/path/to/libvibeqc.so VIBEQC_PROFILE=off \
  VIBEQC_BOUNDED_DIRECT_STREAMING=none VIBEQC_PSSS_RESIDENT_BRA=0 \
  VIBEQC_DIRECT_TILE_VALIDATION=validate \
  python tools/validate_weighted_eri_endpoints.py \
  --worker '{"method":"rhf","basis":"sto-3g","batch":3,"repeats":5}'
```

Repeat with `method=uhf`. For resident traversal set
`VIBEQC_PSSS_RESIDENT_BRA=1`; for paged traversal set
`VIBEQC_BOUNDED_DIRECT_STREAMING=force` and `VIBEQC_PSSS_RESIDENT_BRA=0`.
The worker checks every warm result against its corresponding cold geometry.
Cross-library comparisons use that module's `require_equal`, with energy
`atol=2e-8, rtol=2e-10` and per-molecule force `atol=2e-7, rtol=2e-8`.
The recorded maximum difference is much smaller than these established gates.

The build records retain full CMake flags: fresh Release build trees, NVCC
12.9.1, architecture `120` (real plus virtual images), fast-compile mode off,
compiler cache off, two CUDA jobs and four total Ninja jobs. The builds ran
concurrently on a shared host with a warm filesystem cache. The extraction was
committed during its build without changing the measured source; configure-time
HEAD and source commit are recorded separately. Runtime uses the additional
C++ `<cstdlib>` portability fix. Binary size and timing claims apply only to
these measured configurations.

The response suite requires `VIBEQC_DF_DERIVATIVE_CUDA_TEST=1`. The corrected
Slurm run and the preceding skipped attempt are distinguished in the record.
The virtual-image probe changed only the direct subsystem's images; it is not
a complete PTX-only whole-library benchmark.

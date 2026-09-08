# Weighted ERI derivative gates (#144)

The generated psss candidate remains **opt-in**. Correctness and memory checks
pass, but endpoint timings do not establish consistent non-regression against
the retained handwritten expression. Existing default algorithms, density
screens, and fixed/resident/paged task domains remain selected as before.

The final hardware run used Slurm job **8984**, `main`, `gpu:5090:1`, on the
RTX 5090 with CUDA 12.9.86, Release, sm_120, AOT enabled, fast compile disabled,
and split compilation set to one. All runs preserve Slurm device visibility.

## Numerical and memory results

- [numerical.json](numerical.json): 3,349 primitive records, 141 output tiles,
  four route/budget combinations, and independent libcint scalar/center
  derivatives. Maximum absolute error: `9.9921e-14`.
- Arbitrary ordered and normalized-pair weights, spherical pullbacks,
  coincident centers, mixed output tiles, long contractions, zero weights,
  and unit-weight raw diagnostics are included. Representative fsss is used;
  this does not duplicate #135's full f-shell matrix.
- 63 finite-difference coordinates/step combinations cover physical-atom
  gradients at `2e-3`, `1e-3`, and `5e-4` Bohr. The finest-step maximum error
  is `1.84293e-8`.
- Upload capacities of 7 and 113 records give the same results. Maximum
  reported native host-plus-device numeric storage is 52,832 bytes. Generated
  runs contain 333 generated psss records and 3,016 fallback records; reference
  runs evaluate all 3,349 through the independent fallback.
- [memcheck.log](memcheck.log): Compute Sanitizer reports zero errors. The
  sanitized scalar/gradient output also agrees with the ordinary run.

Native allocation accounting includes host results, device results, and the
reused device upload buffer. Caller inputs, CUDA context/runtime allocations,
and implicit kernel stacks are excluded. The host adapter reports its own
per-tile numeric bound; neither number claims to bound process RSS or all
device memory.

## Complete endpoints

[endpoints.json](endpoints.json) indexes four raw sample files and their hashes.
The matrix contains 24 scenarios and 72 fresh-process endpoint runs: RHF/UHF,
STO-3G/def2-SVP, batches 1/3, fixed/resident/paged queues, and previous-library,
retained-expression, and generated-expression routes. Each endpoint includes
cold execution, three warm samples, changed geometry, and three explicit
changed-geometry warm samples. Batch three mixes a cluster, an s-only system,
and another cluster. RHF uses three waters; UHF uses OH plus two waters.

All route comparisons pass, as do 720 additional whole-batch schedule and
first-molecule cross-batch checks. The maximum cross-check difference is
`3.0051e-12`. Separate fixed-queue profiles require nonzero psss primitive work;
native class counters do not cover paged queues. Profile timings are excluded
from endpoint timing comparisons.

The observed retained/generated time ratios span 0.976–1.057 for original
warm samples and 0.960–1.025 for changed-geometry warm samples. Cold and changed
geometry ratios also cross one. These are measured samples, not a promotion
claim. The previous library is recorded separately to expose effects on the
retained route from the shared native integration.

An exploratory three-OH fixture failed changed-geometry SCF convergence in
the **previous library**. It was replaced before the final matrix with one
radical plus two closed-shell waters. A single water was also unsuitable as
a direct-path benchmark because it is below the native 16-AO persistent-ERI
cutoff. The final inputs and multiplicities are stored in every sample file.

## Code and resources

[provenance.json](provenance.json) records source, generated header, native
object, library, and validation-driver hashes. The scientific implementation
is commit `cdb0ef995f44cf84e903282e8251f192036bc480`; later changes update the
validation drivers, documentation, and these artifacts.

[isolated-resources.json](isolated-resources.json) retains both materialized
and inline-single-use candidates. The precontracted graph has 225 nodes,
compared with 318 in the previous component-cloning graph. Both isolated
helpers use 64 registers and zero stack/spills; each takes approximately one
second to compile. Native main-object compilation takes about 1,237 seconds,
versus 1,240 seconds in the preceding build. These observed build times include
the existing monolithic CUDA translation unit and are not isolated causal
measurements of this change's compilation overhead.

[native-resources.txt](native-resources.txt) contains final cuobjdump records.
The actual generated external psss kernel uses 84 registers and no stack.
The correctness-oriented generic fallback uses 214 registers and a 104,672-byte
implicit stack per thread; this is explicitly outside the numeric buffer
budget. Native psss force workers retain their existing queue mappings.

## Reproduction

Build the Release CUDA library and manual `vibeqc_weighted_eri_probe` CMake
target. Set `PYTHONPATH=.:python`, `OMP_NUM_THREADS=1`, and `VIBEQC_LIBRARY` to
that library. Run the following commands inside a finite Slurm allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:20:00 bash -lc '
set -e
python tools/validate_weighted_eri.py \
  --probe build/cuda-release/vibeqc_weighted_eri_probe \
  --output build/weighted-final-numerics
compute-sanitizer --tool memcheck --error-exitcode 2 \
  build/cuda-release/vibeqc_weighted_eri_probe \
  build/weighted-final-numerics/records.bin sanitized.bin 52832 1
python tools/validate_weighted_eri_endpoints.py \
  --baseline-library /path/to/previous/libvibeqc.so \
  --repeats 3 --output build/weighted-final-endpoints.json
'
```

Host checks: 77 focused tests, nine native CPU suites, 692 Python tests passing
in CI with 74 GPU-specific skips, and pre-commit. Large sample JSON files are
compact; the index and provenance preserve their identities.

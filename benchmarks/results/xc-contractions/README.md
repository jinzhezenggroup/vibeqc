# Native CPU XC contraction evidence

This bundle accepts the numerical scope of #236: fixed-density LDA/PBE E/V,
spin-resolved matrix-free response, and independent AO-center, grid-point and
weight partials. Native CPU execution remains an explicit experimental candidate.
It registers no complete KS/CPKS, stationary molecular-force, Hessian or GPU XC
capability. #231 remains open for its broader scientific-kernel consolidation.

Measured source: **`60dceca7561e6bd04d61d1bbb5ee092044733275`**, clean.
It is permanently reachable on
[`njzjz-bot/vibeqc:evidence/issue-236-measured`](https://github.com/njzjz-bot/vibeqc/tree/evidence/issue-236-measured).
The measurement predates this evidence publication; its SHA is not relabeled
as the later documentation/test commit.

Later review fixes publish cached compiler inputs atomically, preserve matching
cache hits, verify the compiled source identity, and strengthen output-set and
svec metric checks. The retained source and timings remain those of the exact
measured revision above; no current-head construction timing is inferred.

Native source identity:
`c2167197e5aef83f7e66937ae3832c6ebd6a30d3928fccb733e862a313ac6242`.
Measured CPU library SHA256:
`88bc50af34e621b59a0ac22395964e639bbc605f8eb88d9f60efeb2c2b4e6909`.
Rebuilding must reproduce the native source identity; binary hashes can depend
on the recorded toolchain and paths. Every compiled point artifact also retains
its exact compiler, options, source/binary hashes and host scientific owners.

## Coverage and independent gates

The 128 cases contain four geometries (H2, Cartesian f, spherical f, and four
separated atoms with signed s/p/f contractions), two functionals, four observables,
dense/local masks, and tile sizes 7/19 under 4/8 MiB host budgets. Each has five
raw construction/execution samples: **640 complete samples**. The original small
fixtures retain all AOs in their local masks; the separated case retains **14 of
56 AOs** in each region, so its local gather/scatter is a strict subset.

All native versus identical-mask arithmetic blocks pass at atol=1e-11 and
rtol=1e-10. Dense E/V gates use pinned independent PySCF/Libxc fixtures; the
additional separated case runs the pinned independent NumInt implementation
on deterministic inputs outside timed calls. The reference owns AO evaluation,
normalization, density, scalar XC and potential assembly.

Independent density/response projections and each center/point/weight motion,
plus combined motion, retain all raw plus/minus values at steps
`1e-3, 3e-4, 1e-4`. All **144 projections** pass the 3e-8 absolute gate; the
largest observed discrepancy is **1.33e-10**. Shared PairSpace svec transpose
gates preserve sqrt(2) off-diagonal factors. Tests separately cover both
functional-spin layouts, spin exchange, asymmetric directions, stale masks,
vacuum domains, omitted source/factor faults, empty masks, overflow and actual
nonempty local subsets.

Local arithmetic uses the same fixed mask for every differentiated observable.
Differences from unscreened independent collocation are separate diagnostics,
not an approximation error bound. No physical motion is inferred from a frozen
mask or the current value-only Becke partition implementation.

## Cost and memory observations

Eight cold generated point-program compilations total **2.249 seconds**.
Source sizes range from 3,275 bytes (LDA energy) to 57,176 bytes (PBE response).
PBE potential coefficient SSA has 24→18 materialized values with estimated
peak liveness 8→8; response has 54→42 materialized values, arithmetic operations
54→48 and peak liveness 9→10. The baseline already used compact AO panels;
these statistics describe the shared Graph rewrite, not compression of an
invented AO-pair expansion.

Each evaluated tile records native scalar/coefficient/pullback calls and all
logical density, geometry and assembly matrix products. Matrix products are
not an assertion about BLAS's internal kernel launch decomposition. No global
AO table or four-AO-index XC tensor is stored.

For the 56-AO separated PBE potential case, the five-sample medians are:

| Schedule | Tile | Construction | Execution | Numeric plan peak |
| --- | ---: | ---: | ---: | ---: |
| Dense | 7 | 1.84 ms | 12.78 ms | 944,832 B |
| Dense | 19 | 1.82 ms | 6.13 ms | 1,657,536 B |
| Local, 14 active AOs | 7 | 702.66 ms | 17.45 ms | 1,356,192 B |
| Local, 14 active AOs | 19 | 700.91 ms | 17.45 ms | 2,132,640 B |

Local construction and small regional tiles dominate here. These measurements
do not establish a speedup, a historical non-regression gate or production
promotion. No performance threshold was relaxed to accept numerical correctness.

Resource plans cover declared numeric capacities, including immutable copies,
full spin matrices and borrowed spatial requests. Their documented exclusions
include Python object/allocator overhead, BLAS workspace/thread stacks, native
scalar stack/code pages and retained prior results. Separately, the worker's
Linux process-lifetime RSS high water is **124,440,576 bytes**. It includes DAGs,
loaded libraries, earlier cases and independent reference work; it is not a
per-endpoint allocation delta or the shared numeric cap. Per-case tracemalloc
peaks come from separate untimed executions, with untraced native allocations
explicitly outside that observation.

## Reproduction and retained files

Fetch and check out the exact measured SHA in an isolated checkout, install
NumPy and the pinned development-only PySCF 2.14.0/Libxc 7.0.0 environment, and
build its matching native CPU library:

```bash
cmake -S . -B .artifacts/xc-cpu-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=OFF \
  -DVIBEQC_ENABLE_AOT_SHELLS=OFF -DVIBEQC_BUILD_TESTS=OFF
cmake --build .artifacts/xc-cpu-build -j 8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:. \
  python tools/benchmark_xc_contractions.py \
  --library .artifacts/xc-cpu-build/libvibeqc.so \
  --cache .artifacts/xc-audited-cache --output .artifacts/xc-audited.json
python tools/publish_xc_contractions.py --run .artifacts/xc-audited.json \
  --stage .artifacts/xc-audited-publication \
  --destination .artifacts/reproduced-xc-publication
```

Use fresh cache/output/destination paths and a clean source tree. Timed work is
CPU-only and ran after local test/build activity finished. This is one worker
process with five repetitions per case; those repetitions are not independent
process trials. Full native-library build time is not included in the generated
point compilation total. Complete program construction, basis/spatial/prepared
construction and execution times remain separate in the samples.

`samples.json`, `summary.json`, `evidence.json` and `publication.json` form the
shared-policy publication. The approximately 2.2 MiB sample file has a hash-pinned
review-size exception because it retains the full quantitative inventory.
Transient compiler output and benchmark logs remain under `.artifacts/`.

The supplemental `sanitizer.json` retains 16 CPU ASan/UBSan ABI harness results
covering LDA/PBE, both spin layouts, all four observables, malformed counts,
null pointers and zero/overflow sizes. All emitted point-source hashes were
verified identical at the measured revision. It does not relabel the normal
native cache as instrumented. To rerun:

```bash
python tools/check_xc_native_sanitizer.py --output .artifacts/xc-sanitizer
```

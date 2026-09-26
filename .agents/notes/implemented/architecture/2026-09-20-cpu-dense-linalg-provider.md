# Decision: compiler-owned CPU dense linear algebra provider

Status: implemented
Date: 2026-09-20

## Problem

CPU dense linear algebra had no VibeQC-owned provider boundary. CPU reference code used
independent hand-written routines while GPU production paths already used cuBLAS/cuSOLVER.
Adding OpenBLAS calls directly to SCF, DF, post-HF, or TensorIR consumers would couple
scientific code to one vendor library and make threading/oversubscription policy implicit.

## Decision

Introduce `vibeqc::tensor::CpuLinalgPlan` and a native CPU dense-linear-algebra provider
boundary under `src/tensor`.

The first vertical slice initially kept the CMake build default on `scalar`. A matched endpoint
promotion audit was performed and **rejected default OpenBLAS promotion**: complete CPU DF
energy+force endpoints through a 96-dimensional metric did not improve. `openblas` and
discovery-based `auto` therefore remain explicit build choices.

It provides:
- row-major FP64 GEMM;
- lower-triangular FP64 Cholesky;
- a deterministic scalar fallback;
- optional OpenBLAS CBLAS/LAPACKE lowering;
- explicit task-parallel versus provider-parallel thread ownership;
- provider/capability diagnostics;
- CMake discovery through pkg-config or OpenBLAS CMake package metadata.

OpenBLAS is an execution provider, not part of TensorIR or method/scientific identity.
The first real consumer is the dense matrix algebra inside the symmetric matrix-function
VJP used by DF response. The independent SCF reference algebra in
`src/scf/reference/linalg.cpp` is intentionally unchanged.

Task-parallel mode requires single-thread provider execution and only auto-selects
OpenBLAS when a thread-local OpenBLAS thread setter is available. If a build exposes
only global thread control, automatic task-parallel execution stays on the scalar
fallback. Explicit provider-parallel OpenBLAS execution serializes global thread-setting
changes behind a mutex, restores the prior thread count, and allows the BLAS call to own
its requested worker count.

## Rejected alternatives

1. Call OpenBLAS directly from SCF/DF/post-HF code.
   Rejected because provider identity and threading would leak into scientific algorithms.

2. Replace the independent CPU reference linear algebra with OpenBLAS.
   Rejected because that would weaken an existing structurally independent oracle.

3. Use global `openblas_set_num_threads` from concurrent task-parallel calls.
   Rejected because it creates process-global races and nested-oversubscription hazards.

4. Require pkg-config for OpenBLAS discovery.
   Rejected because minimal hosts may have a usable OpenBLAS CMake package but no
   `pkg-config`; node3's SciPy OpenBLAS is one such configuration.

## Invariants

- Scientific code does not name OpenBLAS/CBLAS/LAPACKE.
- The no-external-BLAS build remains correct and testable.
- Task-parallel VibeQC execution never silently requests multiple BLAS workers.
- OpenBLAS without safe local thread control is not auto-promoted under task ownership.
- The independent SCF reference implementation remains separate.
- Provider promotion beyond this bounded consumer requires matched endpoint evidence.

## Evidence

Validated on node3, AMD EPYC 7K62 48-Core Processor, x86_64.

Scalar/no-BLAS build:
- `VIBEQC_ENABLE_CUDA=OFF`
- `VIBEQC_CPU_LINALG_PROVIDER=scalar`
- `vibeqc_cpu_linalg_tests`: pass
- `vibeqc_density_fitting_tests`: pass

OpenBLAS build:
- SciPy OpenBLAS 0.3.34 CMake package
- thread-local setter: unavailable in the shared library
- global runtime thread control: available
- LAPACKE dpotrf: available
- `vibeqc_cpu_linalg_tests`: pass through explicit provider-parallel OpenBLAS
- `vibeqc_density_fitting_tests`: pass

Matrix-function regressions:
- `tests/python/test_matrix_function.py`
- `tests/python/test_matrix_function_native_range.py`
- `tests/python/test_matrix_function_spectral_scaling.py`
- result: 60 passed, 3 skipped

Bounded GEMM probe, Release, three repeats, one provider thread:

| n | scalar GFLOP/s | OpenBLAS GFLOP/s |
|---:|---:|---:|
| 16 | 3.43 | 10.95 |
| 64 | 2.89 | 21.72 |
| 128 | 0.64 | 24.96 |
| 512 | 0.28 | 21.38 |

For n=512, explicit provider-parallel OpenBLAS with four requested provider threads measured
22.40 GFLOP/s in this allocation. These are kernel measurements only; they are not an SCF,
DF, or post-HF endpoint speedup claim.

## Consequences

The CPU runtime now has a stable place to add SYRK/TRSM/eigensolver operations and additional
providers such as MKL/BLIS without changing scientific formulas. Compiler scheduling can later
choose generated SIMD versus provider lowering using the #471 tuning machinery. Array API/TensorIR
work can target this boundary rather than one vendor API.

SciPy's OpenBLAS build on node3 lacks the thread-local setter symbol even though its header declares
it. The capability probe therefore matters: this build can be used explicitly in provider-parallel
mode, while automatic task-parallel execution remains conservative.

## Revisit when

- SYRK/TRSM/eigensolver consumers are migrated.
- Runtime CPU target/provider/workload identity is added to tuning/cache records.
- Matched SCF/DF/post-HF endpoint measurements justify broader default-path promotion.
- A system OpenBLAS build with thread-local control is available for concurrent task-parallel tests.

## References

- #674
- #349
- #351
- #471
- #633

---
Agent: ChatGPT
Model: GPT-5.6 Sol

## Endpoint promotion audit — 2026-09-20

After adding OpenBLAS `dsyevd`, the dense provider was tested on complete CPU
`Calculator(device="cpu", density_fitting="cpu")` energy+force endpoints. Measurements
were pinned to CPU 47 with `taskset`; `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.
Slurm isolation was attempted first, but the single-node partition queued the job for
priority, so the paired fixed-core measurements below are retained as the available
endpoint evidence.

The first all-OpenBLAS policy regressed the endpoints. Avoiding redundant global
thread-count set/restore calls reduced but did not remove the regression. Keeping the
DF metric eigensolve on the scalar schedule while allowing OpenBLAS for matrix-function
GEMMs still did not beat the scalar endpoint:

| endpoint | metric dimension | scalar median/best | OpenBLAS/mixed | result |
|---|---:|---:|---:|---:|
| water/def2-SVP, 5 repeats | 24 | 1.16695 s | 1.17540 s | 0.7% slower |
| water/def2-TZVP, 3 repeats | 43 | 17.2492 s | 17.6209 s | 2.2% slower |
| water tetramer/def2-SVP, 1 paired run | 96 | 111.191 s | 112.258 s | 1.0% slower |

All compared endpoints converged in the same number of SCF iterations. The final mixed
policy restored the scalar eigensolver and produced identical energies for the 24- and
43-dimensional cases to the printed precision; the 96-dimensional case also matched
energy and iteration count, with force norms agreeing to ~2e-11 absolute.

The isolated complete metric-factor + inverse-square-root-response consumer does show
a real dense-LA benefit. With one OpenBLAS thread and the scalar eigensolver retained,
representative best-of-five measurements were:

| metric dimension | scalar | OpenBLAS GEMM response |
|---:|---:|---:|
| 16 | 0.218 ms | 0.212 ms |
| 32 | 1.576 ms | 1.471 ms |
| 64 | 8.408 ms | 7.690 ms |
| 128 | 87.752 ms | 77.827 ms |

Therefore the production/build default remains `scalar`. Explicit OpenBLAS/`auto` builds
retain the provider, but the matrix-function consumer conservatively avoids the external
provider below dimension 128. Dimension 128 is an opt-in lower bound supported by local
consumer evidence, **not** a default-promotion claim; #471 should replace this static bound
with workload/endpoint-aware tuning before broad activation.

The OpenBLAS `dsyevd` capability is retained behind the provider API, but DF metric
factorization remains on the scalar eigensolver because the complete endpoint evidence did
not justify switching it.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Supersession

Superseded default-provider decision: the scalar-default conclusion above was valid before CPU DF J/K and derivative hot paths were migrated. The later endpoint promotion evidence in `../performance/2026-09-20-cpu-df-openblas-promotion.md` changes the build default to `auto` while preserving scalar fallback.

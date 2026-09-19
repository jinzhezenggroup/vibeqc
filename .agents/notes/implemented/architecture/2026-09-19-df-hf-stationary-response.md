# Decision: DF-HF stationary source weights and shared metric response

Status: implemented
Date: 2026-09-19

## Problem

The DF-HF force path already consumed generated coordinate derivatives from #143,
but the method-level source algebra was still encoded in native response adapters.
In particular, the A/M response ownership was not represented through #181's
`StationaryProblem`, and `src/scf/cuda/df_response_weights.cu` carried a local
fixed-rank spectral pseudoinverse Fréchet kernel even though #466 established a
method-neutral symmetric matrix-function custom-rule boundary.

That split made the mathematical ownership harder to audit and made the
retained/discarded spectral response easy to duplicate in later methods.

## Decision

Add `DensityFittingRHFResponsePlan` as the first concrete DF-HF
`StationaryProblem` slice. Its state is the full-rank fitted tensor `B` defined by

```text
M[p,q] B[q,i,j] - A[p,i,j] = 0
```

at a supplied stationary RHF density `D` and energy-weighted density `W`. The
declared objective is

```text
E = D:h - W:S
  + cJ/2 (D:A_p)(D:B_p)
  - cK A[p,i,j] D[k,i] B[p,k,l] D[l,j].
```

The shared stationary reverse therefore owns the h/S/A/M source weights and the
adjoint sign convention. The provider DAG names geometry pullbacks without
generating integral derivatives; #143 remains the owner of those derivatives.
Native runtime code still owns dynamic-size storage, tiling, cuBLAS/cuSOLVER,
streams, callbacks and lifetime management.

Extend `SymmetricMatrixFunctionSpec` with `function="pseudoinverse"` and the
fixed-rank divided-difference rule. Move the production CUDA pseudoinverse VJP
out of the DF method file into a compiler-generated
`generated_symmetric_matrix_function.cuh`, emitted by the versioned custom-rule
lowering in `matrix_function_cuda.py`. The CPU DF metric response likewise
delegates its spectral contraction to the shared
`src/tensor/symmetric_matrix_function.*` primitive after the DF owner has
validated the eigensystem, cutoff and active rank.

The full-rank stationary fitted-state formulation and the truncated
pseudoinverse custom rule are complementary: the former proves/derives method
source algebra without differentiating an eigensolver, while the latter
preserves the existing fixed-rank production branch including
retained/discarded subspace motion.

## Rejected alternatives

- Do not create a second `DFGradientIR` or a DF-specific autodiff engine.
- Do not generate cuBLAS calls, stream/event ownership, tiling, allocation or
  eigensolver runtime merely to reduce line count.
- Do not differentiate eigenvector gauges or replace the fixed-rank Fréchet
  rule with `-M+ E M+`; the latter drops finite discarded-subspace motion.
- Do not chain production response through an inverse-square-root rule only
  when the native consumer's cotangent is with respect to the pseudoinverse.
  Pseudoinverse is now an explicit custom-rule function.
- Do not move SCF iteration history into the derivative graph. The first slice
  is evaluated at the stationary RHF state.

## Invariants

- h and S weights are exactly `D` and `-W` under the full-Frobenius convention.
- A/M weights must match the pre-migration analytic DF-HF algebra and resolved
  nuclear/source finite differences.
- Rank-deficient metric response includes retained/discarded cross terms and
  rejects unresolved cutoff crossings before the custom rule executes.
- The generated CUDA custom-rule launch reuses the same two `a*a` scratch
  matrices and stream as the former local kernel. It adds no allocation,
  transfer, or synchronization to the DF force path.
- Generic native runtime/BLAS/eigensolver code remains native; method-local
  spectral arithmetic must not be reintroduced.
- UHF and multiple density-response terms may compose the same boundary later;
  this change does not claim that dynamic-shape stationary execution is fully
  generated in production.

## Evidence

Local gates on node3:

```text
PYTHONPATH=python:. pytest -q \
  tests/python/test_df_hf_stationary_response.py \
  tests/python/test_matrix_function.py
=> 60 passed

cmake --preset cuda-dev-fast
cmake --build build/cuda-dev-fast -j4
=> CPU/native configuration and build passed
   (node3 PATH did not expose a >=12.9 CUDA compiler)

ctest --test-dir build/cuda-dev-fast -R "density_fitting|native" --output-on-failure
=> 2/2 passed

/usr/local/cuda-12.4/bin/nvcc -std=c++20 -arch=sm_90 ... \
  -c src/scf/cuda/df_response_weights.cu
=> passed
```

The compiler test compares generated stationary h/S/A/M weights to the previous
closed-form RHF DF response algebra at FP64 roundoff and independently compares
all four source contractions to finite differences of the re-solved fitted
state. Matrix-function tests cover full-rank closed form, fixed-rank finite
differences, JVP/VJP duality and cross-subspace motion.

Ownership accounting against `origin/master` records 13,571 -> 13,495
handwritten scientific CUDA lines (net -76); `df_response_weights` is
1,103 -> 1,042. The runtime-sized spectral lowering is generated scientific
source rather than a reclassified handwritten CUDA file.

The repository CUDA 12.9 sm_120 compile gate and CuMetal CUDA suite remain the
authoritative CI checks for production CUDA compilation/runtime portability.

## Consequences

Method-equation ownership is now explicit in the compiler, while native dynamic
execution remains bounded and unchanged in work/data movement. The DF-specific
spectral kernel is retired; future method consumers should reuse the shared
custom rule. TensorIR still has static shapes, so directly executing the
stationary plan for arbitrary runtime nbf/naux is follow-up infrastructure work,
not a reason to restore native method equations.

## Revisit when

Revisit the native A/J/K BLAS adapters when TensorIR/custom-rule dispatch can
bind runtime-sized stationary plans without specializing every nbf/naux and can
match current resident/occupied/packed endpoint resource gates. Revisit the
full-rank fitted-state solver contract when a common bounded native adjoint
solver is available for direct plan execution.

## References

- #358
- #181
- #143
- #465
- #466
- `docs/stationary_problem.md`
- `docs/matrix_function.md`
- `docs/df_derivatives.md`

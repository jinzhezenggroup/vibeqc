# Decision: explicit typed FP32 TensorIR CUDA execution

Status: implemented
Date: 2026-09-19

## Problem

The TensorIR CPU interpreter and AD supported FP32, but the CUDA planner,
emitter, GEMM panels, host staging and resident spans assumed FP64. Removing
the planner's dtype rejection alone would reinterpret 4-byte values as doubles
and invalidate both numerical behavior and memory accounting.

## Decision

Use one small compiler-owned scalar lowering table for C++ types, exact
coefficient conversion, RN intrinsic names and literal suffixes. Reuse the same
TensorIR graph, execution planner, kernels and native Context owner. Each node
retains its declared dtype. Independent FP32 and FP64 graph components may
coexist, but operations on mixed input dtypes still fail; there are no implicit
casts or new mixed-precision algorithm/admission policies.

Float inputs, temporaries, outputs, packing panels, copies and resident spans
use 4 bytes. Doubles use 8 bytes. Integer gather tables remain int64 and all
segments retain 256-byte alignment. Fixed validation scratch remains FP64 and
is explicitly included in the existing host budget; validation arithmetic is
not a fallback for tensor computation. Float symmetry tolerances match the CPU
interpreter, with both host and device comparisons evaluated in FP64.

Elementwise kernels use explicit RN FP32 intrinsics, including __fdiv_rn rather
than approximate __fdividef. Generated scaled-bilinear derivatives reuse #481's
mantissa/exponent and compensated-product algorithm with frexpf, scalbnf,
__fmaf_rn and a 52-bit alignment window (versus 110 for FP64).

The direct and packed matrix routes dispatch SGEMM/SGEMMStridedBatched for
float, and retain DGEMM for double. FP32-containing GEMM plans select
CUBLAS_PEDANTIC_MATH at preparation. This intentionally establishes a strict
FP32 baseline rather than enabling TF32, FP16, emulated arithmetic or claiming
a tensor-core optimization. Compiler options require gradual underflow and
reject arithmetic-changing NVCC environment overrides for FP32 plans.

The ordinary ABI uses byte-oriented pointer tables; the binary pointer-table
layout is unchanged. The resident byte-span ABI stays at version 1. Dtype is
checked at each host boundary and reported on results and DeviceTensor leases.
Plan schema 2 records precision, per-step dtype/itemsize and arithmetic policy;
source/toolchain/options hashes distinguish newly compiled artifacts from old
ones. Resource identities report actual precision instead of hardcoded fp64.

## Rejected alternatives

- Allow float32 plans but cast all host inputs to double: not FP32 CUDA and no
  device memory benefit.
- Globally replace double/8 with float/4: corrupts int64 indices, FP64 paths,
  timing/validation state and independently typed components.
- Enable TF32/fast-math implicitly: changes the numerical contract without
  independent scientific qualification.
- Loosen automatic tuning's FP64 numerical gate to accommodate FP32: promotion
  is a separate task. The existing tuner explicitly rejects non-FP64 plans
  until consumer-appropriate acceptance gates are supplied.

## Validation

Tests cover planning, serialization/replay identities, panel/transfer/span byte
sizes, host dtype rejection, genuine FP32 rounding and subnormal controls,
SGEMM layouts and batches, packed tail tiles, general contractions, views,
fusion/recomputation, empty output/reduction domains, resident leases and
transfer counts, and the independent Fraction-based division-AD regressions.

Exact run outcomes and toolchain/device details are recorded in the PR. The
opt-in CUDA tests are skipped without an allocated device, never counted as
passes. No timing or performance-promotion claim is made.

## Boundaries

FP64 is still the default TensorSpec and the chemistry consumers' precision.
This does not qualify FP32 CCSD, Lambda, Z vectors, forces or Hessians. The
bounded-error arithmetic and higher-derivative boundaries documented for #481
remain in effect. Automatic schedule promotion remains FP64-only. Tensor-core
variants, mixed-dtype casts and lower-precision molecular promotion need
separate numerical and complete-endpoint evidence.

## References

- #482; parent #137; #146/#151 foundations; #477 and prerequisite PR #481.
- CUDA single-precision intrinsics:
  https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html
- cuBLAS math modes:
  https://docs.nvidia.com/cuda/cublas/index.html#cublasmath-t

Agent: ChatGPT
Model: GPT-6 Astra Pro

### Executed validation on the implementation branch

- TensorIR CPU suite: 185 passed, 60 opt-in CUDA cases skipped, 3.76 s.
- Compiler/CC consumers (`test_compiler_structure`, `test_cc_equations`,
  `test_cc_gpu_state`, `test_cc_provenance`): 55 passed, 3.20 s.
- Allocated Slurm job 10017 on RTX 5090, driver 580.95.05, sm_120,
  NVCC 12.9.86 with CUDA 12.4 cuBLAS headers/libraries: 51 passed across
  `test_tensor_cuda_fp32.py` and `test_tensor_scaled_division.py`, 117.69 s
  including compilation. 29 tests exercised the device; 22 were CPU tests.
- Compiler dependency check: 183 modules, zero errors. CUDA ownership inventory:
  183 files checked. Ruff, clang-format, whitespace and evidence checks passed.

These are correctness/regression runs, not performance benchmarks. The full
historical optional-device suite and other GPU architectures are not claimed
as tested by this change.

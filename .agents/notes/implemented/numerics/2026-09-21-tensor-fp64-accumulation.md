# Decision: qualified FP32 compute with FP64 reduction accumulation

Status: implemented
Date: 2026-09-21

## Problem

Issue #528 already represented storage, compute, and accumulation precision separately,
but ordinary TensorIR lowering rejected every request where compute and accumulation
dtypes differed. That left the important FP32-storage / FP32-compute / FP64-accumulate
schedule representable in metadata but not executable.

A wider accumulator cannot be approximated by an output cast: SGEMM still performs
its reduction in FP32. The backend therefore needs an explicit execution contract
whose numerical semantics, artifact identity, resource cost, and qualification scope
remain inspectable.

## Decision

Admit one deliberately narrow mixed-accumulation contract:

- storage dtype: FP32;
- compute dtype: FP32;
- accumulation dtype: FP64;
- operations: `reduce` and `einsum`;
- external qualification is mandatory.

All other compute/accumulation mismatches continue to fail closed.

The lowered scientific DAG retains explicit FP64-to-FP32 and FP32-to-FP64 casts.
A validated `precision_execution` binding connects the lowered reduction node to
the canonical precision-request identity. Precision-schedule schema v3 records the
source-to-lowered execution scope explicitly, then resolves the lowered node as FP32
storage/compute plus FP64 accumulation. TensorPlan, schedule-search, and artifact
identities therefore distinguish it from an otherwise identical all-FP32 equation.

Generated reduction semantics are:

1. operand arithmetic/products execute in FP32 round-to-nearest;
2. each reduced contribution is widened exactly to FP64;
3. the serial reduction uses FP64 round-to-nearest addition;
4. the accumulated value narrows once to FP32;
5. non-reduction node arithmetic such as an einsum's static coefficient remains FP32;
6. an explicit outer cast restores the public FP64 ABI when the source program had one.

Mixed-accumulation einsums are forced through the generated generic reduction kernel.
They do not silently use SGEMM. The static cost record reports the number of FP64
accumulation contributions, and the register heuristic charges the wider live
accumulator/conversion temporary. No extra global FP64 accumulation buffer is owned.

## Rejected alternatives

**SGEMM followed by an FP64 cast** was rejected because it preserves an FP32
reduction and therefore implements different mathematics.

**cuBLAS GemmEx with FP32 A/B, FP64 C, and `CUBLAS_COMPUTE_64F`** was tested
directly on the RTX 5090 and returned `CUBLAS_STATUS_NOT_SUPPORTED` (status 15)
with both CUDA 12.4 and CUDA 12.9.86. The generated reduction kernel is therefore
the qualified implementation for this contract on the tested toolchains.

**A general compute/accumulation cross-product** was rejected. It would advertise
untested contracts such as FP64 storage with FP32 compute or lower-precision
accumulators without independent evidence.

**Automatic promotion** was rejected. Compiler representability and execution do
not establish a method-level domain where mixed precision is profitable or safe.

## Invariants

- Strict FP64 remains the default/fallback.
- No implicit cast, TF32, tensor-core, or fast-math path is introduced.
- Low-precision reductions remain qualification-gated.
- Qualification/request identity is part of the resolved execution identity, not the
  scientific equation hash.
- Generated AD retains parent precision lineage but does not inherit primal numerical
  qualification.
- FP64 accumulation changes only reduction arithmetic; ordinary node compute remains
  FP32 for this contract.
- A mixed contraction must not be routed to SGEMM unless a future backend proves
  equivalent wider-accumulation semantics.

## Evidence

An independent reference case uses FP32-rounded
`[1e8, 1, -1e8, 1]`. Serial FP32 accumulation loses one contribution, while
widening every FP32 term into an FP64 accumulator yields exactly `2.0` before the
final FP32 storage cast. The CPU interpreter and generated source are checked against
that contract rather than only against one another.

Focused host validation after the implementation:

- 147 precision/planner/GEMM/search tests passed; 19 explicit GPU/environment skips.
- Compiler structure audit: 255 modules, 0 dependency errors.
- The dedicated RTX 5090 CUDA regression executes the mixed reduction and mixed
  einsum through CUDA 12.9 and passes the independent `2.0` oracle.
- The source/plan regression proves mixed einsum uses `gemm == "none"`, contains
  FP32 products, FP64 `__dadd_rn` accumulation, and one FP64-to-FP32 narrowing.

Exploratory complete `PreparedCuda.execute` measurements include host validation,
casts, transfers, kernels/libraries, error checks, and detached output allocation.
At 128x128, the first qualification run measured medians of 0.212 ms strict FP64,
0.200 ms full FP32, and 0.190 ms mixed. At 512x512, medians were 1.466 ms, 1.180 ms,
and 1.634 ms respectively. The mixed path reduced the planned device footprint
substantially because it does not instantiate the cuBLAS provider, but lost to DGEMM
at the larger shape. These exploratory results are not promotion evidence; the
checked-in benchmark runner interleaves variants and is the reproducible path for
follow-up measurements.

## Consequences

The compiler can now execute the central #528 mixed-accumulation schedule without a
method-specific tensor equation. Small contractions may benefit from avoiding library
provider overhead and memory, while GEMM-sized contractions can regress because the
current generic kernel gives up tuned library GEMM. Schedule search can retain either
result as positive or negative evidence instead of pretending all mixed schedules are
equivalent.

## Revisit when

Reconsider the lowering when cuBLAS/cuBLASLt supports an equivalent FP32-input,
FP64-accumulator contract on the target, when the generated contraction backend gains
a tiled mixed-accumulation kernel, or when #375 produces complete method-level
promotion evidence for a declared workload/device domain.

## References

- #528
- #375
- `docs/tensor_ir.md`
- `docs/tensor_cuda.md`
- `tools/tensor_precision_benchmark.py`
- `tests/python/test_tensor_precision.py`
- `tests/python/test_tensor_cuda_fp32.py`
- [previous precision identity/AD decision](2026-09-20-tensor-precision-identity-and-ad.md)

Agent: ChatGPT
Model: GPT-5.6 Sol

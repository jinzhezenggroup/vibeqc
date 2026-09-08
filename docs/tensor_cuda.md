# Prepared FP64 TensorIR CUDA plans

CG09 (#146) lowers the immutable [TensorIR](tensor_ir.md) into an executable
CUDA shared library. A prepared executor makes one native call for a complete
program. All tensor arithmetic runs on the allocated GPU; Python validates
and stages supplied arrays and returns detached results. These tools execute
supplied equations and do not implement a complete CCSD/MP2 method.

| Stage | Capability |
| --- | --- |
| CPU planning | Checked shapes, layouts, lifetimes, reservations and bounded GEMM tiles |
| Source and compilation | Whole-program CUDA generation through the existing finite NVCC adapter |
| Baseline execution | Unfused FP64, cuBLAS GEMM/strided-batched GEMM and generated primitive kernels |
| Candidate execution | View elimination, ordered elementwise fusion, smaller panels and root recomputation |
| Selection | Explicit bounded tuning, CPU/baseline parity, resource and complete-endpoint gates |
| Graph capture | Ordinary stream reported explicitly; capture is not implemented |
| Complete molecular CC solver | Outside this executor |

## Prepare and execute

Run from a source checkout with `PYTHONPATH=.:python`. Compilation does not
require a GPU. `PreparedCuda`, device probing and execution do; use the
site's scheduler and preserve its assigned device visibility.

```python
from pathlib import Path
from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_tensor.cuda_plan import plan_cuda, Reservations
from tools.vibeqc_tensor.cuda_execute import compile_cuda, PreparedCuda
from tools.vibeqc_tensor.examples import example_cases

case = example_cases()[1]
target = cuda_target_info("sm_120")  # choose the actual allocated architecture
compiler = CudaCompilerAdapter(Path("/path/to/nvcc"), target)
plan = plan_cuda(case.program, target, max_bytes=256 * 1024**2,
                 reservations=Reservations(t=4096, r=4096, diis=16384))
artifact = compile_cuda(plan, compiler, Path("build/tensor-cuda-cache"))
with PreparedCuda(plan, artifact, device=0) as prepared:
    first = prepared.execute(case.inputs)
    second = prepared.execute(case.inputs)
    print(second.outputs, second.metrics, prepared.graph_status)
```

Input arrays must be FP64 NumPy ndarrays of the declared shape. Negative,
noncontiguous, read-only and overlapping caller layouts are accepted through
prepared C-order staging. Missing/wrong inputs, nonfinite values, declared
symmetry violations, zero division and nonfinite intermediates fail explicitly.
Unused feeds are allowed. Each named result owns independent host storage,
including when two output names refer to the same logical tensor.

A prepared plan owns its stream, cuBLAS handle, events and buffers. It does not
assume that the existing `ContextState` owns an SCF stream. Execution requires
the thread's current CUDA device to match the prepared device. Architecture
mismatches fail before launching an incompatible kernel. Calls on one object
serialize, while independent objects can execute from independent threads.
Cleanup and error recovery drain queued transfers before returning control.

`PreparedTensorBatch(plans, compiler, cache, max_bytes=...)` groups identical
plan identities into compiled-code buckets and creates independent prepared
states. Its budget charges the sum of every plan's peak, including sequential
execution because all states remain resident. `execute(feeds, workers=...)`
preserves system order and supports different nocc/nvir populations.

## Layouts and contractions

Materialized values use logical C-order strides. Eligible binary einsums group
declared labels into batch, M, N and K populations. No spin, orbital or symmetry
relationship is inferred from matching extents. The direct path recognizes
contiguous matrices in either transpose orientation and batch-prefix layouts.
It emits explicit transpose flags, leading dimensions and batch strides and
uses `cublasDgemmStridedBatched` when the batch count exceeds one.

Other eligible layouts use one reused A/B/C panel set. Generated packing
kernels gather through logical view maps; cuBLAS evaluates each packed tile;
generated scatter kernels place the unique output entries in logical order.
The first K tile uses beta zero and later tiles use beta one. Exact rational
factors round once to FP64 and are applied after validating the contraction's
unscaled result. This preserves overflow detection even for a zero factor.

Repeated labels/diagonals, one-sided reductions and n-ary einsums use a
conservative generated kernel. Each output thread evaluates its complete
reduction in deterministic order. This fallback supplies correctness for
small irregular contractions; it is not a substitute for cuBLAS on regular
large GEMMs. Empty output domains launch no kernel and empty reduction domains
produce zero. Index expressions remain compilable for zero extents.

Generated kernels cover ordered addition, products, division/denominators,
transpose, logical reshape, slice, gather (including repeated coordinates),
reduction and explicit broadcast. Packing scatter assigns unique destinations;
there is no new mathematical scatter-add primitive or AD implementation.

## Budgets, lifetimes and limitations

`plan_cuda` checks signed-64-bit byte/stride/reduction products and cuBLAS
integer dimensions before preparation. Its budget covers the following numeric
storage, with 256-byte alignment where needed:

- Resident device inputs and constants, named outputs and gather index tables.
- Reusable intermediate arena slots and their last-use intervals.
- The maximum simultaneously live A/B/C packing panels.
- The explicit cuBLAS workspace (default 4 MiB, configurable down to zero).
- A separate cuBLAS retained-device allowance (default/minimum 96 MiB).
- Physically retained T/R/DIIS/concurrent-work reservations and the device error flag.
- Prepared host input staging, bounded validation scratch and one detached output set.

The provider allowance accounts for internal retained allocations even with a
zero explicit workspace. The audited cuBLAS provider allocates a 64 MiB buffer
plus smaller buffers during handle creation; installing a user workspace does
not release them. `provider_bytes` can increase the conservative allowance.
Preparation checks the device-memory delta around handle creation before
allocating tensor buffers and rolls back if that provider exceeds its allowance.
This provider probe can fail after creating its temporary handle; it does not
establish a portable promise for every future library version. Owned handle
creation/destruction are serialized so they cannot distort each other's check.

The executor binds one ordinary stream and installs its user workspace after
the final cuBLAS stream binding. There are no concurrent per-plan streams or
double buffers; their capacities are explicitly zero. Batch preparation sums
all resident plan peaks. Caller reservations describe additional retained work
and are not silently recycled as intermediate storage.

The combined budget is **not a process-RSS or total-free-VRAM guarantee**.
Caller inputs/previous results, Python metadata, generated code/constants in
the host library, general CUDA context/module/stack storage, provider host
metadata and allocator page rounding are outside the numeric-buffer scope.
Both the explicit workspace and retained-device allowance are included.
`owned_device_bytes` records the successful tensor allocation and
`provider_retained_bytes` records its provider check. `allocation_bytes` is the
tensor allocation capacity; `device_bytes` also includes the provider allowance.
`prepare_device_delta` and
`observed_device_delta` separately report device-wide free-memory differences,
which may include another concurrently prepared context. Initial cuBLAS/JIT
overhead can greatly exceed a small tensor's numeric storage and startup time.
The overall device-wide deltas are observations, not portable upper bounds on
general runtime overhead. Successful preparations must fit their counted
provider allowance as well as their statically planned tensor capacities.

Inputs and outputs are indivisible resident tensors. When fixed storage plus
the minimum panel capacity exceeds the budget, planning fails on the CPU;
it does not promise arbitrary out-of-core execution. Otherwise packing tiles
shrink deterministically until they fit. Partial M/N/K tiles use their actual
sizes and reuse the same panel allocations.

The baseline shares existing SSA intermediates. Last-use analysis follows
virtual views to their materialized ancestors and never overwrites a live
alias or named output. Best-fit free intervals reuse dead storage. A separate
recomputation candidate evaluates output roots sequentially and duplicates
shared intermediates across roots to reduce retention; its additional work,
traffic estimate and peak capacity are exposed in `TensorPlan.to_payload()`.

## Candidate selection and identity

`TensorSchedule()` is the deterministic unfused baseline. Explicit candidate
switches enable view elimination, small elementwise fusion and root
recomputation. Fusion preserves arithmetic order and each intermediate's
finite/zero-division check. It is restricted to complete same-domain
elementwise consumers; a slice cannot hide an overflow by discarding entries.
View chains have a bounded inline depth. These switches are experimental
requests until a particular plan passes tuning gates.

`tune_cuda(baseline, compiler, fixtures, cache, ...)` considers at most eight
candidates and eight fixed-shape fixtures, with 5–30 paired repeats. It warms
the library first, records startup separately, and reuses the shared CG01
interleaved measurement and noise assessment. Timings include input validation,
host layout staging, transfers, packing, cuBLAS, all small kernels, result
allocation and error checks. Optional section profiling synchronizes sections
and is recorded separately; those samples never decide the winner.

Every candidate must match the CPU interpreter and unfused GPU on all outputs,
pass target register/stack/shared-memory limits with no spills, and beat the
baseline on every supplied scale/layout fixture. Both the shared >2%/noise
gate and a paired bootstrap lower bound apply. An unsuccessful search retains
the baseline and preserves all candidate failures/raw measurements. Tuning is
explicit; installation never launches a search. The returned `TensorSelection`
contains the concrete compiled winner and evidence path, with no global dispatch
change or claim about unmeasured shapes.

The local artifact cache verifies each binary hash before loading. Identity
includes equation/spec/layout/shape/precision, plan and schema, all tensor
runtime/codegen/selection sources, target limits, NVCC/PTXAS and host compiler,
compiler flags/environment and host platform. Selection additionally keys the
actual GPU UUID, driver/runtime/cuBLAS, Python/NumPy and the complete measured
fixture bytes/strides. Tensor source identity reuses the CG07 hash/publication
contracts and inventories tensor code separately from shell-class profiles.
Only immutable native code is shared; context pointers and outputs are never
cached. The cache is for locally trusted generated code, not downloaded binaries.

## Validation and reproducible evidence

```bash
PYTHONPATH=.:python python -m pytest tests/python/test_tensor_cuda_plan.py \
  tests/python/test_tensor_cuda_gemm.py tests/python/test_tensor_cuda_tune.py -q

PYTHONPATH=.:python python tools/tensor_cuda_examples.py --mode compile \
  --nvcc /path/to/nvcc --architecture sm_120 --output /tmp/tensor-compile

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH=.:python VIBEQC_TENSOR_CUDA_TEST=1 VIBEQC_NVCC=/path/to/nvcc \
  python -m pytest tests/python/test_tensor_cuda_execution.py -q

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  env PYTHONPATH=.:python python tools/tensor_cuda_examples.py --mode tune \
  --nvcc /path/to/nvcc --architecture sm_120 --output /tmp/tensor-endpoints
```

The runner exports equations, seeds, shape/scale identities, shared CG01
records and complete tuning ledgers. Numerical gates remain
`atol=1e-11, rtol=1e-10`. It separately executes constrained panels with zero
explicit library workspace and forced recomputation. Pytest additionally
checks every intermediate of the independent-loop examples, all transpose
flags, batched/noncontiguous layouts, general contractions, unit tiles, odd
dimensions, empty domains, error recovery, architecture mismatches, independent
contexts, shape buckets and detached outputs. Real-device tests skip in CPU CI
unless explicitly enabled by the caller's GPU allocation.

The [archived CG09 records](../benchmarks/results/tensor-cuda-146/README.md)
include the retained-provider allocation audit, all candidate samples and the
measured selection/fallback results for six shape buckets.

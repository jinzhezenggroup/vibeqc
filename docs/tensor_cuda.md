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
| Candidate execution | Producer/consumer layouts, view elimination, ordered elementwise fusion, smaller panels and root recomputation |
| Selection | Explicit bounded tuning, CPU/baseline parity, resource and complete-endpoint gates |
| Graph capture | Opt-in fixed-region replay with ordinary fallback |
| Complete molecular CC solver | Outside this executor |

## Prepare and execute

Run from a source checkout with `PYTHONPATH=.:python`. Compilation does not
require a GPU. `PreparedCuda`, device probing and execution do; use the
site's scheduler and preserve its assigned device visibility.

```python
from pathlib import Path
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor.cuda_plan import plan_cuda, Reservations
from vibeqc_compiler.tensor.cuda_execute import compile_cuda, PreparedCuda
from vibeqc_compiler.tensor.examples import example_cases

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

## Optional compiled-region replay

`PreparedCuda(plan, artifact, execution_mode="cuda-graph")` opts into shared
compiler/runtime capture. The default remains `execution_mode="ordinary"`.
The first eligible unprofiled call executes ordinarily; the second captures,
instantiates and executes the fixed device region once; later calls replay.
Numeric inputs refresh the existing device buffers without recapture. Validation,
host transfers, detached outputs and arithmetic-error checks remain outside
capture. Each complete endpoint still makes one native call.

```python
with PreparedCuda(plan, artifact, execution_mode="cuda-graph") as prepared:
    prepared.execute(feeds)              # ordinary warmup
    prepared.execute(feeds)              # capture, instantiate, execute
    result = prepared.execute(new_feeds) # replay with refreshed numeric inputs
    print(result.backend, result.metrics["graph_captures"])
    prepared.invalidate_graph()          # next call warms up before recapture
```

The pure CaptureContract reuses specialization records and existing artifact
keys. Identity includes program/compiler/artifact, schedule, shapes/layouts and
precision, GPU UUID and driver/CUDA/cuBLAS versions. Native binding additionally
includes the stream, arena and library-handle addresses. A different plan,
schedule, artifact or device requires a new prepared owner. Invalidation discards
only replay state. Graphs are destroyed before their referenced resources.

Unsupported capture/instantiation and regions exceeding 4096 launches/nodes fall
back to ordinary execution. Failed capture is not retried until invalidation.
Graph-launch or asynchronous execution errors propagate, avoiding duplicate
execution of possibly submitted work. Zero division and nonfinite intermediate
checks remain active. `profile=True` uses ordinary section profiling and leaves
an existing graph reusable. No convergence or DIIS policy changes.

Graph metrics report mode/reason, cumulative capture/replay/fallback counts,
capture/instantiate CPU time, region submission time and a retained device-memory
delta. `diagnostics=True` adds this telemetry to an ordinary call without section
fences. `device_ms` is the event interval around transfers and the device region,
including host-submission gaps, not isolated kernel time. Unprofiled execution
retains one explicit completion fence; pageable transfers may synchronize inside
the CUDA runtime.

Graph storage is outside the exact numeric-buffer budget. Its free-memory delta
is an observation, not an allocation bound; driver host storage is unmeasured.
Global ResourcePlan consumers therefore use ordinary fallback until graph storage
has a budgeted candidate. The resident extension remains ordinary. No automatic
profile promotion or full SCF/CC qualification is implied.

The shared owner is `src/runtime/cuda_graph_region.cuh`. Existing device-tail-launch
SCF/eigensolver experiments use a different control abstraction; ProgramIR and
iterative electronic-structure adoption remain follow-on #460/#370/#507 work.
See the [decision note](../.agents/notes/implemented/architecture/2026-09-19-compiled-cuda-replay.md).

## Layouts and contractions

The baseline materializes values with logical C-order strides. Eligible binary einsums group
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
generated derivative programs from #151 use incidence-matrix einsum nodes
rather than a new mathematical scatter-add primitive.

Generated JVP/VJP programs from #151 are ordinary TensorIR programs and use
the same planning, compilation and execution path. The fixed CC-like RTX 5090
numerical/resource evidence is recorded in
[`benchmarks/results/tensor-ad-151`](../benchmarks/results/tensor-ad-151/README.md).

### Opt-in producer/consumer layouts

`TensorSchedule(layouts=True)` enables a bounded layout pass inside one TensorIR
program. `DenseLayout` records logical shape, a slow-to-fast permutation of logical
axis ordinals, physical element strides and base-address alignment. It is separate
from `TensorSpec`: neither equation serialization nor logical/scientific hashes
change. The descriptor includes transpose-view equivalence, but does not grant
alias permissions or encode arbitrary padded/affine storage.

For each eligible GEMM, the pass considers the two orientations of each operand
and its grouped output order **jointly**, then costs the effects on all consumers
and on the producers themselves. This avoids requiring two individually useful
changes when both operands must change before any packing can disappear. Shared
consumers are costed together; no duplicate tensor or persistent packed copy is
silently created. There are at most 256 trials and four reverse-order sweeps per
tile configuration. Strict cost improvements alone are accepted; ties retain the
existing layout, and an exhausted search retains its legal best-so-far plan.

Generated primitive kernels write contiguous physical elements while evaluating
the corresponding logical coordinates. All generated reads and packed scatters
honor the selected mapping. A GEMM may write an internal result in its natural
matrix order even when the equation names another output order; downstream reads
observe the same logical tensor. Direct NN/NT/TN/TT and grouped/batched eligibility
are checked against **physical** axis order rather than assuming C-order.

Inputs, constants and every named output remain pinned in logical C-order, so
ordinary transfers and resident ABI spans do not change. Existing virtual-view,
fusion and recomputation contracts remain intact. A virtual operand has no direct
buffer descriptor and retains generated packing; this slice does not infer affine
aliases through virtual views, create shared pack-once buffers, or propagate
layouts across ProgramIR subsystems. Those remain follow-up work for #509/#460.

`plan.layout_decision` reports attempted trials, changed steps and semantic panel
conversion bytes, including repeated A packing across N tiles and B packing
across M tiles. Its cost adds one logical tensor-read/write equivalent for each
non-C generic access as a conservative stride penalty. This **cost score is not
measured DRAM traffic or an additional memory allocation**. The pre-existing
`estimated_traffic_bytes` remains a coarse logical node-traffic estimate; the
separate conversion-byte field makes the additional packing work explicit.
Layout selection is rerun whenever admission shrinks packing tiles, so costs and
panel capacities describe the final schedule. Dense permutations do not enlarge
arena slots or alter lifetimes; eliminated panels reduce the charged peak.

Plan schema 2 serializes physical descriptors and the decision. The separate
`plan.layout_identity` can be supplied as an exact `layout_identity` workload
fact to #459 specialization guards; it must not replace scientific, compiler or
full-plan identity checks. Existing artifact keys and native identity checks
already include the complete plan, preventing reuse across different layouts.
Generated JVP/VJP programs use the same pass without new derivative rules.

This switch remains opt-in. `tune_cuda` includes layout-only and layout/view/fusion
candidates and still requires parity, resource and complete-endpoint gates before
selecting them for the supplied fixture domain. A static byte reduction is not a
performance-promotion decision.

A focused qualification runner retains local raw paired timings and identities:

```bash
# Execute within the site's finite GPU allocation.
PYTHONPATH=python python tools/tensor_layout_benchmark.py \
  --nvcc /path/to/nvcc --architecture sm_120 \
  --shape 33 7 65 31 --repeats 10 --output .artifacts/tensor-layout-small
# Repeat with --shape 65 9 97 63 to exercise a larger domain.
```

This measures a complete host-staged tensor contraction endpoint, not a complete
molecular CCSD/DFT calculation. The design rationale and retained qualification
results are in the [producer-layout note](../.agents/notes/implemented/performance/2026-09-19-tensor-producer-layouts.md).

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
switches enable producer layouts, view elimination, small elementwise fusion and
root recomputation. Fusion preserves arithmetic order and each intermediate's
finite/zero-division check. It is restricted to complete same-domain
elementwise consumers; a slice cannot hide an overflow by discarding entries.
View chains have a bounded inline depth. These switches are experimental
requests until a particular plan passes tuning gates.

`tune_cuda(baseline, compiler, fixtures, cache, ...)` uses a structured
`TensorScheduleSpace`: view elimination, fusion, recomputation, direct/packed
GEMM, block threads and M/N/K panel dimensions. Its deterministic bounded walk
visits single-axis changes before higher-order interactions without enumerating
the full Cartesian product. Only implemented ordinary-stream dimensions are
searched; vectorized reductions, cooperative/persistent kernels and shared-memory
GEMM staging are not implied by these controls.

`TensorSearchLimits` defaults to 128 generated candidates and 12 candidate
compilation attempts, plus the mandatory baseline. Explicit `schedules=` remains
supported; it is mutually exclusive with `search_space=`. The static pipeline
checks planner legality/combined host-device memory, removes equivalent execution
plans (including ineffective direct-GEMM tile changes), then applies source-size,
register-pressure and occupancy policies before invoking NVCC. The same plan's
aliases, lifetimes, outputs, reservations and allocation capacities participate
in equivalence checking. `maximum_source_bytes` bounds generated source size;
it is a compile-cost proxy, not a prediction of compilation seconds.

Static register counts are scalar-liveness heuristics for generated kernels and
occupancy is an upper bound without register-allocation granularity. Neither
models cuBLAS internals. Packing panels remain global numeric buffers, not shared
memory. Logical traffic excludes packing/provider/cache traffic; local memory is
unknown until PTXAS reports it. These qualifications are retained in the evidence.
Compiled candidates still require complete PTXAS register/stack/spill/shared data
and feasible per-block resources before any candidate endpoint execution.

Tuning supports eight fixed-shape fixtures, with 5–30 paired repeats. It warms
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
change or claim about unmeasured shapes. Schema-v3 tuning evidence retains
static pruning/deadline/budget reasons and actual compilation-attempt counts.
Artifacts record generated-source and binary sizes alongside compilation seconds
and PTXAS resources.

`TensorScreeningPolicy` adds a ranking-only representative-timing stage after
compiled-resource checks. Its default uses fixture 0, five interleaved A/B pairs,
and at most three full-qualification finalists. Select other existing fixtures
with `fixture_indices=(...)`; the worst median baseline/candidate ratio across
those representatives determines ordering. Exact ties preserve schedule generation
order. No speed threshold or noisy screening observation grants a performance
qualification. Rejected screens retain their negative evidence.

The screen is bypassed when the available compilation budget already fits the
finalist limit, or when its planned A/B-pair count plus finalist qualification
would not reduce measurement work. `screening_plan` records that decision and
counts, explicitly excluding compilation, startup and profiling costs. A reduced
pair count is not a measured wall-time benefit. `screening=None` restores full
qualification of every compiled candidate.

Only baseline and one candidate context remain resident at once. Screening
contexts are closed before a finalist reopens its existing artifact; finalists
are not recompiled. Every finalist repeats warmup and obtains **fresh** full
endpoint samples on **all** supplied fixtures, including the representatives.
The original CPU/unfused-GPU parity, shared noise and bootstrap gates still apply.
Pruned screens get no performance guard. A later numerical failure, slow fixture
or exhausted deadline cannot be overridden by a fast screen, and all-failure
searches retain the baseline. Policy and representative-fixture identities
participate in the selection key, without changing mathematical or CUDA artifact
identity recipes. Non-finite timing observations are rejected and recorded as
JSON `null` with an `invalid_seconds` description rather than invalid JSON NaN/Inf.
See the [screening decision note](../.agents/notes/implemented/performance/2026-09-19-tensor-screening-shortlist.md)
for the sampling tradeoff and validation boundary.

Accepted rows and the selected winner carry shared #459 `ImplementationProfile`
records, with the original artifact key, schedule hash and candidate evidence
hash. Correctness guards bind the equation, numeric-buffer budget, reservations
and target; performance guards additionally restrict promotion to each measured
input shape/dtype/stride domain and the existing baseline execution identity
(GPU UUID, driver/runtime/libraries and Python/NumPy). Values stay in measurement
provenance, not used as a benchmark-ID dispatch policy. Unmeasured layouts cannot satisfy the
performance guard; missing target facts fail closed. These records live in the
existing selection evidence, not a second profile database or automatic loader.
The current API returns a concrete plan for the caller's measured workload; it
does not install new global dispatch policy. See the
[search rationale](../.agents/notes/implemented/performance/2026-09-19-tensor-schedule-search.md).

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

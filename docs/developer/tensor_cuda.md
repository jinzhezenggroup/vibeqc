# Prepared typed TensorIR CUDA plans

CG09 (#146) lowers the immutable [TensorIR](tensor_ir.md) into an executable
CUDA shared library. A prepared executor makes one native call for a complete
program. All tensor arithmetic runs on the allocated GPU; Python validates
and stages supplied arrays and returns detached results. These tools execute
supplied equations and do not implement a complete CCSD/MP2 method.

| Stage | Capability |
| --- | --- |
| CPU planning | Checked shapes, layouts, lifetimes, reservations and bounded GEMM tiles |
| Source and compilation | Whole-program CUDA generation through the existing finite NVCC adapter |
| Baseline execution | Unfused strict FP64, cuBLAS GEMM/strided-batched GEMM and generated primitive kernels |
| Candidate execution | Producer/consumer layouts, view elimination, ordered elementwise fusion, smaller panels, root recomputation, opt-in typed precision variants and opt-in CUB cooperative reductions |
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
plan = plan_cuda(
    case.program,
    target,
    max_bytes=256 * 1024**2,
    reservations=Reservations(t=4096, r=4096, diis=16384),
)
artifact = compile_cuda(plan, compiler, Path("build/tensor-cuda-cache"))
with PreparedCuda(plan, artifact, device=0) as prepared:
    first = prepared.execute(case.inputs)
    second = prepared.execute(case.inputs)
    print(second.outputs, second.metrics, prepared.graph_status)
```

Input arrays must be NumPy ndarrays of the declared shape and declared
`float32`/`float64` dtype. Negative, noncontiguous, read-only and overlapping
caller layouts are accepted through
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
    prepared.execute(feeds)  # ordinary warmup
    prepared.execute(feeds)  # capture, instantiate, execute
    result = prepared.execute(new_feeds)  # replay with refreshed numeric inputs
    print(result.backend, result.metrics["graph_captures"])
    prepared.invalidate_graph()  # next call warms up before recapture
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
See the [decision note](../../.agents/notes/implemented/architecture/2026-09-19-compiled-cuda-replay.md).

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

`TensorSchedule(stream_reductions=True, streamed_gemm_reduction=True)` is an
explicit memory-oriented candidate for einsums that consume streamed producer
values. It evaluates those contractions with the existing generated reduction
kernel instead of repacking the virtual producer for every GEMM panel. It uses
the same TensorIR equation, runtime index maps, finite checks, and existing
plan/artifact identity machinery. The ordinary schedule and earlier streaming
schedule remain available. The candidate can recompute expensive virtual
producers, so it is absent from the default search space. Complete endpoint
qualification is required for each workload before use as a performance choice.

Generated kernels cover ordered addition, products, division/denominators,
transpose, logical reshape, slice, gather (including repeated coordinates),
reduction, explicit broadcast, and the ragged `indexed_gather`, `scatter_add`
and `segment_sum` primitives. Packing scatter still assigns unique GEMM
destinations; TensorIR `scatter_add` instead has explicit repeated-destination
accumulation semantics.

Eligible streamed reductions may explicitly request
`TensorSchedule(reduction_provider="cub")`. The shared lowering-provider
diagnostic advertises both generated cooperative CUDA and CUB BlockReduce for
the same request, but CUB admission requires typed-true CCCL header capability
evidence. Generated CUDA remains the production default. The retained H200 FP64
comparison passed numerical and execution gates but did not show a significant
complete-endpoint improvement, so lower CUB register use did not trigger a
promotion. See the [reviewed evidence](../../benchmarks/results/issue971-cub-block-reduce-h200/publication.json)
and [qualification decision](../../.agents/notes/implemented/performance/2026-09-22-cub-block-reduce-qualification.md).

`plan.batch_schedule` exposes a backend-neutral `BatchScheduleIR` with batch
domains, degree histograms, and baseline-versus-scheduled ragged work. Static
`scatter_add` maps are compiled into deterministic offsets-plus-members tables,
so each output traverses only its members instead of scanning the full source
axis. Members retain ascending source order, preserving the previous reduction
order. `segment_sum` keeps its already-linear contiguous-offset lowering and
`indexed_gather` remains a direct indexed load. This first scheduling slice
does not yet add degree-bucketed warp/CTA queues or runtime-varying topology.

Generated JVP/VJP programs from #151/#501 are ordinary TensorIR programs and use
the same planning, compilation and execution path. The fixed CC-like RTX 5090
numerical/resource evidence is recorded in
[`benchmarks/results/tensor-ad-151`](../../benchmarks/results/tensor-ad-151/README.md).

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

Plan schema 4 serializes physical descriptors, the layout decision, the
semantic-traffic record, and the resolved precision schedule. The separate
`plan.layout_identity` can be supplied as an exact `layout_identity` workload
fact to #459 specialization guards; it must not replace scientific, compiler or
full-plan identity checks. Existing artifact keys and native identity checks
already include the complete plan, preventing reuse across different layouts or
precision schedules. Generated JVP/VJP programs use the same pass without new
derivative rules.

### Opt-in precision schedules

Issue #528 makes precision an explicit typed scheduling dimension rather than a
global mixed-precision flag. `cast` nodes are materialized SSA boundaries;
FP64-to-FP32 lowering emits round-to-nearest conversion and FP32-to-FP64
conversion is explicit. There are no implicit dtype promotions in TensorIR
arithmetic, and the strict CUDA arithmetic identity remains
`ieee-rn-no-tf32`.

`PrecisionDirective` records requested storage, compute and accumulation
dtypes. `lower_precision` converts supported requests into ordinary TensorIR
DAGs with explicit casts while preserving the external input/output ABI and the
source-equation identity. Qualified `reduce` and `einsum` values may use FP32
storage/compute with FP64 accumulation; the
wider accumulator is represented in the resolved precision schedule and plan
identity. Other compute/accumulation mismatches fail closed. Mixed-accumulation
einsums deliberately use the generated generic contraction kernel because the
current cuBLAS path does not provide this FP32-input/FP64-accumulator contract.
FP32 requests for reductions/contractions or sensitive divide/transcendental
operations still require qualification provenance; the conservative generated
candidate leaves those boundaries in FP64. The numerical/backend rationale and
qualification boundary are recorded in the
[FP64 accumulation decision](../../.agents/notes/implemented/numerics/2026-09-21-tensor-fp64-accumulation.md).

Precision candidates are passed to the existing #508 schedule search and tuner.
They share the same planning budgets, deduplication, compilation cache, resource
gates, strict-FP64 reference parity and complete-endpoint promotion evidence.
No precision candidate is searched unless the caller opts in; the default
baseline and fallback remain strict FP64. Cast read/write traffic and
simultaneous source/target storage are recorded in the plan/static cost model,
and the plan identity includes the resolved precision schedule, strict audit
dtype and arithmetic mode. Method-level strict refinement/audit remains the
method controller's responsibility rather than a compiler-side convergence
shortcut.

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
results are in the [producer-layout note](../../.agents/notes/implemented/performance/2026-09-19-tensor-producer-layouts.md).

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
GEMM, block threads, generic-kernel elements per thread, serial-reduction unroll,
packed-GEMM staging width and M/N/K panel dimensions. Its deterministic bounded
walk visits single-axis changes before higher-order interactions without
enumerating the full Cartesian product. The added execution knobs are real
emitter choices: generic kernels can process 2/4 logical elements per thread,
reduce/einsum scalar loops can carry explicit 2/4-way unroll directives, and
global pack/scatter kernels can process 2/4 elements per thread. Cooperative or
persistent reductions and shared-memory GEMM staging are still not implied.

`TensorSearchLimits` defaults to 256 generated candidates and 12 candidate
compilation attempts, plus the mandatory baseline. The default tile axes include
32/64/128/256/512 choices. Explicit `schedules=` remains supported; it is
mutually exclusive with `search_space=`. The static pipeline checks planner
legality/combined host-device memory, removes equivalent execution plans
(including ineffective direct-GEMM tile changes), then applies source-size,
register-pressure and occupancy policies before invoking NVCC. Ready plans are
ranked for the finite compilation budget by estimated endpoint semantic traffic,
then register pressure, generated-source bytes and generation order. This
transparent ranking only decides which candidates deserve compilation; it cannot
promote a schedule. The same plan's aliases, lifetimes, outputs, reservations and
allocation capacities participate in equivalence checking. `maximum_source_bytes`
bounds generated source size; it is a compile-cost proxy that is calibrated
against compiler-reported duration after compilation rather than treated as a
predicted wall time.

Static register counts combine scalar liveness with bounded work-per-thread,
reduction-unroll and staging-width pressure; occupancy remains an upper bound
without register-allocation granularity. Neither models cuBLAS internals. Packing
panels remain global numeric buffers, not shared memory. The plan now carries a
deterministic semantic-traffic record separating planner logical bytes, exact
pack/scatter conversion bytes and declared H2D/D2H endpoint copies. It explicitly
excludes hardware cache/DRAM transactions and cuBLAS internal workspace/traffic.
Successful endpoints repeat those plan-bound byte counts as observed execution
metadata; they are not presented as hardware counters.

Compiled candidates still require complete PTXAS register/stack/spill/shared data
and feasible per-block resources before any candidate endpoint execution. PTXAS
`lmem` is retained when the toolchain reports it; stack and spill evidence remain
separate. Each compiled candidate also stores a resource calibration record
(static register estimate versus PTXAS registers plus compiled stack/spill/shared/
local facts) and a compile calibration record relating generated-source bytes to
compiler-reported duration. These calibrations are audit evidence, not promotion
shortcuts.

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
and PTXAS resources. Candidate rows additionally retain compile/resource
calibration and profiled endpoint traffic metadata, so winning evidence contains
the static estimate, compiler facts and successfully executed plan facts in one
ledger.

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
See the [screening decision note](../../.agents/notes/implemented/performance/2026-09-19-tensor-screening-shortlist.md)
for the sampling tradeoff and validation boundary.

Accepted rows and the selected winner carry shared #459 `ImplementationProfile`
records, with the original artifact key, schedule hash and candidate evidence
hash. Correctness guards bind the equation, numeric-buffer budget, reservations
and target; performance guards additionally restrict promotion to each measured
input shape/dtype/stride domain and the existing baseline execution identity
(GPU UUID, driver/runtime/libraries and Python/NumPy). Values stay in measurement
provenance, not used as a benchmark-ID dispatch policy. Unmeasured layouts cannot
satisfy the performance guard; missing target facts fail closed. These records live in the
existing selection evidence, not a second profile database or automatic loader.
The current API returns a concrete plan for the caller's measured workload; it
does not install new global dispatch policy. See the
[search rationale](../../.agents/notes/implemented/performance/2026-09-19-tensor-schedule-search.md)
and the [resource-aware completion note](../../.agents/notes/implemented/performance/2026-09-20-tensor-resource-aware-scheduler.md).

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

PYTHONPATH=python python benchmarks/tensor_cub_qualification.py \
  --nvcc /path/to/nvcc --architecture sm_90 --allocation-id <finite-job-id> \
  --source-revision <full-commit-sha> --source-archive <verified-archive> \
  --source-archive-sha256 <sha256> --output .artifacts/benchmarks/cub-h200
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

The [archived CG09 records](../../benchmarks/results/tensor-cuda-146/README.md)
retain the provider-allocation audit and measured selection/fallback results for
six shape buckets. The complete candidate tuning ledgers and raw samples are
hash-pinned in existing Git history and can be restored through the checked-in
retention manifest.

Precision-request identity and qualification scope are part of the resolved schedule,
not the source equation hash. Cast AD uses the declared arithmetic linearization
rather than the derivative of bit-level rounding; see the
[precision identity and AD contract](../../.agents/notes/implemented/numerics/2026-09-20-tensor-precision-identity-and-ad.md).

The shared topology layout and admission boundary are recorded in the
[ragged batch scheduling decision](../../.agents/notes/implemented/architecture/2026-09-20-ragged-batch-schedule-ownership.md).

# Decision: shared compiled CUDA region replay

Status: implemented, bounded TensorIR slice; broader qualification remains open.
Date: 2026-09-19

## Problem

Prepared TensorIR already owns stable device buffers and a stream, but repeated
execution still submits every generated launch through the host. A method-local
graph implementation would duplicate the runtime lifecycle needed by later
ProgramIR, SCF and CC consumers.

## Decision

Use a pure CaptureContract over the existing specialization records and artifact
identity, plus a method-neutral native CudaGraphRegion owner. TensorIR binds its
existing context to that owner. Capture only the fixed device computation;
refresh host inputs, return outputs and check arithmetic errors outside the graph.
The first eligible call executes ordinarily, the second captures and executes,
and later calls replay. Default execution remains ordinary.

## Invariants

Never execute a scientific region twice to hide warmup or a graph launch failure.
Capture errors must end capture before ordinary fallback. Preserve error checks,
input validation, detached outputs, current-device checks and context locking.
Graph identity covers artifact/compiler, workload/shape/layout/precision,
schedule, actual device/runtime/library state and owner-local native addresses.
An explicit invalidation discards the graph, not scientific state. Destruction
releases graph references before buffers, streams and library handles.

## Rejected alternatives

Capturing pageable host transfers or caller arrays would couple graph validity to
caller addresses and transfer implementation details. Independent method-owned
graph stacks would repeat lifetime and failure logic. Automatic promotion without
complete-endpoint evidence would overstate a launch optimization's benefit.

## Budget and control boundaries

Global ResourcePlan consumers remain on ordinary execution because graph-retained
storage is not yet a budgeted candidate. Device free-memory deltas are reported,
not treated as exact allocation bounds; driver-owned host storage is unmeasured.
The graph node ceiling is 4096. Capture failures remain sticky until invalidation.
Profiling temporarily uses ordinary execution without discarding a valid graph.

Existing device-tail-launch SCF/eigensolver experiments have distinct device-side
control semantics and are not replaced. The resident ABI also remains ordinary.
ProgramIR adoption, SCF/CC integration and iterative electronic-structure endpoint
qualification remain open under #460, #370 and #507.

## Validation procedure

Run the pure capture/native-mock tests and the existing TensorIR, specialization,
compiler-structure and CUDA-ownership tests. Under an allocated CUDA job, run
`tests/python/test_tensor_cuda_execution.py` with `VIBEQC_TENSOR_CUDA_TEST=1`.
The added real-device tests cover direct/packed GEMM, changed strided inputs,
profiling, explicit invalidation, arithmetic-error recovery and budget fallback.

`tools/benchmark_tensor_graph.py` records interleaved complete host-staged
TensorIR endpoints against independent NumPy arithmetic. Use both fixed and
changed numeric inputs, and report all trial medians rather than a best sample.
Capture/instantiate timing excludes initial warmup and first-launch upload;
the derived amortization count is correspondingly a setup-only estimate.
CUDA event intervals include transfers and host-submission gaps, not just kernels.
Changing numeric inputs is not a molecular changed-geometry qualification.

## Revisit when

ProgramIR has a stable execution-region contract, graph-retained storage has a
budgeted resource alternative, and an iterative SCF/CC endpoint is ready for
numerical, failure-semantics and paired performance qualification. Only then
consider broader promotion or integration of method-side state transitions.

## References

#507; #459; #350; #460; #370; docs/tensor_cuda.md.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Initial bounded evidence

At base 5a7fdeb, the expanded CPU/source suite passed 282 tests (41 optional
checks skipped). The allocated full TensorIR CUDA execution suite passed 34
tests (one optional public-HF/resource check skipped). Shared compiler structure,
CUDA ownership and SCF structure checks passed.

On RTX 5090 / sm_120, driver 580.95.05, CUDA compiler 12.9, the generated
24-stage multiply/add program has 50 captured device nodes. Each row is the
median of seven interleaved trials, 50 paired repeats per trial. Both paths
include input validation/staging, host transfers, all device work, arithmetic
checks and detached output allocation. Ordinary submission uses the same
optional diagnostic ABI as graph execution.

| Elements | Numeric inputs | Ordinary endpoint (ms) | Replay endpoint (ms) | Speedup | Ordinary/replay submission (ms) |
| ---: | --- | ---: | ---: | ---: | --- |
| 512 | fixed | 0.282818 | 0.198146 | 1.427x | 0.148967 / 0.006883 |
| 512 | changed | 0.285674 | 0.197329 | 1.448x | 0.151782 / 0.007084 |
| 16384 | fixed | 0.879239 | 0.792162 | 1.110x | 0.154191 / 0.007369 |
| 16384 | changed | 0.869775 | 0.787863 | 1.104x | 0.148250 / 0.007319 |

All four cases matched the independent NumPy output reference with observed
maximum absolute difference zero. Each retained one capture across all measured
repeats. Capture took 0.084--0.094 ms and instantiation 0.179--0.195 ms; the
setup-only amortization estimate is 3.1--3.5 replays, excluding warmup and first
launch/upload. Trial-median endpoint ranges did not overlap between modes in
any case. Observed graph-retained device free-memory delta was zero at runtime
accounting granularity; this does NOT mean the graph has no memory cost.

Reproduce inside a GPU allocation, with the appropriate compiler/provider paths:

```bash
PYTHONPATH=python VIBEQC_NVCC=/path/to/nvcc python tools/benchmark_tensor_graph.py \
  --cache /tmp/tensor-graph-cache --output /tmp/tensor-graph.json \
  --size 512 --depth 24 --trials 7 --repeats 50 --changed-inputs
```

Repeat for sizes 512 and 16384, with and without `--changed-inputs`. This is a
launch-sensitive TensorIR endpoint, not an SCF/CC speedup claim or a molecular
changed-geometry qualification. No tuning profile was promoted. Full JSON output
is regenerated locally rather than adding payloads to benchmarks/results.

The existing evidence-retention pre-commit check fails identically at base HEAD
and with this patch: benchmarks/results contains 105281931 bytes against a
100663296-byte budget. This slice adds no files/bytes under that directory and
does not delete historical evidence or relax the budget to mask the failure.

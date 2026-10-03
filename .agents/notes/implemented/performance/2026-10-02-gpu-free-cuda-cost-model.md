# Decision: GPU-free CUDA screening stays a relative cost model

Status: implemented
Date: 2026-10-02

## Problem

GenerativeQC already records semantic traffic, arithmetic work, register pressure,
PTXAS resources, and per-SM theoretical occupancy. Those facts are useful before
real-device benchmarking, but the shared model did not expose global launch
parallelism. A one-block grid can therefore look healthy under a per-SM occupancy
calculation even though it cannot fill a many-SM GPU.

This matters when GPU access is scarce: the compiler needs a cheap way to reject or
deprioritize obviously poor candidates before spending real-device benchmark time.

## Decision

Add `common.cuda_cost_model.static_cuda_cost` as a GPU-free screening layer over
the existing `GpuProfitability` and `CudaTargetInfo` contracts.

The report:

- prefers PTXAS register/shared-memory evidence when available;
- otherwise keeps static register/occupancy estimates explicitly approximate;
- computes theoretical resident blocks and occupancy from known resource limits;
- optionally combines grid block count with a caller-supplied or runtime-enriched
  SM count to expose one-wave global underfill;
- reports arithmetic intensity when semantic traffic and operation counts are both
  known; and
- exposes a deterministic shortlist key rather than a predicted duration.

The shortlist uses the whole-device occupancy bound, combining resident capacity
and available grid blocks. Raw wave saturation stays a diagnostic: ranking that
fraction directly would reward higher register pressure for shrinking its
denominator, even when the same grid supplies exactly the same active threads.
Known resource combinations that cannot admit one resident block sort after
viable candidates, before spill and occupancy preferences are considered.

`tools/analyze_cuda_cost.py` makes the same model usable from retained PTXAS logs
or pre-compilation estimates without initializing CUDA.

The CLI rejects incomplete declared resource rows and explicit architecture
contradictions, including mixed-architecture logs. Headerless resource snippets
remain usable with the requested target disclosed as a caller assumption, not
verified compiler provenance. Missing rows cannot silently disappear from the
resource maximum.

## Rejected alternatives

A cycle-accurate simulator was not made part of the compiler. It requires a
calibrated architecture/execution model and, for common trace-based workflows,
real-device evidence. Its output would also be too expensive for routine candidate
generation.

A fitted "milliseconds" predictor was also rejected. The repository does not yet
have sufficiently broad, versioned calibration data across kernels, GPU SKUs,
clocks, cache states, and launch shapes to justify an absolute timing claim.

SM count is not attached to `sm_120`: compute capability is an architecture
property while SM count is a device-SKU/topology property. Offline callers that
know the intended GPU pass that count explicitly.

## Invariants

- The model must not probe, allocate, synchronize, or execute a GPU.
- Missing evidence stays unknown; it must not be silently replaced by a guessed
  device topology or runtime.
- Occupancy and grid saturation are upper-bound screening signals, not achieved
  occupancy.
- Candidate promotion still requires complete real-device endpoint evidence under
  the existing performance gates.
- Screening priorities are meaningful only among candidates at the same evidence
  stage.

## Evidence

CPU-only unit coverage:

```text
PYTHONPATH=python python -m pytest -q tests/python/test_cuda_cost_model.py
```

The tests cover register/shared-memory limits, explicit target-device SM-count
input without a GPU probe, unknown-topology preservation, global-grid underfill,
screening ordering, payload scope, and invalid-input rejection.

## Consequences

Compiler and tuning code can share one inexpensive parallelism/resource report and
spend GPU benchmark budget on a smaller candidate set. The model can identify
obvious pressure and underfill problems, but it cannot model latency hiding,
instruction throughput, cache behavior, clocks, memory-controller effects, or
driver/runtime overhead accurately enough to claim a speedup.

## Revisit when

Add a calibrated timing layer only after GenerativeQC has retained enough
cross-device endpoint data to validate prediction error out of sample. A
cycle-accurate simulator remains an optional external validation tool rather than a
promotion oracle.

## References

- `.agents/notes/implemented/performance/2026-09-21-gpu-profitability-model.md`
- `.agents/notes/implemented/performance/2026-10-01-stationary-geometry-point-lanes.md`

## Follow-up: explicit experimental calibration

The [2026-10-03 calibrated timing decision](2026-10-03-calibrated-cuda-kernel-timing.md)
adds an optional caller-calibrated kernel estimate without changing screening or
production promotion. It narrows the rejection of absolute timing to distinguish
explicit experiments from validated predictions; no measured calibration or
held-out accuracy claim is supplied.

# Decision: experimental CUDA timing requires explicit workload evidence

Status: implemented
Date: 2026-10-03

## Problem

PR #1787 adds calibrated seconds on top of #1710's GPU-free static screening.
The original draft could treat PTXAS spill bytes as total runtime memory traffic,
use per-SM occupancy despite unknown global underfill, lose static evidence
caveats, and serialize seconds without the calibration needed to reproduce them.
It also left the meaning of work counts across multiple launches unspecified.

## Decision

Keep timing separate from screening and production selection. Require a named
device, canonical architecture, SM count, workload/precision/traffic convention,
measurement provenance, achieved rates, saturation threshold, and engineering
uncertainty fraction. No device calibration or fitted default is supplied.
Retain calibration, cost evidence, model identity, and all caveats in each report.

Interpret work as totals across repetitions of one homogeneous kernel with the
same per-launch grid and resource shape. Add launch overhead once per launch.
Only whole-device occupancy supports a timing estimate by default; per-SM fallback
requires explicit opt-in and remains visibly optimistic. Missing work, global
parallelism, or dynamic spill traffic leaves the total unknown, while independent
component times remain available. A known no-op is zero without resource evidence.

PTXAS spill loads/stores are static compiler diagnostics, not byte counters over
threads and loop iterations. Do not multiply by a guessed thread/loop count, and
do not add them directly to aggregate traffic. Nonzero/unknown spills require an
explicit dynamic total excluding bytes already included in semantic traffic.
Compiler-proven zero spills permit a zero-spill assumption.

Expose timing as an optional calibration input to the existing offline CLI.
Multi-kernel resource aggregation remains valid for screening but is rejected
for timing, since combining maxima or applying one roofline to heterogeneous
serial kernels loses the actual sequence of compute/memory bottlenecks.

## Rejected alternatives

- Silent per-SM fallback: it can conceal an arbitrarily underfilled global grid.
- Static spill bytes as total traffic: the units omit dynamic multiplicity.
- Vendor peaks or default occupancy/uncertainty parameters: these invent evidence
  and conceal workload-specific assumptions.
- A nonlinear occupancy fit, universal SKU-only rates, or learned correction:
  there is no retained held-out calibration set to justify these yet.
- SCF iteration prediction or automatic cold/warm/geometry composition in the
  generic compiler: those require method/runtime ownership, explicit reuse state,
  and host-stage/overlap evidence absent from a kernel resource report.
- Calling an engineering fraction a confidence interval: it has no demonstrated
  residual coverage. Neither static occupancy nor this band bounds actual time.

## Invariants

- No runtime import, GPU probe, or production auto-selection dependency.
- Unknown evidence is not zero; zero is explicit, including zero-launch no-ops.
- Static screening ordering and semantics remain unchanged.
- Calibration and work must agree on precision and operation/traffic conventions.
- Architecture and known SM topology contradictions fail before reporting seconds.
- Invalid inputs fail; arithmetic overflow produces unknown times, never NaN/Inf.
- Endpoint validation and independent numerical gates still govern promotion.

## Evidence

CPU-only tests cover hand-computed roofline/underfill cases, launch/repetition
accounting, saturation, static versus dynamic spills, missing and contradictory
evidence, zero work/grids, calibration validation, strict JSON, CLI provenance,
and rejection of heterogeneous PTXAS timing. Reproduce with:

```bash
PYTHONPATH=python python -m pytest -q \
  tests/python/test_cuda_time_estimator.py \
  tests/python/test_cuda_cost_model.py tests/python/test_cuda_cost_cli.py
python tools/check_compiler_structure.py
```

The rates in tests are synthetic arithmetic fixtures. No real-GPU prediction
accuracy, measured calibration, endpoint speedup, or production readiness claim
follows from these tests.

## Consequences and revisit conditions

This is an auditable experimental interface, not a validated timing predictor.
Whole-device evidence and dynamic traffic requirements intentionally produce more
unknown results than the initial draft. Revisit the correction curve, family/SKU
hierarchy, residual bands, or endpoint composition only with retained held-out
measurements and explicit runtime scenario ownership.

This narrows the earlier rejection of any absolute timing layer in the
[GPU-free screening decision](2026-10-02-gpu-free-cuda-cost-model.md): optional
caller-calibrated experiments are now supported; unvalidated production timing
claims and promotion remain rejected. Current usage is documented in
[Experimental CUDA kernel timing](../../../../docs/developer/cuda_time_estimator.md).

## Follow-up: measured RTX 5090 profile

The [RTX 5090 calibration decision](2026-10-03-rtx5090-cuda-timing-calibration.md)
adds actual Slurm-scheduled measurements, isolated training/holdout qualification,
and an explicitly selectable FP64/streaming profile. The original absence of
measured calibration above describes this decision's initial implementation;
the kernel/endpoint and production-promotion boundaries remain unchanged.

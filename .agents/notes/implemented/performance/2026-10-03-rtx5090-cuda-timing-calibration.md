# Decision: retain a measured RTX 5090 timing profile with isolated validation

Status: implemented
Date: 2026-10-03

## Problem

The initial #1787 implementation required caller calibration but shipped only
synthetic arithmetic tests. That protects formula semantics without providing
usable achieved rates or demonstrating how its linear occupancy approximation
behaves on real hardware. The requested follow-up was actual device calibration.

## Decision

Add a standalone, ccache-built CUDA probe and a CPU-replayable calibration fitter.
Measure the allocated RTX 5090 through Slurm, preserving CUDA_VISIBLE_DEVICES.
The compiler estimator still performs no device probe, build, or calibration.

The experiment has 23 training cases (empty launches, independent FP64 FMA chains,
and streaming copies) and 17 held-out cases. The latter use different grids,
iteration counts, array sizes, launch counts, and a separate mixed FP64/streaming
kernel. Streaming arrays are individually at least four times L2 capacity.
Seed/initialization/reduction operations are included in the FMA work count, and
all semantic operation/byte counts cover the complete measured launch batch.

Retain all wall-time and event-time samples. Use the median complete synchronized
batch wall time for fitting: host submission, launch checks, event recording, and
final event synchronization are included; setup, transfers, warmup and host-oracle
validation are outside the timed interval. This avoids labelling device event
latency as a host-visible kernel-batch endpoint prediction.

Fit only training data. Empty launch batches determine launch overhead. Search a
fixed 0.001 saturation-threshold grid; each candidate's rates are geometric means
of corrected achieved rates. Equal-weight compute/memory log-error loss selects
one common threshold. Set the engineering band from training residuals alone,
rounded upward to 0.05 with a 0.10 floor. Held-out accuracy and band coverage are
reported separately, and failed qualification exits nonzero while keeping data.

Freeze median/P95/maximum relative-error gates at 20%/35%/50% before validation;
require them both globally and within each held-out family. These tolerances
qualify the recorded probe workload only. No register-pressure, spill-heavy,
cache-resident, other-precision, concurrent-stream, or chemistry claim follows.

## Rejected alternatives

- Peak TFLOPS/bandwidth as rates: calibration must measure achieved behavior.
- Cache-resident copy as DRAM-like bandwidth: array sizing makes cache scope
  explicit and avoids an accidental warm-L2 profile.
- Event-only timing as wall time: it can omit host dispatch and synchronization.
- Fitting held-out residuals or widening the band to cover them: this leaks the
  evaluation data into the model. A regression test changes only holdouts and
  proves the fitted parameters and engineering band are unchanged.
- A nonlinear correction or separate compute/memory thresholds: the retained
  common-threshold model passes the declared domain gates; extra parameters are
  not justified by this evidence.
- Automatic global selection of the profile: matching architecture/SM count does
  not establish precision, kernel, cache, software or clock compatibility.

## Evidence and limitations

The [retained bundle](../../../../benchmarks/results/cuda-timing-rtx5090-20261003/README.md)
contains the exact calibration, all measurements, scored holdouts, numerical
errors, compiler/PTXAS identity, source and binary hashes, and Slurm/device
provenance. It records actual results rather than embedding a second numerical
summary in this note. CPU replay must reproduce the retained profile and scores.

The probe validates 17 deterministic spread indices per case against independent
host std::fma/copy references, with a 2e-12 absolute-error gate. A small native smoke
mode covers all kernel families and non-divisible grid-stride tails under CUDA
Compute Sanitizer. It is excluded from performance fitting.

An exploratory seven-sample run preceded the final nine-sample collection. Its
results supported retaining the original common-threshold formula. One attempted
formal run failed during compilation because the filesystem was full; it supplied
no timing samples and was not used for parameter selection. Only this task's
reproducible documentation build was removed, and compiler temporary files moved
to task-owned tmpfs storage. No ccache reset/disable or GPU scheduler bypass was
used.

## Consequences and revisit conditions

The estimator now has one explicitly selectable measured profile and an auditable
way to regenerate it. Its engineering band is still not a statistical confidence
interval. Repeat qualification for other precisions, dependent-instruction
families, cache regimes, resource pressure, launch protocols, or software/clocks
before applying it there. Complete chemistry endpoints still require their own
setup, iteration, force/response, transfer, and numerical acceptance evidence.

This extends the earlier [calibrated-kernel decision](2026-10-03-calibrated-cuda-kernel-timing.md),
which intentionally had no measured calibration at implementation time.

Follow-up: the [refined timing model decision](2026-10-04-refined-cuda-timing-model.md)
adds new training coverage and fresh validation for separate resource response,
family rates and fixed batch cost. It supersedes the decision to defer nonlinear
corrections while preserving this v1 profile, its decoded measurements and replay.

# Decision: aggregate derivative work by class and primitive signature

Status: implemented
Date: 2026-09-16

## Problem

The six executed-domain counters added for signature scheduling establish how
many shells, primitives and weights run, but cannot distinguish dynamic Boys
series work, generated polynomial work, weight folding and gradient scatter.
Issues #392–#394 need those distinctions before changing their ownership or
lowering. Counts alone also cannot assign time to these operations.

## Decision

`VIBEQC_DF_SHELL_WORK=1` supplies an optional diagnostic sink to the existing
generated geometry and native shell consumer. Geometry records actual Boys
series iterations. The compiler emits fixed preparation and per-component
convolution counts from its existing cache layout and component powers. Native
owners aggregate these with actual sparsity, public expansion lengths, pair
folding and physical center identities. No mathematical operation is replaced.

A fixed buffer holds 24 packet rows, 16 shards and 27 unsigned 64-bit counters:
82,944 device bytes and the same host capacity. The response arena charges
this storage before selecting its panel; host storage outlives exceptional
stream drains. The launcher reads and drains only when diagnostics are enabled.
It publishes both class totals and primitive-signature rows before reuse.
Readback bytes and stream drains enter the existing resource ledger.

The reducer independently reconstructs signature visits from public basis
metadata and actual panel intersections. Native tests independently enumerate
the same domain for sparse, partial-panel, full/symmetric/packed and mixed
Cartesian/spherical cases. Six detailed totals must equal the existing device
counters. Generated-operation tests explicitly enumerate coefficient loops.

## Semantics and invariants

- Counts describe emitted source operations, not optimized instructions or
  hardware transactions. Cache values include emitted zeros.
- `boys_small_argument` is the `T < 1e-8` subset of `boys_series` (`T < 30`).
  Series and large-argument counts partition evaluations.
- Public nonzero weights are logical weights after pair folding. Symmetric
  off-diagonal pairs load two public values before producing one such weight.
- A gradient update is shared-atom work if either other mathematical center
  has the same physical atom. Shared and distinct counts partition A/B/C.
- Packet rows retain each signature's work. A packet has one kernel duration;
  dividing that duration by primitive counts does not measure signature time.
- Nsight activities must match trace class-launch counts and generated block
  schedules. CUDA event intervals may include stream idle time and remain a
  separate observable.
- Normal runs allocate no diagnostic storage and issue no diagnostic updates,
  readbacks or drains. Strict FP64, metric, screening and force gates remain.

## Alternatives and costs

One diagnostic atomic per coefficient or series iteration would distort the
work being studied. Each owner instead accumulates locally and publishes its
totals once; deterministic counts come from the generated work model.
Host-only reconstruction cannot observe dynamic Boys branches or sparse
Cartesian cancellation, so it validates rather than replaces device evidence.

Runtime-disabled instrumentation still changes static register allocation:
SSS packet registers rise 96 → 132 and PSS 130 → 148; DSS stays 252. A separate
compile-time diagnostic kernel family would avoid that coupling at a cost in
code size and maintenance. Qualification did not justify that redesign:
five-sample clean endpoint medians change 0.923888040 → 0.924347867 s at 384 AO
and 5.298158605 → 5.331490926 s at 768 AO (+0.05% and +0.63%). All samples
replay an identical frozen density per size and take three SCF updates.

The library grows from 198,259,648 to 205,297,872 bytes. Intrusive overhead,
process residency, per-class resources and all raw samples are retained in the
[qualification records](../../../../benchmarks/results/issue395-df-work/README.md).
No enabled-counter endpoint is a performance claim.

## Evidence

The 384 diagnostic observes 21,976,064 primitive/Boys evaluations, 263,463,122
positive-series iterations, 64,583,424 generic polynomial calls and
2,925,586,944 convolution iterations. Its d-containing classes account for
12.97% of primitive work and 57.62% of convolution work. Their measured
intrusive Nsight activity is 82.689 ms out of 199.283 ms; counts alone do not
establish that time fraction or identify an instruction bottleneck.

The generated/compiler/Libcint tests, native shell-pair oracle, Compute
Sanitizer and complete CUDA force tests are recorded with counts and log
hashes in the qualification's validation record. Both retained sizes include
independent GPU4PySCF energy/force gates and matching scientific domain totals.

## Revisit when

Use the same ledger to prove which work #392–#394 remove. Update conservation
checks and independent tests when an ownership or lowering changes; do not
silently reinterpret an existing field. Reconsider compile-time separation if
disabled-counter complete endpoints regress materially on a qualified workload,
or if a larger diagnostic set causes unacceptable register/code growth.

## References

- [Issue #395](https://github.com/jinzhezenggroup/vibeqc/issues/395)
- [Current shell derivative contract](../../../../docs/developer/df_shell_derivatives.md)
- [Performance requirements](../../../../docs/maintainer/performance_engineering.md)

# Decision: explicit precision boundaries with qualification-bound identities

Status: implemented
Date: 2026-09-20

## Problem

Storage, compute and accumulation choices affect numerical behavior, traffic and
live memory. A typed lower-precision DAG alone cannot identify which external
qualification applies. Program equation hashes intentionally exclude provenance,
so qualification labels kept only in provenance collide in plan/search caches.
Cast AD also needs an explicit scientific contract, not just agreement between
two implementations introduced together.

## Decision

Represent precision changes with explicit cast nodes; no implicit promotion,
TF32, fast-math or device-dependent dtype guessing is introduced. Resolve dtype,
sensitivity, cast traffic and strict-audit ownership into PrecisionSchedule.
Schedule schema v2 includes the canonical precision-request identity and sorted
source-value/qualification pairs. The request binds the source equation and
per-value directives; TensorPlan binds this schedule to its actual target and
resource/layout identity. Re-lowering records the parent schedule identity so
an earlier qualification cannot disappear when a later request is empty.
Validate serialized request identity before planning. Qualification labels are
external identifiers, not proof of numerical validity or permission to bypass
the existing independent numerical and complete-endpoint promotion gates.

The cast derivative contract is an arithmetic linearization: JVP converts the
incoming tangent to the target dtype; VJP converts the incoming cotangent back
to the source dtype. A round trip therefore rounds tangent/cotangent values to
FP32 before returning FP64. This deliberately ignores discontinuities of the
bit-level rounding map; it is not a finite-difference derivative of a staircase
function, nor an exact Euclidean adjoint across finite-precision rounding.
Independent NumPy conversion assertions exercise both AD implementations using
values not exactly representable in FP32, rather than comparing only those two
implementations to one another.

## Invariants and rejected alternatives

Do not include external qualification in Program.logical_hash: that hash remains
a scientific equation identity. Do not discard qualification during schedule
resolution, optimization, repeated lowering, plan deduplication or promotion.
The validated existing dtype/cast graph remains unchanged by this identity fix.
Strict FP64 remains the method-controller audit/default boundary. Automatic
candidate generation excludes sensitive operations and reductions. Distinct
compute and accumulation dtypes remain rejected until implemented and qualified.
Keeping only a free-form note, or declaring low precision safe because two new
AD paths agree, was rejected.

## Evidence and remaining scope

Three regression cases fail on the pre-fix identity path: different qualification
scope, qualification inherited through repeated lowering, and corrupted request
identity. The repaired scope/cast/AD/search host suites pass. Independent NumPy
round-trip expectations cover both generated and reference tangent/cotangent
paths. No molecular method default or performance promotion is established.

## Revisit when

Revisit the linearization contract for error-controlled higher derivatives and
when method controllers supply structured workload/device/evidence qualification
objects. Separate compute/accumulation lowering needs its own backend and
independent numerical validation before admission changes.

Refs #528, #375, #174, #508; `tests/python/test_tensor_precision.py`.

The later [FP64 accumulation decision](2026-09-21-tensor-fp64-accumulation.md)
supersedes the rejection of distinct compute/accumulation lowering and extends
schedule payload v2 to v3 with an explicit source-to-lowered execution scope.
The identity, qualification-scope, and cast-AD decisions here remain current.

Agent: ChatGPT
Model: GPT-6 Astra Pro

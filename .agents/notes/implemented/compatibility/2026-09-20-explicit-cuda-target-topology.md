# Decision: separate CUDA architecture metadata from runtime device topology

Status: implemented
Date: 2026-09-20

## Problem

The static CUDA architecture catalog carried `sm_count=170` for `sm_120`, and
generic integral-compiler APIs silently defaulted to that target. An SM count is
a property of one concrete GPU, not of compute capability 12.0, so the catalog
could make a measured RTX 5090 topology look universal.

## Decision

Static `CudaTargetInfo` entries keep architecture-invariant limits but leave
`sm_count` unknown (`None`). Runtime probes enrich the same target with a
positive SM count. `require_sm_count()` is the fail-fast boundary for any future
schedule/resource policy that genuinely needs device topology.

Generic schedule construction, capability reporting, and autotuning trial
enumeration no longer use the compatibility `DEFAULT_CUDA_TARGET` as a default.
They require an explicit target or architecture. The historical exported
constant remains only as a compatibility alias for downstream code that chooses
to import it explicitly.

Production manifests remain allowed to retain measured `sm_count` provenance:
that metadata records the device used to qualify a tuned profile and is not the
architecture catalog.

## Rejected alternatives

- Replacing `sm_120` with another implicit architecture such as `sm_80` would
  move rather than fix the hidden-target problem.
- Inventing a fake `sm_generic` compilation target would blur schedule
  constraints with NVCC code-generation identity and could leak an invalid
  architecture into artifact keys or compiler commands.
- Removing `sm_count` from tuning/profile provenance would discard useful
  device-compatibility evidence; only the static catalog must leave it unknown.

## Invariants

- Static architecture catalog entries never contain concrete device SM counts.
- A positive SM count originates from runtime/device evidence.
- Generic compiler APIs never choose `sm_120` merely because a caller omitted a
  target.
- Explicit offline/AOT architecture selection remains supported.
- Measured production/local profiles may record concrete topology as provenance.

## Evidence

`tests/python/test_cuda_target_policy.py` covers catalog targets, runtime
enrichment, fail-fast topology access, explicit scheduling/reporting, and
target/architecture mismatch rejection. Existing codegen tests retain their
historical `sm_120` baseline explicitly rather than through a hidden default.

## Consequences

Callers that previously omitted a CUDA target must now state the architecture or
pass a runtime-enriched `CudaTargetInfo`. This is intentionally visible API
friction: source generation and scheduling should not acquire device identity
from an unrelated module constant.

## Revisit when

A backend-neutral schedule-constraint type is introduced that can represent a
portable CUDA schedule without carrying NVCC architecture identity.

## References

- #594
- #159
- `python/vibeqc_compiler/common/cuda_target.py`

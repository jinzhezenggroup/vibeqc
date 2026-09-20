# Decision: type and gate DFT grid/XC execution schedules

Status: implemented
Date: 2026-09-20

## Problem

Issue #168 needs generated grid/XC schedules that can be compared without changing
the functional, quadrature, screening, precision, density source, or requested
derivatives. Existing prepared XC execution already had a resident CUDA
LDA/PBE-potential path and a CUDA-collocation plus generated CPU contraction
fallback, but the choice was implicit and could not be represented in the local
profile or complete-endpoint promotion contract.

The exact implementation base is
`348b5c899d64d5d3758fd41daa9878796840a1e9`. Open PR #713 owns moving the
resident XC scientific CUDA bodies into compiler-generated source. The #660
children own stationary-force timeline/replay/batching/weight-fusion work. This
slice deliberately does not duplicate either ownership.

## Decision

Add an immutable compiler-owned `GridXcExecutionSchedule` with only two
currently executable lowerings: `device_fused` and `host_unfused`. The first
keeps supported LDA/PBE potential contraction and Vxc assembly on the CUDA
prepared path. The second keeps CUDA AO/features but explicitly downloads jets
and uses the existing generated CPU XC/potential contraction. Explicit fused
selection fails if its capability disappears; the unfused path is the reliable
fallback.

Keep mathematical/workload identity separate from schedule identity. A DFT
workload records architecture, functional identity, ingredients and jet outputs,
grid/model and screening identity, FP64 precision, spin/observable, density
route, and source identity. Deterministic pre-compilation admission requires
measured workspace/source sizes and rejects unsupported capabilities or
live-value, workspace, and generated-source bounds. The live-value estimate is a
conservative planning model, not claimed PTXAS register data.

Extend the existing #136 four-file local profile bundle with optional
`dft_schedules`; do not create another cache or CLI. A DFT winner must have
exact workload/schedule hashes plus legality, resource, independent numerical,
performance, and complete endpoint evidence. Runtime selection uses the same
active bundle and returns a typed-schedule payload only for an exact workload
hash.

Add a DFT endpoint gate on top of the existing numerical/performance gate. It
requires at least five equal paired samples, one scientific identity, complete
energy and analytic-force outputs, explicit synchronization and interleaving,
matched SCF iterations, and the existing energy/force/translation tolerances and
noise-aware speedup gate. A slower/noisy candidate is ordinary negative evidence
and is not promoted.

## Invariants

Schedule identity never changes functional/grid/screening/precision/source
identity. Existing default prepared behavior is unchanged. Unsupported fused
execution raises instead of silently changing the route. Existing HF profile
bundles remain valid. No fixed-density E/V timing can satisfy the DFT
energy-plus-force promotion gate.

## Evidence

Host unit tests cover scientific/schedule identity separation, deterministic
admission and rejection, profile backward compatibility and exact invalidation,
schedule payload round-trip, and complete endpoint acceptance/rejection.
The CUDA-gated density-candidate test executes both schedules on the same PBE
fixture/mask/source and checks each against independently stored energy and
potential references as well as against each other.

The Windows Runner cannot execute the native CUDA fixture and its POSIX profile
installation tests because it lacks the built native library/`fcntl`. The
configured one-shot `ssh qz` path was also unavailable during this run with a
websocket HTTP 500 handshake, so no real-device performance result is claimed
here. Real-device evidence must be collected through the repository Slurm/qz
path before any schedule is promoted.

## Consequences and revisit conditions

The generated/prepared grid-XC schedule and profile/promotion contracts are now
explicit and independently testable. The public CUDA KS/force endpoint still
runs through the native C++ path rather than `PreparedXCContractions`, so this
slice does not claim full #168 closure or a complete-SCF schedule speedup.
After #713/native endpoint wiring exposes the same schedule identity to the
production KS loop, run real-device cold, warm same-geometry, changed-geometry,
multi-scale and batch interleaved evidence and publish a winner only if the full
energy-plus-force gate passes. Retain negative evidence otherwise.

## References

- #168
- #136
- #163
- #235
- #660 and children
- PR #713
- `docs/local_autotuning.md`
- `docs/density_sources.md`

Agent: ChatGPT
Model: GPT-5.6 Sol

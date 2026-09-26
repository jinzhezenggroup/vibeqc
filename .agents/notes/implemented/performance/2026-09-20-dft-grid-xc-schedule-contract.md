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
legacy one-shot `ssh qz` tunnel was unavailable with a websocket HTTP 500
handshake, but the supported Inspire job path was subsequently used for
real-device qualification.

At implementation revision
`cc3fc40d548f082cbd67aab815824b0bd959c89b`, Inspire job
`vibeqc-168-smoke-v3-cc3fc40d` completed successfully on a real NVIDIA
GeForce RTX 4090 (49,140 MiB, driver 595.71.05, CUDA 12.9.86). A clean
reconfigured native CUDA build completed all 261 targets and linked
`libvibeqc.so` with SHA-256
`698aa44aee146b3d8dbe86e15c6734cecb8eb184d180c16d475a1ca9e235b571`.
The focused schedule/candidate/fallback suite then passed 7/7 tests in 10.57 s,
including independent stored PBE energy/potential checks for both
`device_fused` and `host_unfused`.

A separate fixed-density ablation on the same source/library retained 48
executions: first executions plus seven alternating-order warm pairs for each
of H2 (2 AO/32 points), water (7 AO/48 points), and spherical-f (16 AO/32
points). Median warm wall times were respectively 9.60/17.68 ms,
14.60/17.95 ms, and 11.48/18.09 ms for fused/unfused, corresponding to
1.84x, 1.23x, and 1.58x fused speedups. Every execution met the unchanged
independent PBE energy/potential gate. The raw JSON is retained under
`/inspire/qb-ilm/project/chemicalreaction/czxs25220150/experiments/vibeqc/issue-0168-dft09/rtx4090-ablation-cc3fc40d`
with SHA-256
`e45f191e9ca0885da1c8ffc74b7f164b11d732344a54bb94308ff7a4e7d9637d`.
This component evidence is deliberately marked non-promotion evidence.

## Native-KS follow-up

After PR #713 and PR #725 merged, the public CUDA KS owner gained an execution
policy seam without changing the semilocal mathematics. KS options v3 selects
`device_fused` or `host_unfused`; v1/v2 descriptors preserve the historical
device-fused default. The unfused path explicitly stages the current density to
the audited CPU semilocal integrator and uploads Vxc into the same CUDA SCF
control flow. Resource planning accounts for the missing resident XC arena and
the added host staging.

Runtime profile application is batch-local. It derives the same compiler-owned
scientific workload identity from geometry/GridSpec/functional/source/target
without materializing another quadrature, and applies a validated local winner
only when every item exact-matches the profile and selects one common resolved
schedule. Geometry changes or mixed matches keep the portable default.

At revision `39c31ff8bfd9e532aee7ac02638b4ea64089d31a`, qz job
`vibeqc-168-endpoint-c124-39c31ff8` retained compile-cold records plus seven
interleaved synchronized warm complete `energy+forces` pairs on an RTX 4090.
`device_fused` passed `dft_endpoint_gate` against `host_unfused` for H2,
water, and an H2 batch of two with median speedups 1.116x, 1.141x, and 1.146x;
the corresponding 95% bootstrap lower bounds were 1.085x, 1.134x, and 1.140x.
SCF branches matched, and the largest energy/force discrepancies were below
`2e-13`/`2e-14`. Changed-geometry replay also preserved parity.

The raw runner recorded the schedule-family hashes before binding the default
`point_tile=256`; it is retained as benchmark evidence, not relabeled as an
installable local profile. The resolved execution hashes are recorded alongside
the raw evidence in
`benchmarks/results/issue168-grid-xc-schedules/complete-endpoint-summary.json`.
The profile bundle validator still requires matching legality, resources,
independent numerical evidence, source hashes and the resolved schedule
identity before activation.

## Consequences and revisit conditions

The grid/XC execution choice is now represented consistently across prepared
contractions, native CUDA KS, resource planning, complete endpoint evidence and
local profile lookup. `device_fused` is the measured complete-endpoint winner
for the retained RTX 4090 PBE/STO-3G workloads and remains the portable default.
No claim is made that this one hardware/workload family is globally optimal;
new architectures, grids, functionals, arithmetic policies or algorithmic
candidates still require their own exact workload identity and promotion
evidence.

## References

- #168
- #136
- #163
- #235
- #660 and children
- PR #713
- `benchmarks/results/issue168-grid-xc-schedules/README.md`
- `docs/local_autotuning.md`
- `docs/density_sources.md`

Agent: ChatGPT
Model: GPT-5.6 Sol

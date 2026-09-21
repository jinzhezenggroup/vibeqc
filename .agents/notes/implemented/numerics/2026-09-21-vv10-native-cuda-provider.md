# Decision: extend the bounded VV10/rVV10 primitive through one CPU/CUDA provider

Status: implemented
Date: 2026-09-21

Supersedes the CPU-only production-boundary decision in
2026-09-21-vv10-bounded-native-cpu.md; that note remains the rationale for the
first #812 slice.

## Problem

The merged A/B/C work and #812 CPU slice established the versioned
VV10/rVV10 definition, independent direct oracle, fixed-density KS potential,
analytic nuclear source chain, and bounded native CPU pair traversal. #491D
still required a mathematically matched real CUDA lowerer, resource accounting
that includes transactional publication, and consumption by the existing
fixed-density MethodIR primitive without duplicating #781's generic KS plan.

## Decision

Keep one native nonlocal plan/C ABI for both backends. The prepared context
selects CPU_REFERENCE or CUDA and binds the exact device id; CUDA never silently
falls back to CPU. Both backends consume the same variant, b/C/coefficient,
fixed grid, total density and total-density gradient semantics.

CUDA assigns one thread to each outer grid point and performs that point's
ordered j accumulation serially. The full Ngrid x Ngrid matrix is never
materialized and no scientific floating-point sum uses atomics. The only
atomic operation publishes a failure flag. Energy is reduced on the host in
outer-point order.

The public C boundary stages feature/geometry outputs privately and copies them
to caller buffers only after scientific success. maximum_bytes therefore
bounds the worst provider-owned execution peak, including staging:

- CPU: 12*N doubles of host storage for the six retained scientific vectors
  plus worst-case feature/point/weight staging;
- CUDA: 7*N host doubles plus (21*N + 1) device doubles in geometry mode.

Diagnostics publish total, host and device workspace separately, plus N^2 pair
work and the effective tile count. Python plan/evaluation identities bind
native source identity, backend/device, primitive identity and exact parameters,
budget/tile/derivative capabilities, and hashes of every fixed-grid input.

vibeqc_compiler retains no runtime import. It accepts an injected pair provider,
while the no-provider path remains the independent small-grid oracle.
FixedDensityMeanField.from_method injects the native provider by primitive
presence and the already prepared Fock backend; there is no wb97m-v
method-name branch.

## Ownership boundaries

This change does not create a second MethodIR-to-native KS execution-plan
mechanism. PR #781/#396 owns that generic public seam. It also does not
implement #164 tau/r2SCAN. Public complete omegaB97M-V remains fail-closed
until those independent dependencies and their end-to-end qualification land.

## Qualification gates

Both VV10 and rVV10 must match the independent fixed-grid oracle for energy,
vrho/vsigma and explicit geometry derivatives. CPU/CUDA parity, tile
invariance, RKS/UKS total-density semantics, changed-input identity, ragged
atomic publication, stale density/grid negative gates, multi-step rebuilt-grid
nuclear finite differences, bounded-resource rejection, and real-device
cold/warm/changed-geometry timing/peak-memory evidence are required.

## Evidence

The final integrated tree was validated against a production CPU
libvibeqc build on qz and the final maintained CUDA translation unit on a real
NVIDIA H100 80GB HBM3 (driver 595.58.03, CUDA 12.9):

- the CPU integration suite covering the nonlocal Python ABI, native runtime,
  MethodIR execution, independent reference, kernel replay, RKS/UKS, ragged
  execution, stale identities and rebuilt-grid gradients passed 51 tests; the
  six CUDA-parametrized cases skipped only because that particular full library
  was intentionally built CPU-only;
- final-source C++ and CUDA compilation plus direct Vv10Plan CPU/CUDA execution
  gave energy differences of 2.17e-19 for both variants, maximum vrho
  differences of zero, and maximum remaining derivative differences below
  4.3e-22;
- the final Python/C-ABI CUDA provider matched the independent fixed-grid
  potential oracle to 4.34e-19 in the representative H2 fixture, recovered
  RKS/UKS equivalence, and retained the three stationary nonlocal source
  families;
- complete rebuilt-grid directional finite differences at 1e-3, 3e-4 and 1e-4
  matched the analytic VV10 and rVV10 nuclear gradients. The largest
  translational residual observed was 8.7e-19, and every displaced execution
  received a distinct deterministic identity;
- for 8192 points with geometry derivatives, the native CUDA path reported
  67,108,864 pair evaluations and 32 outer tiles. Cold, warm and
  changed-geometry wall times were 0.936342 s, 0.356483 s and 0.356609 s.
  Provider-owned workspace was 458,752 host bytes plus 1,376,264 device bytes
  (1,835,016 total); process-level peak GPU memory observed by nvidia-smi was
  520 MiB.

Monolithic NVIDIA builds on this qz notebook intermittently failed before
reaching the nonlocal source because the platform process environment lost its
working directory while compiling unrelated GFN2/direct-JK CUDA translation
units (getcwd/assembler output-path failures). The final nonlocal CPU source,
C API and CUDA source were therefore also compiled directly as one isolated
scientific slice on the same H100. Repository CI remains the full-library CUDA
build gate; this environment failure is not used as numerical evidence.

## References

- #491
- PR #812
- PR #781 / #396
- #164
- tests/python/test_vv10_native_runtime.py
- tests/python/test_vv10_native_integration.py

Agent: ChatGPT
Model: GPT-5.6 Sol

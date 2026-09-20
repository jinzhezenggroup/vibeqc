# Decision: native CUDA CPKS shares the SCF XC owner and point differential

Status: implemented
Date: 2026-09-20

## Problem

Native CPU LDA/PBE RKS/UKS CPKS is qualified, but exporting a CUDA state and
running the CPU kernel does not qualify a CUDA response endpoint. Issue #179
requires reuse of the shared solver and semilocal kernel policy, with actual
device execution, strict state identity and explicit resource boundaries.

## Decision

Extend the existing `CudaXcPlan` with a response mode. Keep reference and
directional feature panels, reusing AO evaluation, density-feature contractions,
the shared point directional derivative and potential assembly. After reference
features are retained, the direction reuses the AO-density product scratch.
Ordinary SCF layouts remain unchanged. `xc_point_response.hpp` becomes a shared
host/device policy; this inventories existing scientific formulas as CUDA-bearing
source and makes no claim to retire scientific CUDA code.

The private native response owner retains its own arena, reference density,
direction buffer and stream. The exact snapshot token gates preparation and every
action. It re-reads the actual native derivative source, verifies packed basis
equality, and copies explicit points/weights rather than rebuilding a grid.
The re-read/export is a real preparation cost and has separate diagnostics.
RAII drains the stream before releasing borrowed buffers on exceptional exits.
Outputs are staged until the point/domain status and live token both pass.

The existing spin CUDA Fock adapter supplies a J-only seam for semilocal response.
No K is requested. Python owns the AO/MO transforms and the existing Krylov
controller. Shared spin layout preserves alpha/beta responses and cross-spin
correlation; no second solver or XC expression is added.

## Rejected alternatives

- Routing a CUDA snapshot through CPU AO/point evaluation would hide scientific
  fallback and fail the device-action requirement.
- Creating a second XC formula/assembler would duplicate SCF-domain policy,
  especially tail and empty-spin derivatives.
- A missing ECP wire suffix cannot prove an all-electron Hamiltonian. The live
  native proof checks both ECP terms and atom core counts; old libraries without
  proof stay unbound.
- Calling the host-controlled solve fully resident would hide AO/MO and Krylov
  storage and all matrix transfers. Resident blocked execution remains separate.

## Invariants

- Preserve unchanged LDA/PBE SCF-domain equations, no clipping or skipped tails.
  Exact vacuum permits only zero direction; empty spin permits tangent directions
  only. Meta-GGA, hybrid, range-separated, DF CPKS and ECP remain unsupported.
- Preserve method/grid/basis/provider/reference identity and token revocation on
  successful or failed SCF replay, response closure and batch closure, even when
  a zero RHS avoids all operator actions.
- One retained device budget covers response XC and J. Borrowed SCF/eigen state,
  preparation transients, host arrays/Krylov and CUDA runtime/library allocations
  remain explicitly outside it.
- Count both initial snapshot export and preparation's second export. Successful
  XC actions upload one spin AO direction, download one response matrix plus
  28 bytes of scalars/status, and use two explicit fences. Failure cleanup fences
  are also counted. J host payload counters are not PCIe measurements.

## Evidence

`tests/python/test_response_native_cuda.py` invokes the same independent
libcint/Libxc, finite-rotation, transpose, reconverged-density and complete
multi-RHS assertions used for native CPU RKS/UKS, on CUDA water and LiH+ states.
CPU AO/XC/J and SCF rerun seams are patched to fail during response actions.
Small H2/H2+ cases exercise resource/domain rejection, replay and closure, and
actual r2SCAN/ECP states test method/Hamiltonian rejection.

`vibeqc_xc_response_cuda_tests` uses all 78 committed independent 450-digit
original-energy directional fixtures. Acceptance keeps the CPU-tier relative,
exchange/correlation cancellation and subnormal gates unchanged. The test has
only a device wrapper kernel; production formulas have one shared source.

Reproduce with a Release CUDA build, `VIBEQC_RESPONSE_CUDA_TEST=1`, the matching
`VIBEQC_LIBRARY` and an explicit finite Slurm `main`/`gpu:5090:1` allocation.
Conda Python DT_RPATH can resolve an older cuSOLVER before the configured toolkit;
preload a coherent CUDA 12.9 set (nvJitLink, cudart, cublasLt, cublas, cusparse,
cusolver) before Python when necessary. Never alter Slurm's visible devices.

## Consequences and revisit conditions

Each XC action still recomputes AO/reference features and crosses a host matrix
seam. That bounded implementation is intentional until complete solve profiling
justifies caching or resident integration. Extend the existing owner/vector seams
when addressing those costs; do not reinterpret retained bytes as endpoint peak
memory or point tests as a complete performance result.

## References

- Issue #179; native CPU RKS/UKS #649/#659; spin CUDA #661.
- [Current response contract](../../../../docs/response.md).
- [CPU spin rationale](2026-09-20-native-uks-cpks.md).

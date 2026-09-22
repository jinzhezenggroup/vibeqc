# Decision: Keep FP32 AO arithmetic behind an FP64 storage boundary

Status: implemented
Date: 2026-09-22

## Problem

The native device-fused DFT path evaluates AO values in FP64. Route B needs to
measure whether AO arithmetic itself benefits from lower precision without
confounding that experiment with density contraction, XC arithmetic, or
reduction precision.

A first FP32-compute/FP32-storage prototype reduced the AO-panel footprint, but
it forced the unchanged FP64 density/feature/potential contractions to convert
AO values repeatedly. On the measured resident PBE endpoint that version was
slower than strict FP64.

## Decision

Add an explicit, non-default CUDA LDA/PBE candidate named
`Fp32ComputeFp64Storage`.

The compiler lowers the same symbolic axis-derivative DAG twice. The candidate
uses FP32 temporaries, constants, coordinates, primitive exponents,
contractions, and AO values inside the AO kernel, then converts once at the AO
kernel output and stores the resident AO panel as FP64. Density products,
features, XC point evaluation, Vxc assembly, reductions, geometry, basis data,
density matrices, and scalar totals retain the existing FP64 ABI and kernels.

Strict FP64 remains the default and its AO kernel body stays unchanged. The
candidate is an internal execution capability only: it is not promoted by the
method controller and does not redefine `precision=auto`. r2SCAN rejects this
candidate until it receives a separate qualification.

## Rejected alternatives

- FP64 AO arithmetic followed by FP32 storage: this reduces bytes but does not
  test lower-precision arithmetic.
- FP32 AO arithmetic plus FP32 AO-panel storage while leaving downstream
  contractions FP64: on the RTX 5090 resident PBE endpoint this was about 14%
  slower than strict FP64 (0.8625--0.8630x speedup) because the FP32/FP64
  boundary is crossed repeatedly downstream.
- Converting density/XC/Vxc arithmetic in the same change: that mixes route B
  with routes C/D and prevents a clean ablation.
- Enabling the candidate through automatic precision selection before broader
  endpoint and force qualification.
- Enabling r2SCAN immediately: meta-GGA tail/ratio sensitivity deserves a
  separate selective-precision qualification.

## Invariants

- FP64 remains the default.
- The candidate performs real FP32 AO arithmetic; it must not silently execute
  the AO DAG in FP64 and only convert the result.
- The resident AO panel and all downstream density/XC/Vxc arithmetic remain
  FP64 in route B.
- No TF32/FP16/BF16 arithmetic is introduced.
- Unqualified precision combinations fail explicitly.
- Compiler-owned symbolic AO derivative algebra remains shared between FP64 and
  FP32 generation.

## Evidence

On node3, NVIDIA GeForce RTX 5090 (sm_120), CUDA 13.0.88, the PBE RKS native
fixture produced FP32-compute versus strict-FP64 differences of:

- XC energy: 2.2772308205798453e-09 Eh
- integrated electron count: 6.3236120229070991e-09
- maximum Vxc element: 8.2214303698258107e-09

The production candidate keeps the FP64 AO-panel allocation unchanged. For a
resident PBE RKS endpoint with `nao=12`, 49,152 grid points, tile size 256,
10 warmups, and 100 timed enqueues using CUDA events, three repeated runs gave:

- 46.851475 ms -> 45.575791 ms, 1.0279904x
- 46.813613 ms -> 45.575854 ms, 1.0271582x
- 46.857676 ms -> 45.565649 ms, 1.0283553x

Thus the isolated AO-compute change is about 2.8% faster on this case. The
discarded FP32-storage prototype measured 46.80--46.82 ms strict versus
54.23--54.29 ms mixed, about 0.863x.

sm_120 resource inspection reports 48 registers for the strict AO kernel and 40
for the FP32-compute AO kernel, with no local stack in either. This supports the
interpretation that the arithmetic kernel is not losing occupancy; the negative
FP32-storage result appears at the precision boundary with unchanged FP64
consumers.

Compiler/structure tests pass locally and the native CUDA target compiles under
sm_120. The existing r2SCAN tail-grid native test fails with device error 3 in
both this branch and its #981 parent under this CUDA 13 environment; that
pre-existing failure is not treated as evidence for or against the candidate.

## Consequences

Route B alone can retain a modest AO-compute gain without changing the resident
storage ABI. FP32 AO-panel storage should be revisited together with route C,
where density contractions can consume FP32 directly and accumulate in FP64,
rather than paying repeated conversions at an FP64 consumer boundary.

## Revisit when

Promote this candidate beyond an internal execution mode only after
representative molecule/basis/grid endpoint benchmarks and energy/force gates
are available. Re-test FP32 panel storage when route C supplies an FP32-aware
density contraction. Qualify r2SCAN separately.

## References

- #168
- #174
- #375
- #528
- #981

## Independent review qualification

The emitted FP32 AO kernel now has an allocated CUDA regression against all six
hash-checked independent libcint grid fixtures: H2, water, Cartesian and spherical
f shells, diffuse and tight primitives. All 20 spatial derivative components
through order three are checked independently at a maximum-error gate of
`5e-6` times that component's reference maximum. This componentwise norm avoids
ill-conditioned pointwise ratios at cancellation zeros and prevents large third
derivatives from hiding lower-order errors. The gate qualifies AO arithmetic,
not mixed SCF, response or force endpoints.

The six tests passed on RTX 5090 with CUDA 12.9 and `--fmad=false`, matching the
integrated native grid translation unit's contraction policy. The fixture bases
were normalized using the preserved, already qualified strict native library;
the AO kernel itself was freshly emitted and compiled from this branch. The
existing translation regression and the integrated expression/compiler suites
also passed (43 host cases).

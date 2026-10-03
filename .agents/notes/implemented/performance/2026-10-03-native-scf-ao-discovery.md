# Decision: explicit native SCF sampled-AO discovery

Status: implemented, experimental; numerical and endpoint qualification pending
Date: 2026-10-03

## Problem

The resident force caller can consume geometry-bound maps, but the native SCF
owner still supplies every AO to density and potential assembly on every tile.
The explicit compact-panel API alone does not change a complete SCF endpoint.

## Decision

An unused physical FP64 `CudaXcPlan` may explicitly discover sampled-jet maps
using its own immutable device basis/grid, the compiler's ordinary AO collocator,
and the same all-jet flag reducer used by resident discovery. Discovery performs
no density contractions or CPU AO work. It borrows the charged dense AO/work
panels and transfers only flags/errors; sorted host maps are copied once into
an appended, caller-owned device arena. A plan cannot change maps after any
physical or replay body has started, including unpublished replay work.

The SCF experiment is opt-in with `GENERATIVEQC_CUDA_KS_ACTIVE_AO=1`, admitted
only for device-fused FP64 WB97M-V, with an explicit sampled-jet cutoff of
1e-16. Unset or 0 retains dense SCF. Invalid switch values are rejected.
This is an experimental qualification control, not a promoted product default.
The native owner is rebuilt on changed basis/geometry/grid; density updates
reuse its immutable map. The force caller has a separate order-2 map and must
still pass complete energy/force gates when composed with this order-1 SCF map.

## Resources and fallback

Admission reserves one full global-AO map per tile, plus host indices, offsets,
and one tile's flags. The host setup peak is bounded by 64 MiB and conservatively
retained in the enclosing owner's host ledger. Mean selected AO counts never
reduce admission. The ordinary global resource ledger charges device storage.
Mandatory KS, VV10 and eigensolver allocations precede optional maps. A typed
device/budget OOM for the enlarged XC arena retries the original dense arena;
host registry OOM and unrelated errors propagate. Host-budget misses select the
dense route without discovery. No external library or reference engine enters
production preparation.

The optional additive C diagnostic reports the current result's actual XC
submission count and its owner's immutable preparation work. Preparation time
is lifetime setup time, not a cost to add again on every warm sample. Counters
for point/AO and point/AO-square work describe one complete XC traversal.

## Qualification

Native tests derive independent expected masks from CPU AO values, then use
full-global-AO bilinears to validate selected rho/gradient, XC energy/electrons,
and the complete potential. Cartesian and spherical f cases span >32 AOs and
129/257-point tiles, with a small cutoff, partial maps, and all-empty maps.
Resource tests cover exact bounds, host/device misses, arena canaries and
post-evaluation refusal. Real-GPU and complete endpoint results are not yet
claimed; append their exact source/library receipts after execution.

## Revisit when

Promote only after independent full cold/warm/moved-geometry energy/force gates,
constrained-resource fallback checks, actual work reporting, and matched large
endpoint timing establish a useful workload domain. The preliminary force-only
24-atom experiment did not establish an advantage over GPU4PySCF.

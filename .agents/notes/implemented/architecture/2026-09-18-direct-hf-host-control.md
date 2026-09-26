# Decision: separate direct-HF bucket lifetime from CUDA Graph lifetime

Status: implemented; production acceptance pending
Date: 2026-09-18

## Problem

After the #262--#277 SCF decomposition, `src/scf/cuda_rhf.cpp` still owned the
opaque direct-HF bucket plan, topology/options admission and invalidation,
warm-start caches, diagnostics, retry policy, CUDA Graph handles, Graph capture
and replay, and the remaining coupled numerical launch sequence. That left
changes to cache/Graph mechanics in the same review/rebuild owner as scientific
launch orchestration and kept #240's final host-control boundary unresolved.

## Decision

Move `CudaRhfBucketPlan` plus public cached-bucket lifecycle/admission into
`cuda/rhf_bucket_internal.hpp` and `cuda/rhf_bucket.cpp`. Give direct-HF CUDA
Graphs a dedicated RAII owner in `cuda/rhf_graph.*` that exclusively manages
capture, instantiate/upload, replay and destruction. Keep `CudaResources`
responsible for the stream, cuBLAS/cuSOLVER workspaces and numeric arena. Keep
`cuda_rhf.cpp` responsible for the exact numerical launch/dataflow order and
pass capture bodies to the Graph owner through a narrow host callback.

Move the basis-layout inspection and small-HF resource query with the bucket
owner because they describe host topology/resource admission rather than
scientific execution. Preserve the existing plan identity tests, cublas retry,
warm-cache semantics and split-provider Graph route.

## Rejected alternatives

- Split `cuda_rhf.cpp` by line count alone: the residual launch body shares one
  arena layout, pointer set, convergence/error order and Graph capture closure;
  a mechanical split would create a broad context object and callback facades
  without independent ownership.
- Put Graph handles into a generic resource object: Graph topology is specific
  to the direct-HF iteration transaction, while streams/library workspaces are
  reusable runtime resources. Keeping them together caused unrelated resource
  teardown edits to own HF Graph policy.
- Move scientific launch families into the bucket owner: that would re-couple
  plan lifetime to direct numerical implementation and undo the #273/#274/#277
  dependency direction.
- Rewrite kernels, tolerances or SCF logic while extracting the boundary: #240
  is structural and requires behavior preservation, not a new HF algorithm.

## Invariants

- RHF/UHF energies, forces, iteration order, convergence/failure semantics,
  screening/tolerance values and mixed-precision policy are unchanged.
- Graph capture still synchronizes once before the iteration capture, uploads
  before replay, and for the split-provider route synchronizes before the
  post-eigensolver capture and after its upload.
- Graphs are destroyed on the owning device before their stream is destroyed.
- The bucket plan remains opaque outside SCF CUDA internals; warm-state policy
  transitions preserve resident/frozen cache semantics byte-for-byte.
- Graph ownership cannot depend on bucket policy or numerical launch owners;
  bucket ownership cannot include device implementations.

## Evidence

Baseline master `91dfd97` has `cuda_rhf.cpp` at 4,892 lines / 276,146 bytes.
The current PR head has the driver at 4,452 lines / 255,625 bytes,
`rhf_bucket.cpp` at 357 lines / 16,167 bytes, its private header at 121 lines /
5,517 bytes, and `rhf_graph.cpp` / `.hpp` at 81 / 57 lines. The structure audit
checks 221 modules with zero errors and its dedicated Python suite passes 104
checks. Ordinary-C++ syntax validation passes for both new owners and the
remaining driver against the CUDA 12.9 XsyevBatched API contract.

Production acceptance is still pending on the exact PR head. The available local
node does not expose CUDA 12.9+, so this note does not infer values from generic
green CI, an older SHA, or development-only compile samples. #240 must remain
open until a suitable CUDA 12.9+ environment records all of the following on
the final SHA:

- clean Release build: pending exact-head measurement
- shared-library size: pending exact-head measurement
- device-link evidence: pending exact-head measurement
- Graph-owner incremental rebuild set: pending exact-head measurement
- bucket-owner incremental rebuild set: pending exact-head measurement
- driver incremental rebuild set: pending exact-head measurement
- native/runtime gate: pending exact-head measurement
- Python endpoint gate: pending exact-head measurement
- weighted/reference gate: pending exact-head measurement

After those measurements are captured, replace the pending entries with the
commands/results and promote this note's production-acceptance status. Until
then, this record documents the implemented ownership split only and is not a
#240 closeout record.

## Consequences

Cache/plan and CUDA Graph lifetime changes now rebuild/review independently of
the large direct launch driver, while scientific launch sequencing stays in one
place. The private plan contract adds one internal header and Graph teardown has
its own allocation-measurement lock acquisition before resource teardown. Plan
object size may change by the Graph owner's device id; the existing resource
query reports `sizeof(CudaRhfBucketPlan)` and therefore remains exact.

The remaining driver is still large, but its size is now justified by a single
coupled orchestration transaction rather than unresolved ownership mixing.
Future extractions should require a new stable state/lifetime boundary, not a
line-count target alone.

## Revisit when

Revisit if direct-HF Graph topology is shared with another method, if the
residual launch transaction gains independently reusable state, if bucket
admission becomes a method-neutral runtime service, or if profiling shows that
the new C++ ownership boundary causes measurable endpoint overhead.

## References

- Issue #240
- `docs/scf_module_boundaries.md`
- `tools/check_scf_structure.py`
- `src/scf/cuda/rhf_bucket.cpp`
- `src/scf/cuda/rhf_graph.cpp`

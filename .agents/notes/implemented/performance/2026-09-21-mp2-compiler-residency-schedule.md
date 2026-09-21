# Decision: compiler owns MP2 source-reuse and residency schedules

Status: implemented
Date: 2026-09-21

## Problem

Conventional MP2 already gained `NativeBlockProvider::get_many()` so several
MO blocks can consume one AO source traversal, while the unmerged RI-MP2 CUDA
prototype kept whitened AO-to-MO `B[Q,i,a]` values resident on the device.
Both optimizations still embedded their schedule-selection policy in native
method code. That made successful performance decisions difficult to reuse,
audit, or eventually combine with the shared compiler resource scheduler.

## Decision

`vibeqc_compiler.method.mp2_schedule` is the canonical owner of the two bounded
execution decisions:

- conventional MP2 maps provider request capacity to paired direct/exchange
  jobs that share one AO source scan;
- CUDA RI-MP2 selects complete resident `B` or the largest bounded virtual
  block plus a bounded occupied-`j` batch.

`tools/generate_mp2_native.py` emits `src/posthf/mp2_schedule_generated.hpp`
from that compiler module. Native post-HF code consumes the generated choices;
it retains allocation, CUDA library calls, scientific equations, error handling,
and exact provider resource accounting.

The RI CUDA path reuses the existing SCF DF source and metric factor owner.
Whitened source rows, AO-to-MO transforms, fitted-integral products and OS/SS
reductions stay device resident; only final scalar results and explicit
diagnostics cross the host boundary.

## Rejected alternatives

Keeping the batching and blocking formulas directly in `mp2_energy.cpp` and
`ri_mp2_cuda.cu` was rejected because it leaves compiler-visible reuse and
placement decisions as method-local handwritten policy.

Rewriting the complete RI-MP2 algorithm as a new TensorIR implementation in
this change was also rejected. TensorIR already owns generic resident execution
and last-use allocation; duplicating the scientific correlation implementation
would enlarge the validation surface. This change first moves the production
schedule boundary into compiler ownership while preserving the independently
validated native CUDA consumer.

## Invariants

- MP2 equations, denominators and OS/SS semantics are unchanged.
- A shared AO scan never changes deterministic MO job order.
- Memory remains bounded; insufficient RI residency falls back to a smaller
  virtual block or fails before execution when even one block cannot fit.
- No exact molecule, basis, AO count, GPU product name, or benchmark identity
  participates in production schedule selection.
- CPU RI remains an independent numerical oracle.
- Native runtime owns allocation/execution; compiler owns schedule generation.

## Evidence

- Compiler schedule unit tests cover sequential/shared source reuse, full
  resident B, bounded B and impossible-budget rejection.
- Generated-header reproducibility is checked with the existing MP2 generator
  contract.
- The CPU MP2 contract and gradient native tests pass on the integrated branch.
- Current-source sm_120 CUDA correctness compilation passes with AOT disabled;
  both `libvibeqc.so` and `vibeqc_mp2_cuda_status_tests` build successfully.
- Slurm job 10698 on an allocated RTX 5090 completed with exit code 0: the
  native CUDA MP2 status/OOM rollback test passed, and
  `tests/python/test_ri_mp2_cuda_residency.py` passed (1/1).
- Historical matched endpoint performance evidence for the resident
  implementation remains under `benchmarks/results/issue367-ri-mp2-gpu/`;
  the fast-compile current-source build is correctness evidence only and is not
  used for a new performance claim.

## Consequences

The #366 source-reuse mechanism and #367 residency mechanism now have one
compiler-owned scheduling boundary without replacing the native scientific
implementation. A later shared scheduler can consume the same dimensions,
resource facts and lifetimes without first extracting policy from C++.

## Revisit when

Move the method-specific schedule onto a more generic ProgramIR/TensorIR
resource candidate when that layer can express bounded runtime block loops and
provider-produced tiles without duplicating MP2 equations or weakening the
existing memory/failure contracts.

## References

Issues #366, #367, #682. The earlier source-reuse rationale is in
`.agents/notes/implemented/performance/2026-09-21-mp2-shared-ao-source-batches.md`.

Agent: ChatGPT
Model: GPT-5.6 Sol

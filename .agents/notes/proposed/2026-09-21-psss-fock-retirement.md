# Proposal: retire the dedicated handwritten pSSS Fock path

Status: proposed
Date: 2026-09-21

Implemented and qualified on 2026-09-23. This original proposal is retained for
historical rationale; the [packed Fock claim-grid decision](../implemented/performance/2026-09-23-packed-fock-claim-grids.md)
records the required scheduling repair and final 17-configuration endpoint gate.

## Problem

Issue #356 keeps the fixed-topology pSSS Fock route as a measured native
exception even though the compiler already emits the pSSS value/Fock row and
bounded streaming selects it. PR #806 reduced generated pSSS Fock to 131
registers with zero stack and zero spills, leaving endpoint qualification and
selector promotion as the remaining retirement gate.

Changing the generated mask alone is unsafe: `direct_angular_fock.cu` also
launches a dedicated pSSS worker that ignores that mask. Keeping both would
double-count pSSS Fock contributions.

## Decision

Promote pSSS in the fixed-topology generated-Fock mask, remove the dedicated
fixed pSSS worker, delete `direct_fock_psss.cuh`, and delete the handwritten
pSSS value formula from `direct_native_psss.cuh`.

Bounded streaming already selects generated pSSS Fock. Remove its legacy
pSSS-specific fallback call while preserving a shared generic order-one
fallback. Keep the native pSSS force scheduler/geometry adapter; its derivative
mathematics already comes from `generated_weighted_eri::psss_force`.

## Rejected alternatives

- Unmask generated fixed pSSS while retaining the special worker: double count.
- Keep a hidden handwritten pSSS fast path after promotion: violates #356's
  retirement objective once qualification passes.
- Generate queue/screening runtime only to reduce native LOC: it is not
  duplicate pSSS scientific mathematics.
- Add another pSSS compiler specialization: the current IntegralIR/AOT row
  already provides the multi-output value lowering.

## Invariants

- RHF/UHF pSSS Fock have one compiler-owned mathematical definition.
- Fixed/bounded selectors remain single-counted.
- Generic bounded order-one fallback remains available.
- Existing pSSS force scheduling, screening, resident reuse, and scatter remain.
- Promotion is not complete until matched Release/sm_120 endpoints satisfy the
  existing 2% structural-retirement ceiling.

## Evidence before endpoint promotion

- PR #806: 131 registers, zero stack, zero spills for generated pSSS Fock on
  CUDA 12.9.86 / sm_120.
- Focused source/codegen tests: 38 passed, 6 skipped.
- Affected pSSS AOT, fixed Fock, bounded fallback, force and HF-host translation
  units compile with NVCC 12.9 / sm_120.
- The retirement ledger drops one dedicated pSSS Fock scientific file.

## Consequences

After endpoint qualification, future pSSS value/Fock mathematics comes only
from compiler IR/lowering. Native queue/screening runtime remains independently
owned. Remaining #356 work covers the other native Fock/J-K, force and bounded
fallback families.

## Revisit

A reproducible endpoint regression beyond the retirement ceiling keeps this
proposal unpromoted and should be addressed in shared optimizer/scheduler work
rather than by retaining duplicate pSSS science indefinitely.

## References

- #356
- #806
- #832
- #682

Agent: ChatGPT
Model: GPT-5.6 Sol

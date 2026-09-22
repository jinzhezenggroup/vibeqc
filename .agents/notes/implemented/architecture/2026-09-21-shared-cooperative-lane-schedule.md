# Decision: share cooperative lane ownership across derivative consumers

Status: implemented
Date: 2026-09-21

## Problem

Issue #670 recovered the large-system one-electron derivative regression with a generated nucleus-cooperative CUDA schedule, but its launch geometry still lived as one-electron-native constants. That left the optimization as a method-local fast path rather than a compiler-owned schedule primitive. DF shell derivatives independently use the same execution idea: one logical task is owned by a lane group, lanes share bounded state, stripe an inner work axis, rendezvous, and reduce within the group.

## Decision

Introduce `CooperativeLaneSchedule` as an integral-domain compiler schedule primitive. It owns only subgroup/lane-group/workgroup geometry plus whether shared state and a group reduction are part of the execution contract. It does not own scientific equations, task meaning, shared-state contents, or numerical policy. This deliberately remains below the cross-domain schedule vocabulary from #833/PR #874 rather than creating a second global ScheduleIR.

The one-electron derivative compiler now emits a separate nucleus-cooperative execution-policy artifact. The native CUDA runtime consumes its schedule code, lane count, groups per block, block size, and schedule identity; the generated S/T/V derivative DAG is unchanged.

DF shell scheduling now wraps the same cooperative schedule contract while retaining DF-specific shared-memory accounting. Cooperative Rys derivative classes therefore reuse the same compiler ownership API rather than maintaining an unrelated lane/group record.

## Rejected alternatives

Keeping `128 threads / 32 lanes` as native one-electron constants would preserve performance but would not satisfy compiler ownership. Moving one-electron scientific derivative formulas into a generic schedule layer was rejected because schedule reuse must not duplicate or generalize method mathematics. A universal kernel shape was also rejected: consumers share ownership semantics, not their scientific loops or caches.
## Invariants

- S/T/V derivative mathematics and generated scientific source identity remain separate from schedule identity.
- DF derivative recurrence mathematics and public weight semantics are unchanged.
- Cooperative geometry must validate against explicit `TargetInfo`; no molecule, AO-count, or GPU product-name dispatch defines the schedule.
- Native runtimes retain allocation, launch, synchronization, failure, and numerical execution semantics.
- A correct bounded fallback remains available when a cooperative schedule is unsupported or unqualified.

## Evidence

- `tests/python/test_cooperative_schedule.py` validates target-resource selection, CUDA-catalog portability, and reuse by both one-electron and cooperative DF Rys consumers.
- Focused cooperative + DF shell + production-selector tests: 15 passed.
- `tools/check_compiler_structure.py`: 277 compiler modules, 0 dependency errors.
- Ruff on changed Python/compiler files: passed.
- CUDA 12.9/sm_120 compile reached and passed the changed rhf_policy.cpp, one_electron_derivatives.cu, and generated DF shell derivative consumers before the intentionally stopped repository-wide slow build.
- CUDA 12.9/sm_120 compile reached and passed the changed rhf_policy.cpp, one_electron_derivatives.cu, and generated DF shell derivative consumers before the intentionally stopped repository-wide slow build.
- Generated one-electron derivative inventory records a distinct schedule identity and `ao_pair × nuclear_center` ownership without changing the derivative IR inventory.

## Consequences

The existing #693 32-lane / four-group production geometry remains semantically equivalent while its ownership moves into compiler-generated policy. Future #459/#597 selection can replace the portable catalog schedule with target/workload-selected variants without editing one-electron scientific code. The same primitive can be consumed by additional integral/derivative lowerings and projected into the shared #833 schedule topology once PR #874 lands.

## Revisit when

Revisit the contract if a backend needs cooperative groups spanning hardware subgroups, non-power-of-two lane groups, cross-group collectives, or schedule effects that cannot be represented by lane ownership plus target/resource validation.

## References
- #670
- #682
- #459
- #597
- #833 / PR #874
- PR #693

Agent: ChatGPT
Model: GPT-5.6 Sol

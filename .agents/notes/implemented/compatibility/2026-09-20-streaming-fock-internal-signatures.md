# Decision: specialize generated streaming-Fock internal signatures

Status: implemented in PR #715; integration remains gated by required CI
Date: 2026-09-20

## Problem

Streaming-Fock source previously duplicated the internal parameter list in the
worker, CUDA entry points, and host launch packing. Non-mixed specializations
retained three unused precision arguments. Editing those lists independently
risks a declaration/call mismatch.

## Decision

`GeneratedKernelSignature` is an ordered internal declaration/forwarding
manifest. Production and profile-namespaced streaming-Fock emission consume the
same manifest for workers, kernels, and host-wrapper calls.

After capability selection, only specializations without `mixed_fock` remove
`mixed_precision_enabled`, `fp64_threshold`, and `fp32_work_count`. Mixed
specializations retain all live precision arguments. The FP64 work counter stays
present. This is capability-proven pruning, not inference from a benchmark or
an unverified textual search for an unused name.

## Public ABI and forwarding boundary

The public runtime registry/host-wrapper signature is unchanged. Its retained
parameters preserve existing callers even when an internal specialization no
longer forwards them. Wrapper expressions adapt typed primitive/position
pointers, while component-lane workers use their internal `task_head` name and
receive the public wrapper's `bra_head` argument.

Manifest declaration order and forwarding order must remain identical. Unknown
pruning requests, duplicate argument names, and unknown replacements fail
explicitly rather than emitting a mismatched call.

## Identity and invalidation

The manifest's `identity` property describes ordered C types and names; it is not
claimed as a new independently persisted cache-key field. Signature and pruning
changes alter emitted CUDA declarations/calls and must flow through the existing
generated-source/compiler-identity invalidation contract. They must not reuse an
artifact produced for a different generated source. The public registry ABI must
not be changed implicitly by internal signature specialization.

## Rejected alternatives

Independently editing declarations and launch packing would preserve the main
source of ABI drift. Pruning precision controls in mixed-capable specializations
would change live execution semantics. Changing the public wrapper ABI is not
necessary to remove dead arguments from the generated internal boundary.

## Evidence and limits

All five tests in `tests/python/test_generated_kernel_signature.py` passed in the
repair review: declaration/forwarding order, non-mixed pruning, stable public
wrapper adaptation, mixed retention, and component-lane head-name adaptation.
The separate Krylov lint repair passed its 32-test response module.

Required CUDA compilation and endpoint CI remain mandatory. These source tests
alone are not claimed as complete GPU numerical qualification. Reduced emitted
source size is not proof of a compile-time or runtime speedup; no new performance
claim is made by this note.

## Revisit when

Capabilities, precision semantics, wrapper ABI, schedule-specific argument
names, or artifact identity change. Keep legality, declaration, and forwarding
checks coupled and retain mixed/non-mixed and production/profile coverage.

## References

Issue #673 slice C; PR #715; `python/vibeqc_compiler/integral/signature.py`;
`tests/python/test_generated_kernel_signature.py`.

Agent: ChatGPT
Model: GPT-6 Astra Pro

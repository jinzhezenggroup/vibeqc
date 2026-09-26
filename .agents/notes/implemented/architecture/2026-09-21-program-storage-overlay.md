# Decision: keep physical alias/effect facts outside ProgramIR v2 (#831)

## Context

The first #831 slice introduced one shared storage analysis for ProgramIR and
TensorIR. ProgramIR v2 itself deliberately remains a serial scientific/provider
dependency graph whose buffers are disjoint logical values. Physical aliasing and
provider memory effects are execution facts and can vary without changing those
scientific dependencies.

## Decision

Add a versioned `ProgramStoragePlan` overlay bound to the immutable ProgramIR
identity. It records only explicit physical facts:

- `BufferAliasBinding` may declare a logical value as an owner, a read-only view
  of another ProgramIR buffer, or an unknown borrowed alias;
- `CallEffectBinding` may mark a provider boundary explicit or opaque;
- unspecified buffers remain independent owners and unspecified calls retain the
  existing explicit-effect assumption;
- the resulting facts are lowered into the existing `common.storage` analysis.

This avoids changing ProgramIR v2 serialization or scientific identity while
making physical storage decisions independently versioned and auditable.
## Fail-closed rules

A view must name an existing owner in the same memory space, fit within that
owner's capacity, and cannot be produced before its owner exists. Alias cycles,
unknown buffers/calls, duplicate bindings, and malformed replay all fail.
`AliasKind.UNKNOWN` is restricted to borrowed inputs. It blocks reuse for the
whole memory space through the shared analyzer. `MemoryEffect.OPAQUE` retains all
touched owners through the region boundary rather than assuming provider-local
lifetimes.

The overlay never allocates, donates, mutates, reorders, or executes ProgramIR
values. Runtime owners remain unchanged.

## Evidence

The focused fixture models a full-capacity logical view that would otherwise look
like a second materialization. The unbound ProgramIR reports a 320-byte live peak;
the explicit view overlay groups it with its 128-byte owner and reports 272 bytes,
while preserving the owner's lifetime through every view consumer. Diagnostics
publish ranges, interference, slot assignment, peak bytes, blocked spaces, and
which logical view allocations were elided.

Opaque-effect and unknown-alias tests prove the optimization reverses to
conservative retention when physical behavior is not explicit.
## Scope boundary

This is the versioned alias/effect contract requested by the previous #831 slice;
it does not claim #831 complete. No production copy is removed in this PR. A
follow-up may attach this overlay only where a real subsystem boundary proves the
same physical alias/layout and then measure endpoint resource/performance evidence.
Layout propagation, in-place donation, allocator migration, and asynchronous
leases remain out of scope here.

Validation:

```text
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_program_storage.py tests/python/test_storage_analysis.py \
  tests/python/test_program_ir.py tests/python/test_tensor_cuda_plan.py
71 passed
```

---
Agent: ChatGPT
Model: GPT-5.6 Sol

# Decision: stage a force-only psss Direct-HF retirement candidate

Status: implemented candidate, not promoted
Date: 2026-09-20

## Problem

The compiler-generated psss weighted derivative is numerically qualified but the
archived 72-run complete endpoint matrix did not establish consistent
non-regression against the retained handwritten expression. The observed
retained/generated warm ratios crossed one, so deleting the native path from that
evidence would violate the Direct-HF retirement gate.

That candidate also paid two avoidable costs which are not part of the scientific
definition:

- the generated helper emitted the scalar value and all four center derivatives
  although the Direct-HF force consumer reads only independent centers 0/1/2;
- native/generated selection was tested inside the primitive-quartet loop even
  though the selected route is fixed for the prepared batch.

## Decision

Keep `VIBEQC_PSSS_WEIGHTED` as an A/B control and keep the handwritten path as
the production default for now, but revise the generated candidate before the next
qualification campaign.

The generated low-order header now emits `psss_force` from exactly nine roots:
centers 0/1/2 times x/y/z. It retains the full `psss` helper for the arbitrary
external-weight API, which still needs its scalar value and complete four-center
result.

The native scheduler instantiates the psss primitive traversal with a compile-time
`GeneratedMath` parameter. The runtime A/B decision is made once per shell task;
the inner primitive loops contain no route branch. The generated specialization
uses an uninitialized geometry record and assigns every field reachable from the
force-only DAG before evaluation, avoiding zero-initialization of unused generic
geometry storage.

## Invariants

- The retained native psss expression remains the default until fresh endpoint
  evidence passes the existing numerical/resource/performance gates.
- Fixed, resident-bra and paged schedulers keep their current ownership and
  screening/density semantics.
- Primitive orientation, normalization, density weights and translational
  recovery are unchanged.
- The arbitrary external-weight `psss` helper remains full-result and unchanged
  in semantics.
- This slice does not claim a speedup or handwritten-science retirement.

## Requalification

Use the existing `tools/validate_weighted_eri_endpoints.py` matrix over RHF/UHF,
STO-3G/def2-SVP, batch 1/3, fixed/resident/paged queues, cold/warm and
changed-geometry replay. Compare the retained expression and revised generated
candidate from one exact build and retain the same numerical and resource gates.

If the revised generated route passes the complete non-regression gate, a follow-up
retirement slice may remove `VIBEQC_PSSS_WEIGHTED` and the handwritten weighted
psss derivative body. If it still shows a reproducible endpoint loss, keep it as a
measured performance exception and move #356 to another family.

Agent: ChatGPT
Model: GPT-5.6 Sol

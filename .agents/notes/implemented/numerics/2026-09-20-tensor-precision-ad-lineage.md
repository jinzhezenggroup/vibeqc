# Decision: retain precision qualification identity through generated derivatives

Status: implemented
Date: 2026-09-20

## Problem

The merged #635 qualification repair binds evidence to primal precision schedules
and repeated lowering. Generated JVP/VJP provenance still omitted that identity,
so different qualified primals produced identical derivative execution identities.
The explicit precision admission classifier also omitted exp and ragged reductions.

## Decision

Preserve the original v2 request/source-value qualification validation and cast-AD
oracle. Append a stable parent precision-schedule fingerprint to generated JVP/VJP
provenance and bind it in the resolved derivative schedule. Repeat this through
nested AD and later precision lowering. Strict graphs without explicit precision
provenance remain unchanged. Inheritance partitions execution/evidence identity;
it does not confer derivative qualification.

Use the canonical transcendental set for sensitivity and classify scatter_add and
segment_sum as reductions. Keep backend legality checks independent of whether a
caller supplies a qualification reference.

## Rejected alternatives and invariants

- Do not overwrite the concurrently merged v2 qualification repair or ragged AD.
- Do not place evidence references into mathematical logical hashes.
- Do not treat qualified primal evidence as proof for its gradient or response.
- Do not maintain a second incomplete list of transcendental operations.
- Keep all existing AD equations: the generated-AD change adds only an import and
  two provenance insertions, verified against the current-master blob.

## Evidence

Unqualified host FP32 exp(100) was admitted and then failed non-finite validation,
while its strict FP64 result remained finite. CUDA rejects unsupported FP32
transcendentals independently; no CUDA execution is claimed for that reproducer.
Different evidence references reproduced identical JVP/VJP precision, plan and
search-dedup identities before this repair.

Twelve isolated host regressions pass on the reconciled precision implementation:
exp admission; primal/JVP/VJP/nested-AD/re-lowering identity partitioning; JSON
replay; independently specified cast tangent/cotangent conversion in both
directions; request integrity; strict-AD provenance; ragged reduction sensitivity.
Full integration against the final master-based tree is a separate CI gate.
No new real-GPU run or endpoint speedup is claimed by this follow-up.

## Consequences and revisit

Distinct parent evidence now partitions derivative execution identities even for
identical arithmetic. This intentionally favors sound evidence boundaries over
cache sharing. Revisit only with an explicit separation between code-only caches
and qualified execution records, or a separately validated structured evidence
contract. Independent method-level derivative acceptance remains mandatory.

## References

- #635, #528, #151.
- `docs/tensor_precision.md` and the original precision identity/AD note.
- `tests/python/test_tensor_precision_review.py`.

Agent: ChatGPT (VibeQC PR review)
Model: GPT-6 Astra Pro

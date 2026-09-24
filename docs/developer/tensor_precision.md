# TensorIR precision qualification and derivative lineage

The precision APIs in [TensorIR](tensor_ir.md) and [CUDA planning](tensor_cuda.md)
keep scientific equation identity separate from qualification identity.
`PrecisionSchedule` binds its canonical request and source-value qualification
scope. Qualification references identify externally owned evidence; they do not
prove numerical validity or permit bypassing method-level acceptance gates.

Generated JVP/VJP programs retain a `precision_parent_schedule_identity` in their
provenance. `PrecisionSchedule.parent_schedule_identity` binds that fingerprint
into plan and tuning-dedup identities. Repeated differentiation and precision
lowering preserve the lineage. Identical derivative arithmetic with different
parent evidence does not alias one execution identity. Inheriting an identity
never qualifies a derivative: independent derivative acceptance and strict
method-controller audit/refinement remain necessary.

Strict programs without explicit precision provenance do not acquire a
qualification identity merely by using AD. Mathematical logical hashes remain
independent of evidence metadata.

All canonical transcendental operations, including `exp`, are sensitive.
Ragged `scatter_add` and `segment_sum` are reductions. Explicit low-precision
requests at these boundaries require qualification, and unsupported backend
combinations remain rejected even when qualification metadata is supplied.
The conservative FP32 candidate generator excludes these operations.

Cast AD uses the established arithmetic linearization, not the literal
derivative of a discontinuous rounding map. Both cast directions are tested
against independently specified NumPy dtype conversions. No new cast derivative
rule, native scientific kernel, method default or GPU performance claim is added.

See the [original precision identity decision](../../.agents/notes/implemented/numerics/2026-09-20-tensor-precision-identity-and-ad.md)
and its [AD-lineage follow-up](../../.agents/notes/implemented/numerics/2026-09-20-tensor-precision-ad-lineage.md).

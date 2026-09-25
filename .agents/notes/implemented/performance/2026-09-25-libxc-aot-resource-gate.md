# Decision: fail-closed CUDA resource eligibility for bulk Libxc AOT probes

Status: implemented
Date: 2026-09-25
Agent: ChatGPT
Model: GPT-5.6 Sol

## Problem

The bulk Libxc census already records PTXAS resource observations, while #1134
and #1137 define and persist exact build-cache identities. A compiled CUDA census
probe still must not become a package candidate merely because object generation
succeeded. Missing PTXAS fields, spills, excessive registers, stack, local memory,
or shared memory can invalidate an otherwise reproducible artifact.

## Decision

`vibeqc_compiler.xc.bulk_aot` now has an explicit `CudaResourceLimits` policy
and a deterministic `gate_cuda_resources` result. Every required PTXAS field is
mandatory. Missing observations fail closed as unavailable; exceeding any
caller-supplied bound rejects the artifact. CUDA measurements without an explicit
policy are recorded as `not-run` and `package_eligible=false`.

The census CLI accepts the five resource bounds only as an all-or-none set.
When supplied, a failed resource gate contributes to the command failure status.

This remains a compiler-artifact gate. It does not establish numerical parity,
GPU runtime execution, SCF/force qualification, or public-method admission.

## Rejected alternatives

Treating missing PTXAS counters as zero would silently promote incomplete
evidence. Hard-coding one architecture's thresholds into the compiler would mix
target policy with artifact measurement. Reusing generic integral schedule limits
would also imply a consumer/schedule contract that the scalar bulk-XC probe does
not have.

## Invariants

- Resource observations are nonnegative integers and unknown values never pass.
- Register, stack, local/shared memory, and spill limits are all explicit.
- Omitting a resource policy never grants package eligibility.
- Passing this gate never substitutes for numerical or endpoint qualification.

## Evidence

Focused unit tests cover unknown observations, every limit dimension, invalid
policies, and the default no-policy state. Repository CI remains authoritative for
the complete compiler structure, lint/type, and integration gates.

## Revisit when

A concrete packaged CUDA XC consumer owns architecture-specific resource policy.
At that point the consumer may supply versioned defaults, but the measurement
layer should remain policy-agnostic and fail closed.

## References

- #1123
- #1130
- #1134
- #1137

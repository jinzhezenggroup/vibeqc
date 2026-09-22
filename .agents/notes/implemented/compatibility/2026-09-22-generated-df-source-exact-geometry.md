# Bind generated DF response owners to exact geometry and basis values

Status: implemented
Date: 2026-09-22

## Problem

The generated-source response path has no retained host raw tensor. An empty
raw span plus a non-null source established ownership, but did not establish
that the orbital/auxiliary arguments described the same immutable source.
Matching dimensions and matching final-density tokens are not substitutes for
matching the coordinates and primitive basis used to generate raw values.

## Decision

At source construction, retain one exact compact identity per batch item and
per orbital/auxiliary system. Encode representation, atom records, shell order,
atom bindings, angular momentum and primitive exponents/coefficients as ordered
64-bit words. Floating-point fields use their exact bit representations. This
is not a hash, so no hash-collision assumption enters scientific admission.
Occupation/charge policy is intentionally not part of the DF integral identity.

Before generated force-response work, compare the supplied systems with these
immutable identities without allocation or GPU access. Reject mismatches before
borrowing raw storage, factors or metric data. Equal systems with different
allocations are valid; mutating a former input in place does not alter the saved
identity. Both orbital and auxiliary records and the selected batch index must
match. Host-materialized routes keep their existing ownership checks.

## Memory and compatibility

The identity constructor counts words first, then reserves storage once. Source
retained-host and preparation-peak ledgers include both identity-vector capacity
and every owned word buffer. Device storage and numerical kernels are unchanged.
This adds a conservative exact boundary check, not a new cache or fallback.

## Evidence

The compiled real entry-guard regression accepted stale orbital geometry before
the repair and rejects it afterward. Exact identity tests cover equal copies,
one-bit coordinate changes, atomic metadata, shell binding/order/shape, primitive
parameters, representation, source mutation and occupation independence. The
actual source, setup and force-response CUDA-facing C++ units compile against
CUDA 12.9 headers. A new CPU Release build passes 51 native tests and the focused
resource/benchmark suite; these host results are not a fresh full GPU campaign.

Agent: ChatGPT (Even-PR Review R6)
Model: GPT-6 Astra Pro

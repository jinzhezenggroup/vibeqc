# Decision: make Libxc production-domain evidence spin-aware

Status: implemented
Date: 2026-09-25

## Problem

The first production-domain profile encoded one union of case IDs and the receipt
builder formed its required matrix as every case crossed with both spin layouts.
That made unpolarized qualification require alpha/beta-specific cases such as
`spin/zero-a`, `spin/zero-b`, `spin/full-a`, and `spin/full-b`. A numerical
runner could only satisfy that contract by inventing meaningless unpolarized
interpretations, so real retained evidence must not be produced against v1.

## Decision

Version the contract to `vibeqc.libxc-production-domain-profile.v2` /
`semilocal-boundary-matrix/v2` and bind the exact cases-by-spin matrix into the
profile identity. Density, sigma, tau, and control cases remain required for both
spin layouts. Alpha/beta zero-spin, near-zero-spin, balanced-spin, and
full-polarization cases are required only for polarized execution.

The union `case_ids` remains in the detached payload for inventory/reporting
compatibility, but receipt validation and evidence identity use
`cases_by_spin`. A case that exists in the union but is invalid for the declared
spin layout is rejected before a receipt can be built.

## Rejected alternatives

- Keeping the Cartesian-product v1 contract would require fabricated
  interpretations of alpha/beta cases for a one-density unpolarized ABI.
- Silently skipping impossible rows in the numerical runner would make the
  retained receipt disagree with its versioned profile and weaken fail-closed
  admission.
- Changing matrix semantics without a version bump would allow stale v1 evidence
  to appear current.

## Invariants

- Production-domain evidence is exact, identity-bound, and fail-closed.
- Both polarized and unpolarized layouts remain required.
- Every retained row must be physically meaningful for its declared layout.
- No profile change grants runtime, SCF, force, response, or public-method
  capability by itself.

## Evidence

Regression coverage locks the v2 schema/profile payload, verifies polarized
spin-edge rows remain required, verifies they are absent from the unpolarized
matrix, and rejects a forged unpolarized `spin/zero-a` receipt row.

The authorized node3 worker was offline during this change, so repository CI is
the executable validation authority.

## Consequences

Existing synthetic v1 qualification payloads are stale by design. There is no
retained real bulk production-domain pass inventory to migrate yet, so the
version correction precedes the B1 numerical evidence campaign.

## Revisit when

A future spin representation adds distinct one-channel semantics that make
alpha/beta-specific cases meaningful for an otherwise unpolarized endpoint, or
the production-domain case taxonomy itself is replaced.

## References

- #1118
- #1120
- #1128
- #1280

Agent: ChatGPT
Model: GPT-5.6 Sol

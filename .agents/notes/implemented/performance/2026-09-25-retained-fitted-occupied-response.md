# Decision: feed retained forward B into the existing occupied response

Status: implemented
Date: 2026-09-25

## Problem
`packed-single` avoids raw generation in SCF but its default force route repeats
AO-space fitted projection b times and contracts the metric in AO-pair space.
The existing source-occupied algorithm already has the correct low-rank adjoint,
but previously required raw input and cleared the fitted view.

## Decision
Add an explicitly selected `VIBEQC_DF_OCCUPIED_RESPONSE_SOURCE=fitted` route with
`VIBEQC_DF_RESPONSE_SPACE=occupied`. It requires `packed-single`, full metric rank,
and the existing current/corrected canonical factor proof. Defaults remain
unchanged pending independent device qualification.

For D=w C C^T, B=A M^-1/2: form S_Q=C^T B_Q C in bounded auxiliary batches,
then U=M^-1/2 S using the existing eigenfactor scaling primitive. Feed U to the
existing occupied metric-adjoint and derivative-weight consumers. Charge vectors
receive the same second inverse root. The route never treats B as a raw-A lease.

The projection borrows the response's exchange interval before metric work and
writes S to the disjoint retained interval. It adds no permanent tensor or new
allocation. An insufficient panel capacity fails; it never steals a value or
force budget. B survives for future warm replay.

## Invariants
The existing owner/epoch/density/geometry, force coefficients and corrected-state
validation remain authoritative. Rank-truncated or unproved factors cannot use
this route. The raw and general-density fallback remains unchanged. Distinct
counters distinguish fitted occupancy from physical raw-source occupancy.

## Evidence
Emitted BLAS transpose/stride/ragged-tail/error contracts pass host stand-in
tests. Synthetic full-rank adjoints are checked independently. Neither is CUDA
compilation, molecular runtime qualification, nor measured performance evidence.
Small-system and 96-atom device gates remain required before promotion.

## References
#1078, #409, #1334. Agent: ChatGPT. Model: GPT-5.6 Sol.

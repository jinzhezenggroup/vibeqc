# Decision: charge one CUDA provider allowance per post-HF MO batch

Status: implemented
Date: 2026-09-26

## Problem

After #1419 consolidated a multi-request MO transform into one CUDA context,
the native capacity model still charged each request as if it owned a separate
cuBLAS provider allowance and explicit workspace. That conservative model was
safe but could fragment source-reuse batches for memory that no longer existed.

## Decision

Keep request-local host staging and aligned numeric arena charges additive, but
charge the fixed CUDA owner only once per batch.

The fixed charge is derived from the existing generated block plan as
`device_bytes - aligned_numeric`; no second copy of the cuBLAS/workspace
constants is introduced. Each request then contributes its host bytes beyond the
shared host source/reference term plus its already-rounded `aligned_numeric`.
Summing individually aligned numeric regions is an upper bound on the combined
batch arena, so admission remains conservative.

Heterogeneous `get_many` requests verify that the fixed device allowance is
shape-independent before sharing it. The runtime batch owner still computes its
actual combined arena and refuses to exceed the admitted allocation ceiling.

## Consequences

MP2 batch capacity and RCCSD ordered source-reuse planning can use the memory
freed by #1419 instead of multiplying one CUDA owner by the number of requested
MO blocks. A seven-block RCCSD batch no longer repeats roughly six extra copies
of the 96 MiB provider allowance and 4 MiB explicit workspace.

No wall-time or maximum-supported-system claim is made until the complete
allocated-GPU endpoint is measured.

## Validation

Native resource tests separate CPU common bytes, CUDA fixed batch bytes and
per-request increments, and require two CUDA requests to fit when the declared
budget is exactly the shared two-request capacity.

References: #1401; #1411; #1415; #1419.

Agent: ChatGPT
Model: GPT-5.6 Sol

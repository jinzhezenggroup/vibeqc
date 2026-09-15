# Decision: eliminate panel-driven DF-response recomputation

Status: implemented
Date: 2026-09-15

## Problem

The DF-response path could satisfy its nominal scratch budget while doing far more
scientific work than required. At 768 orbital and auxiliary AOs, a 128 MiB
response allowance split the old path into 77 auxiliary panels. The panel loop
recomputed every `R_Q = D^T A_Q D` for every panel and reread the same raw
three-center data.

The failure mode was therefore work amplification, not simply an individually
slow kernel or an insufficient memory limit.

## Decision

Treat memory and work as separate planner objectives. For this response path,
reuse already-owned resident J/K temporaries when their identity, lifetime,
stream ordering, and ownership are valid; upload the raw `A` data once; compute
each `Q` projection once; and contract the complete auxiliary response with BLAS.

Keep an explicit bounded fallback for cases where the resident-storage
preconditions do not hold. Resident reuse must not turn a bounded path into an
unbounded allocation policy.

## Rejected alternatives

- Keep recomputing projections independently inside each auxiliary panel: this
  obeys the local scratch allowance but multiplies cubic/projection work and raw
  transfers with the panel count.
- Treat a smaller panel/scratch budget as automatically safer/faster: a lower
  memory schedule can be catastrophically more expensive when it repeats source
  work.
- Require a new unbounded full-resident response allocation: unnecessary when
  compatible capacity is already owned elsewhere, and it would weaken the
  bounded fallback contract.

## Invariants

- Numerical acceptance gates remain unchanged.
- Reused buffers must match the required scientific identity/generation and have
  valid lifetime and stream/event ordering.
- Borrowed capacity is accounted without double counting.
- A bounded fallback remains available when reuse is invalid or unavailable.
- Performance qualification uses complete endpoint timing plus semantic work
  counts, not kernel timing alone.

## Evidence

For the qualified 768-AO case with a 128 MiB new response-scratch allowance:

| Metric | Old panel-driven path | Reuse path |
| --- | ---: | ---: |
| Auxiliary panels | 77 | response no longer drives repeated Q projection |
| AO projection GEMMs | 118,272 | 1,536 |
| Raw three-center H2D traffic | ~282.6 GB | ~3.62 GB |
| RTX 5090 warm energy+force | 114.804 s | 12.083 s |

The optimized path reused three already-owned resident J/K temporaries and
retained the existing numerical gates and bounded fallback.

## Consequences

Planner quality must be evaluated in terms of scientific work as well as peak
scratch. Similar nested-tiling patterns in integrals, DFT, response, or post-HF
code should be audited for outer-loop-invariant recomputation.

## Revisit when

Revisit this decision if a different data layout, streaming architecture, or
hardware memory hierarchy makes deliberate recomputation faster at the complete
endpoint while preserving the same bounded-resource and numerical contracts.
Such a change must show work counts and endpoint evidence across relevant sizes.

## References

- PR #373
- `docs/performance_engineering.md`
- root `AGENTS.md`

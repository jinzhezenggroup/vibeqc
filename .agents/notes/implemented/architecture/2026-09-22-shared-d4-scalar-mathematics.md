# Decision: shared d4 scalar mathematics

Status: implemented
Date: 2026-09-22

## Decision

Share charge scaling, coordination weights, damping and C6/ATM scalar helpers between the retained D4 evaluator and GFN2 CUDA. Preserve newer CPU aliasing and scratch-budget guards while integrating the older candidate.

## Invariants and rejected alternatives

The new header remains handwritten scientific ownership. Consolidation does not qualify a new production schedule or remove all SCC/periodic duplicate mathematics. Do not replace newer bounded CPU owners with the older patch.

## Evidence and remaining qualification

CPU fixed-charge reference tests and source-ownership checks are the local gates. GFN2 SCC/CUDA/periodic parity and endpoint performance remain required before merge.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.

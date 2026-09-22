# Decision: stationary second order executor

Status: implemented
Date: 2026-09-22

## Decision

Execute stationary second-order plans through a generic source inventory and one response solve, accepting structural adapters instead of method-name dispatch.

## Invariants and rejected alternatives

Require exact source coverage, validate directions before work, and publish no partial result after a failed source. A generic executor does not admit unsupported method Hessians.

## Evidence and remaining qualification

Stationary HVP plan and executor suites pass (37 tests). Independent complete-endpoint finite differences are still required for every new production method adapter.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.

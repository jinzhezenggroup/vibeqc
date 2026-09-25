# Decision: uniform quartet compaction

Status: implemented
Date: 2026-09-22

## Decision

Use the system grid dimension for equal-quartet-count batches and retain flat traversal for ragged inputs. This removes repeated owner lookup for homogeneous compaction only.

## Invariants and rejected alternatives

Keep per-system isolation and preserve the ragged path. This is not the general pooled task queue requested by #992 and must not close that issue.

## Evidence and remaining qualification

The 1/4/16/64-system benchmark is retained as a reproduction tool. CUDA execution, ragged/UHF failure isolation and complete endpoint measurements remain merge gates; no speedup is claimed.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.

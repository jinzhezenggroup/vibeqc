# Decision: Include cold numerical errors in rebuild evidence admission

Status: implemented
Date: 2026-09-22

## Problem

The issue-206 rebuild runner records cold, warm, and changed-geometry results,
but protocol version 2 admitted a complete record using only warm and changed
errors. Cold initialization can select a different owner or numerical path;
later correct replays do not establish cold correctness.

## Decision

Protocol version 3 computes and records the cold paired energy and full-force
errors with the existing finite-real, complete-shape validator. The overall
numerical gate includes that pair as well as every warm and changed pair.
An excessive cold error retains the raw result and returns exit status 2.

The timing boundaries, number/order of samples, convergence checks, scientific
tolerances, and production CUDA implementation are unchanged. This repair is
not a claim that the runner supplies repeated/interleaved cold timing or all
remaining issue-206 resource/performance evidence.

## Historical evidence

Version-1/2 records and their previously published verdicts remain unchanged.
Their stored cold arrays can support an explicit offline numerical re-audit,
but the original status alone is not proof of version-3 cold admission. The
separate version-2 geometry-reset timing correction remains applicable.

## Evidence

Hardware-free tests execute the real CLI with fake native/reference engines.
All warm and changed samples agree, while one cold energy or force differs by
1e-3. Both malformed campaigns incorrectly passed before this repair; the
matching-cold control already passed. The repaired gate rejects both cold
errors and preserves the matching control without accessing a physical GPU.

## References

- Refs #206; follow-up to the bounded implementation merged in PR #984.
- `benchmarks/issue206_rebuild.py`.
- `tests/python/test_issue206_rebuild_protocol.py`.
- `2026-09-22-df-rebuild-reference-protocol.md` in this directory.

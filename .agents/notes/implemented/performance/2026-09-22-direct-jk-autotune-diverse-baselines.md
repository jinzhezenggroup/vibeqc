# Decision: diversify bounded Direct J/K autotune search

Status: implemented
Date: 2026-09-22

## Problem

Direct Fock schedule search can expose hundreds of candidates for low-order shell
classes. `--max-candidates` previously selected the first N enumeration entries,
so a quick search could spend its entire budget on scalar-algebra variants of one
packed execution geometry and never measure thread/subgroup/component mappings.

The Fock baseline index also omitted manifest rows that use their primary schedule
for Fock without a separate `fock_schedule`. In addition, an already-shipped
subgroup baseline was rejected by the new-proposal subgroup promotion gate.

## Decision

Bounded searches now sample distinct execution geometries round-robin by schedule
kind before revisiting algebra variants within a geometry. The explicit candidate
limit remains per class, and the shipped production baseline is still appended
outside that budget when needed.

Fock baseline resolution uses `fock_schedule` when present and otherwise uses the
manifest row's primary schedule. The experimental subgroup gate applies only to
new proposals; an existing production subgroup mapping remains an admissible
comparison baseline.

## Rejected alternatives

Keeping prefix truncation was rejected because enumeration order is an internal
compiler detail, not a tuning priority. Random sampling was rejected because it
would make quick tuning non-reproducible. Treating every subgroup mapping as
experimental was rejected because it invalidated an already-qualified production
baseline and made same-process replacement comparisons impossible.

## Invariants

- Production baselines must be present in Fock comparisons even when outside the
  candidate budget.
- New subgroup schedules still require explicit production acceptance unless the
  caller opts into the experimental promotion gate.
- Candidate bounding must be deterministic and independent of benchmark timing.
- This change does not alter production Direct J/K execution by itself.

## Evidence

Focused Python regression coverage checks shared-schedule Fock baselines,
production subgroup admission, and geometry diversity under an eight-candidate
budget. A first RTX 5090 diagnostic run exposed the baseline defects: `ssss` and
`psss` had no production comparison, while the shipped `ppps` subgroup row was
rejected as experimental.

The corrected RTX 5090 run measured 45 Fock trials and retained a valid production
baseline for all five requested classes. Best quick-search deltas versus the
same-process baseline were about +0.01% (`ssss`), +0.12% (`psss`), +0.10%
(`psps`), and +0.41% (`ppss`); the shipped `ppps` 128-thread/8-task subgroup
mapping beat every sampled alternative. These shell microbenchmark differences
are too small to justify a production manifest change without complete SCF
endpoint evidence, so this change deliberately promotes no new kernel schedule.

## Consequences

Small tuning budgets cover more qualitatively different GPU mappings. This may
reduce the number of algebra variants measured in a quick pass; deeper passes can
raise the limit after promising geometries are identified.

## Revisit when

Replace the deterministic round-robin heuristic if compiler cost models can rank
execution geometries accurately enough to prune them before compilation.

## References

- #597 Direct J/K target/autotuning ownership
- #459 specialization/profile identity
- `python/vibeqc_compiler/integral/tuning/driver.py`

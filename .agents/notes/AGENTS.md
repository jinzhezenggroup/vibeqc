# Agent Notes

Agent Notes preserve durable engineering/scientific rationale for future humans
and agents. They complement, rather than replace, current-state documentation and
tests.

## What belongs here

Create a note when a non-trivial decision changes one or more of:

- architecture, subsystem ownership, package boundaries, or artifact identity;
- numerical algorithms, precision, derivative/response semantics, or acceptance
  gates;
- performance strategy in a way that changes algorithmic work, data movement,
  tiling, reuse, or resource planning; or
- compatibility/fallback behavior whose removal or retention needs durable
  justification.

A note is especially useful when substantial diagnosis was required or a rejected
approach is likely to be proposed again. Do not create notes for routine bug fixes,
small tests, typo/docs-only cleanup, or mechanical refactors with no durable
tradeoff.

## Layout and lifecycle

Use stable, descriptive files under one of:

```text
.agents/notes/
  implemented/{architecture,numerics,performance,compatibility}/
  proposed/
  rejected/
```

Create category directories as needed; do not maintain a central index that every
parallel branch must edit. Use `YYYY-MM-DD-short-slug.md` filenames.

Implemented notes record decisions that became repository behavior. Proposed
notes describe a decision that is not yet implemented. Rejected notes preserve a
serious alternative when remembering why it was rejected prevents repeated work.

Do not silently rewrite historical rationale after a materially different decision
lands. Add a superseding note and link both directions. Small factual corrections
and additional evidence may be appended without changing the original decision.

## Recommended format

```markdown
# Decision: <short title>

Status: implemented | proposed | rejected
Date: YYYY-MM-DD

## Problem
What made the decision necessary?

## Decision
What was chosen, including important boundaries/fallbacks?

## Rejected alternatives
What plausible alternatives were considered, and why were they not chosen?

## Invariants
What must future changes preserve?

## Evidence
Tests, work counts, numerical gates, benchmark conditions, or reproduction data.

## Consequences
Expected tradeoffs and maintenance cost.

## Revisit when
Concrete conditions that would justify reopening the decision.

## References
Issues, PRs, docs, commits, or reproducible commands.
```

Keep `docs/` focused on current truth and link to notes for historical reasoning.
Tests protect current behavior; notes protect future maintainers from repeating
already-understood mistakes.

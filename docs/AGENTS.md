# Documentation instructions

These rules apply to files under `docs/`.

## Keep docs current-state focused

Documentation should describe the system as it exists now: public behavior,
scientific contracts, ownership, supported paths, invariants, setup, and current
validation procedures.

Avoid turning current docs into a chronological project log. Historical benchmark
snapshots, migration narratives, discarded designs, and one-time debugging
findings belong in `.agents/notes/` when their rationale is worth preserving.
Link the relevant note from the current-state doc instead of duplicating the full
history.

## Update rules

- When behavior changes, update the current-state doc in the same change.
- When the reason for a non-trivial decision matters to future work, add or
  supersede an Agent Note under `.agents/notes/` and link it from the affected
  doc.
- Keep measured evidence in docs only when it is part of a current acceptance
  criterion or reproducible qualification procedure; historical measurements
  belong in notes.
- Prefer stable repository-relative links and commands over workstation-specific
  paths or transient artifact names.
- Do not silently rewrite old rationale to match a new design. Preserve the old
  note and add a superseding note when the decision materially changes.

See `.agents/notes/AGENTS.md` for the note format and lifecycle.

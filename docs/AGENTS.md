# Documentation instructions

These rules apply to files under `docs/`.

## Organize documentation by audience

- `learn/`: the minimum quantum-chemistry concepts needed to use VibeQC correctly.
- `user/`: task-oriented public behavior and workflows.
- `reference/`: authoritative lookup material such as units, names, options, and generated capabilities.
- `developer/`: architecture, numerical algorithms, compiler/code generation, implementation contracts, and extension interfaces.
- `maintainer/`: validation, performance qualification, evidence, ownership, generated artifacts, CI/release/project-health material.
- `agent/`: explanatory workflow for coding agents. Normative instructions remain in repository `AGENTS.md` files.

Keep `docs/index.md` as the stable audience router. Keep this file at the docs root so it continues to scope all documentation.

## Keep docs current-state focused

Documentation should describe the system as it exists now: public behavior, scientific contracts, ownership, supported paths, invariants, setup, and current validation procedures.

Historical benchmark snapshots, migration narratives, discarded designs, and one-time debugging findings belong in `.agents/notes/` when their rationale is worth preserving.

## Update rules

- Update current-state docs in the same change as behavior.
- Preserve non-trivial rationale in Agent Notes rather than chronological prose in current docs.
- Keep measured evidence in docs only when it is a current acceptance criterion or reproducible qualification procedure.
- Prefer stable repository-relative links and commands.
- Do not manually copy generated method/capability tables into prose; link the authoritative reference.

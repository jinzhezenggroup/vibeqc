# Generated documentation and data

Some paths under `docs/` are machine-readable repository interfaces and should not be moved merely for visual cleanliness.

Current examples include:

- `docs/public_methods.md`, generated from `manifests/public_methods.json`;
- `docs/codegen_capabilities.json`, consumed by validation tooling; and
- `docs/cuda_ownership/`, consumed by ownership/reporting tools.

Reader-facing pages should link these authoritative artifacts rather than duplicate them. A future relocation should update producers, consumers, CI checks, and docs atomically.

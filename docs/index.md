# VibeQC documentation

VibeQC documentation is organized by **reader and task**, not by the internal source tree.

## Choose your path

- **[Learn quantum chemistry](learn/index.md)** — start here if charge, spin, basis sets, SCF, DFT, gradients, or Hessians are unfamiliar.
- **[User Guide](user/index.md)** — install VibeQC and run calculations.
- **[Reference](reference/index.md)** — look up method identities, capabilities, units, and other authoritative facts.
- **[Developer Guide](developer/index.md)** — understand or extend the implementation.
- **[Maintainer Guide](maintainer/index.md)** — validation, performance qualification, evidence, ownership, generated artifacts, and roadmap work.
- **[Agent Guide](agent/index.md)** — workflow map for coding agents; repository `AGENTS.md` files remain normative.

## Generated material

The generated [public method table](public_methods.md), `codegen_capabilities.json`, the `cuda_ownership/` ledger, and the generated Libxc import/coverage reports remain at the docs root because repository tooling currently writes or consumes those paths. Guides should link to those sources rather than copying their contents.

Historical rationale and discarded designs belong under `.agents/notes/`, not in current-state guides.
Machine-readable checker inputs and repository inventories belong under `manifests/`, not in the documentation tree.

## Build locally

From the repository root:

```console
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

The generated HTML is written to `docs/_build/html/`.

```{toctree}
:hidden:
:maxdepth: 3
:caption: Guides

learn/index
user/index
reference/index
developer/index
maintainer/index
agent/index
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Generated and root reference

public_methods
libxc_bulk_import
libxc_maple_coverage
density_sources
xc_expressions
cuda_ownership/README
```

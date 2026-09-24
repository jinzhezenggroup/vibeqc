# Decision: make the common source registry own GFN1 generator inputs

Status: implemented
Date: 2026-09-24

## Problem
The GFN1 parameter and geometry generators were registered products, but each still embedded the xTBloom revision and checkout paths for `gfn1.json` and its audit manifest. That duplicated source identity outside the common registry and allowed command-line source overrides to bypass the registered product contract.

## Decision
Both GFN1 generators now bind through `load_product_sources` to `xtbloom-gfn1-parameters`, read only digest-validated registered source files, and expose registry/cache overrides rather than arbitrary source/manifest overrides. Rendering and checked-in output bytes remain unchanged.

## Rejected alternatives
Keeping the hard-coded paths as defaults would leave two scientific-source owners. Retaining arbitrary `--source` and `--manifest` flags would also permit generation from inputs that are not covered by the registered product identity.

## Invariants
The common registry owns upstream repository/revision/path/digest/license identity. The audited GFN1 manifest remains an independent compatibility gate inside each renderer. Normal generation and verification remain offline, and changing generator code requires updating the registered generator digest.

## Evidence
`python3 tools/source_registry.py verify`, both GFN1 generator `--check` commands, and focused registry/GFN1 pytest coverage pass without changing either generated product.

## References
Issue #800.
Agent: ChatGPT
Model: GPT-5.6 Sol

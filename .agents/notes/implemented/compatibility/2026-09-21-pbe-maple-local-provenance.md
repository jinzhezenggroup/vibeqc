# Decision: bind local PBE construction code into functional provenance

Status: implemented
Date: 2026-09-21

## Problem

The canonical PBE cutover moved density/spin-coordinate mapping and normalization
into `pbe_maple.py`. Hashing only `expressions.py`, upstream Maple sources, and a
manually versioned importer string did not identify those local mathematical
implementation bytes. A changed adapter could retain the previous functional
identity even though it constructed a different expression.

## Decision

For active PBE components, include content hashes for the local adapter and Maple
importer alongside the existing entry, transitive-source, and semantic-version
identities. Use content only, not absolute installation paths. Compute these
hashes when describing provenance rather than caching them with imported modules.
Non-PBE compositions do not acquire these new dependencies.

## Evidence and limits

Six regression cases (exchange, correlation, composite PBE; adapter or importer)
first fail because a changed local source leaves the identity unchanged. They
pass after binding those source hashes and also check that LDA identity is
unchanged by the PBE-specific dependency. No mathematical expression, independent
fixture, tolerance, native tail model, or public method capability changes.

References: #807; #744; #739.

Agent: ChatGPT
Model: GPT-6 Astra Pro

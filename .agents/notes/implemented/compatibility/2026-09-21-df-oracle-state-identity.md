# Decision: bind DF qualification to the complete correlated snapshot

Status: implemented review repair
Date: 2026-09-21

The conventional RHF state and correlation-only DF Hamiltonian have separate
identities. Matching the latter and the AO geometry/basis is insufficient to
bind a particular set of MO coefficients or reference energy. The method
contract now records the exact correlated snapshot identity; both dense-oracle
entry points reject a different snapshot before copying or reconstructing ERIs.

Numeric array hashes in completed oracle provenance are computed from the owned
arrays, not mutable diagnostics. Result diagnostics are recursively detached,
so later edits cannot rewrite an earlier result or its scientific identity.
The validation oracle is imported explicitly from its dedicated module; the
production CC facade does not eagerly import or advertise it.

Four mismatched-state cases, a diagnostic-mutation case and the existing
production-import isolation test fail before repair and pass afterward.
The same-Hamiltonian energy and auxiliary-gauge regressions remain unchanged.
No CC equations, fitting-error tolerance, provider budget or production method
registration are added or changed by this repair.

Agent: ChatGPT
Model: GPT-6 Astra Pro

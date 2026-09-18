# Decision: Qualify individual heavy-element ECP records and complete endpoints

Status: implemented
Date: 2026-09-18

## Problem

The scalar ECP engine accepts an element-independent Gaussian parameter schema.
Synthetic angular extensions and Na fixtures do not establish that physical
heavy-element records work with tight/diffuse primitives, large removed cores,
effective charges, SCF occupations, and complete force/resource contracts.

## Decision

Add an independent qualification slice for the unmodified LANL2DZ Rb and Cs
records distributed with the pinned PySCF 2.14.0 test dependency, with STO-3G H.
Their 28/46-electron cores leave ionic charge 9. Neutral singlet hydrides and
singly charged doublet cations exercise 10/9 explicit electrons, direct
RHF/UHF, and the existing 13-AO CPU/CUDA budgeted path. Pin canonical parameter
hashes and load tables only in test code, preserving source/license provenance
without redistributing numerical tables or adding an online runtime lookup.

The qualification adds tests, a reproducible evidence driver and documentation.
It does not alter production arithmetic, quadrature tolerances, capability
limits, schedules or the independent CPU oracle. It is independent of the
separate nonlocal-f extension in PR #446: the chosen records use local label f
and nonlocal s/p/d projectors. The branch starts from master after #442.

## Acceptance boundaries

- Test each local/nonlocal matrix against independently selected Libcint blocks,
  not only their sum, to expose compensating errors.
- Check the existing two-grid convergence contract and all physical atom/xyz
  derivatives at two displacements, including nonsymmetric AO contractions.
- Compare complete neutral RHF and charged UHF energies/forces with PySCF.
- Check complete-energy directional differences, changed-geometry replay and
  restoration under declared resource budgets; inspect the CUDA ledger.
- Preserve every unsupported-method boundary and make no broad heavy-element,
  parameter-family, geometry-domain or performance claim.

## Rejected alternatives

Raising a global element capability flag from a small synthetic fixture would
confuse parser support with chemical/numerical qualification. Importing PySCF
at runtime would create a hidden production reference dependency. Weakening
quadrature or force gates merely to accept heavier records would change the
scientific contract; this slice uses the established gates unchanged.

## Evidence and revisit conditions

`tests/python/test_ecp_heavy.py` is the acceptance suite;
`tools/qualify_ecp_heavy.py` records identities, endpoint errors and resources.
The measured evidence is in `benchmarks/results/ecp-heavy-171/`.
No generated or handwritten CUDA scientific/runtime LOC changes are involved.
Future element families, angular additions, molecular environments or method
combinations require their own pinned parameters and complete endpoint gates.
Revisit scheduling or quadrature only with evidence identifying a deficiency.

Refs #171; see [the current ECP contract](../../../../docs/ecp.md).

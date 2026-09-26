# Decision: qualify bounded Stuttgart RLC Na/K endpoints

Status: implemented
Date: 2026-09-18

## Problem

LANL2DZ evidence does not establish correctness for a different ECP parameter
family. Stuttgart RLC adds a useful boundary: the local residual vanishes while
the effective-charge Coulomb tail and signed nonlocal channels remain active.

## Decision

Qualify the exact PySCF 2.14.0 Stuttgart RLC Na/K orbital/ECP records, paired
with STO-3G hydrogen, on direct RHF/UHF CPU/CUDA. Use neutral singlet hydrides
and singly charged doublet cations. Keep parameter import confined to test and
qualification code. No production formulas, dispatch limits or schedules change.

PySCF removes zero coefficients from the local channel. The test converter
restores the published zero UL term to preserve the highest/local channel in
the existing owned schema. It does not add or remove a physical potential.
Parameter hashes describe the installed reference records before conversion.

## Rejected alternatives

Dropping the empty local record would silently make the d projector local.
Reusing LANL2DZ numerical results would not qualify this parameter family.
Neither raw-matrix agreement alone nor a converged SCF alone establishes
complete forces or replay/resource correctness.

## Invariants and acceptance gates

- Core counts 10/18 leave ionic charge +1 and molecular electron counts 2/1.
- Local residual matrices and derivatives are exactly zero; nonlocal matrices
  are nontrivial and checked separately against Libcint.
- Matrix errors and grid differences: 2e-9 absolute; derivative grid
  differences: 2e-8. All-center reference finite differences use two steps.
- Complete HF energy error: 2e-8 Eh; force error: 2e-6 Eh/bohr.
- Budgeted geometry replay preserves results, complete-energy directional
  derivatives and valid CUDA allocation-ledger bounds.
- Unsupported methods, formats and angular domains retain existing guards.

## Evidence

`tests/python/test_ecp_stuttgart.py` contains the numerical and resource gates.
`tools/qualify_ecp_stuttgart.py` records complete endpoints and exact identities.
See `benchmarks/results/ecp-stuttgart-171/README.md` for measurements, source
provenance and reproduction. Single-call timings are not performance promotion.

## Consequences and revisit conditions

This is bounded evidence for these states and geometries, not general Stuttgart
support or a relativistic method beyond the supplied scalar potentials.
Additional elements, families, basis records, difficult geometries and methods
need their own explicit gates. Revisit the zero-channel conversion only if the
owned schema or external reference representation changes.

## References

- Issue #171; current contract in `docs/ecp.md`.
- PySCF 2.14.0 `pyscf/gto/basis/stuttgart_dz.dat` and its original references.
- PySCF `parse_nwchem_ecp.py` zero-coefficient filtering.

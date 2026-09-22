# Decision: preserve explicit-grid offsets in GPU4PySCF comparisons

Status: implemented
Date: 2026-09-23

## Problem

GPU4PySCF 1.8.1's default block iterator skips grid blocks with no surviving
AOs. Its full-grid density consumer advances by yielded blocks, so skipping
an interior block associates later densities with the wrong weights and leaves
the tail uninitialized. Strict ordering alone reaches zero-size AO kernels.
For LDA, that strict empty branch also returns an extra component axis.

## Decision and invariants

`benchmarks/_gpu4pyscf_grid.py` wraps only the comparison engine's integrator.
Force strict ordering and represent empty AO blocks by one identically zero AO
row with index zero. Preserve every coordinate, weight and derivative component;
remove the empty LDA branch's extra singleton axis. The adapter is idempotent
and accepts positional or keyword block-loop arguments. Native production has
no dependency on this helper or on GPU4PySCF.

Dropping grid points, changing cutoffs, or relaxing numerical gates would change
the comparison and is not an acceptable workaround. Installed dependencies are
left intact. Fail explicitly if a future iterator lacks the strict-order API.

## Evidence

Twelve allocated RTX 5090 regressions passed (4.74 s): LDA, PBE and r²SCAN;
RKS and UKS; direct and DF. A constructed grid includes active blocks, interior
empty blocks, an empty tail and a final 13-point partial block. Independent
PySCF density, XC energy, XC potential, electron count, Coulomb energy and
complete effective-potential comparisons pass at unchanged 2e-10/2e-9 gates.
The test asserts that empty blocks actually occur and that no points disappear.

Revisit when upstream handles empty blocks correctly throughout both density
and potential consumers. Keep these regressions when removing the adapter.

References: #1079; `tests/python/test_gpu4pyscf_explicit_grid.py`.

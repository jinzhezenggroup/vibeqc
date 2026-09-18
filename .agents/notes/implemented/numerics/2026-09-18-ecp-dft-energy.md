# Decision: enable scalar ECP AO consumers for semilocal KS energies

Status: implemented
Date: 2026-09-18

## Problem

The shared CPU/CUDA one-electron providers already add checked local/nonlocal
ECP residuals to the effective-charge Hamiltonian. KS already consumes those
providers, core-adjusted occupations and nuclear repulsion, and its resource
inventory includes the ECP setup workspace. Public KS execution nevertheless
fails in Python because the ECP basis preflight rejects every AO request.

## Decision

Allow ECP AO values and first spatial jets through the existing per-operator
capability boundary. These are Gaussian valence basis evaluations, independent
of the local/nonlocal potential. Retain scalar format/angular/core validation
and the current order limit. Reuse the established KS and ECP providers without
adding a separate ECP-specific XC formula, SCF engine or native adapter.

## Invariants

- ECP residual, effective-charge attraction and nuclear terms are counted once.
- XC uses valence density; no implicit nonlinear core correction is introduced.
- Element identity remains the physical atomic number for the grid.
- LDA/PBE RKS/UKS remain energy-only methods. Complete DFT forces depend on #163.
- Unsupported formats, higher jets/angular momenta and density fitting remain
  subject to their existing explicit gates.
- PySCF/Libcint/Libxc are independent test oracles, never a CUDA execution step.

## Evidence and limits

`tests/python/test_ecp_dft.py` checks installed LANL2DZ Na/STO-3G H neutral and
cationic endpoints in both representations and backends, with two reference
initial guesses and matched quadrature. Tests compare separate physical energy
components and spin populations, not just a potentially cancelling total error.
They also cover exact-budget mixed batches, geometry rebuilds and failure
isolation. `tools/qualify_ecp_dft.py` records source and library identities.
The [retained qualification](../../../../benchmarks/results/ecp-dft-171/README.md)
records 12 passing tests per backend, 16 independent energy endpoints and
zero Sanitizer errors. The native library reuse is bound to 604 unchanged
build inputs and exact prior binary hashes.

The slice qualifies the specified semilocal energy endpoints; it does not close
#171 C2, qualify DFT forces, prove XC grid convergence or establish universal
ECP-family/functional coverage. Native scientific LOC and production formulas
are unchanged. No performance claim is made.

## Rejected alternatives and revisit conditions

Removing ECP metadata before AO evaluation would obscure invalid potential
records and break model identity. Duplicating ECP integration in KS would risk
double counting a term the common provider already supplies. Broadly enabling
all AO derivative orders would imply untested downstream consumers. Revisit
the higher-jet and force boundaries only with the complete #163 derivative
chain and independent ECP method/geometry evidence.

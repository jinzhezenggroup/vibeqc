# Decision: admit complete CPU stationary derivative work before compilation

Status: implemented
Date: 2026-09-20

## Problem

The CPU stationary diagnostic bounded its tiles but still enumerated arbitrary
ordered AO quartets and ran two ECP quadratures before exposing a work ledger.
Tile bounds alone do not make an endpoint practical or qualify public CPU forces.

## Decision

Use metadata-only admission before derivative compilation, provider calls and
contractions for both native and reference selectors. Limit primitive records,
XC points, grid pair traversals plus center-pair validation, and ECP triangular
AO-pair quadrature samples. Keep the existing CPU provider's independent fixed
160/32 and 224/44 grid policy; a test reads its native calls to detect policy
drift instead of importing the generated CUDA policy into the CPU oracle.

The primitive record count is exact for the current unscreened enumeration and
is checked against actual execution before result publication. The ECP count
is conservative because native radial shells with zero potential may be skipped.
The explicit small dense-provider domain also caps primitive and radial-term
preparation. Work units are semantic loop visits, not floating-point operations.

## Rejected alternatives

Do not promote public CPU forces merely because the diagnostic now rejects
expensive work. Public endpoint admission still needs a complete host-array
inventory, resource-plan reservations, batch failure/cleanup qualification and
an explicit production contract for its independent CPU ECP provider. Do not
silently use CUDA grid constants as the independent CPU provider's policy.

## Invariants and evidence

`test_stationary_cpu_work.py` independently enumerates mixed contraction records
and checks both native quadrature calls, both grid schedules and domain gates.
`test_ecp_stationary_cpu.py` checks all four LDA/PBE RKS/UKS methods against
independent PySCF full-response gradients (1e-7 Eh/bohr) and multistep reconverged
energy differences (2e-7 Eh/bohr), for Cartesian and real-spherical s/p records.
Both selectors reject one unit below each declared bound before compiler/provider
execution, recover at exact equality and still reject revoked snapshots.
`test_dft_complete_cpu.py` retains the all-electron analytic/replay/failure gates
and verifies zero ECP work with an ECP budget of one.

## Consequences and revisit conditions

Existing callers above the new defaults must supply explicit larger work budgets,
up to 2**40 per counter, while the ECP dense domain stays fixed. This change does
not bound SCF preparation, snapshot export, process RSS or compile time. Revisit
the count model when screening, batching, provider grids or primitive enumeration
changes. Public CPU, higher angular ECP and performance promotion remain #171 work.

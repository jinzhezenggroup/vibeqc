# Decision: promote bounded Python CUDA ECP first forces

Status: implemented
Date: 2026-09-20

## Problem

The v5 ECP snapshot and complete nine-source CUDA diagnostic were qualified,
but Python public force requests still rejected every ECP record. Merely
removing that guard would silently promote spherical and higher-angular domains
and would leave the public force sign, batch lifecycle and failure paths untested.

## Decision

Reuse the existing prepared energy owner, compiler plan and two-grid ECP provider.
Promote only Cartesian s/p ECP basis records for Python CUDA LDA/PBE RKS/UKS.
Keep native-C, CPU and wider ECP capabilities unchanged. Preserve default numeric
host/device bounds and semantic work caps from the shared consumer. Expose its
successful item work records separately from the native SCF resource ledger.
The mathematical equations and numerical tolerances are unchanged.

## Evidence and consequences

`test_ecp_public_cuda.py` requires real-device independent analytic and energy-FD
oracles for all four methods, exact-byte-budget mixed batches, cold/warm/rebuilt
owners, failed-item isolation, work admission and recovery. Host-side tests keep
serialized records, CPU, spherical and d-shell capability boundaries explicit.
The real physical qualification is LANL2DZ Na / STO-3G H, not arbitrary ECP data.
The force is the negative gradient; energy-only selection remains explicit.
No performance or complete-residency claim follows from this endpoint.

## Revisit

Broader element/angular/representation and CPU routes need their own complete
force and resource gates. Keep #171 open for those distinct acceptance domains.

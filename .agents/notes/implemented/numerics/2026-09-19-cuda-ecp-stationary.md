# Decision: bind CUDA ECP to the complete stationary diagnostic

Status: implemented
Date: 2026-09-19

## Problem

The existing seven-source CUDA diagnostic cannot differentiate an ECP energy:
bare nuclear charges and absent local/nonlocal terms change the Hamiltonian.
CUDA wire v3 does not bind ECP parameters, even though its energy owner can use ECP.

## Decision

Introduce ECP-only CUDA wire v5, extending v3 with the actual owner's ECP records.
Preserve CPU v2/v4 and all-electron CUDA v3 bytes. Re-read the live owner behind
its token when exporting derivatives. Reuse the generated native CUDA ECP
provider with its existing two-grid convergence check; CPU remains an independent
oracle. Reuse the shared nine-source TensorIR plan for ECP contractions and the
complete reduction. Effective charges feed the existing attraction and nuclear
primitive programs.

## Rejected alternatives

Inferring an ECP from occupation/core counts cannot distinguish parameter sets.
CPU derivative fallback would violate the CUDA execution contract. Reimplementing
ECP contraction formulas would create a second scientific implementation.
Advertising public forces before endpoint/resource qualification would inherit
support that this diagnostic does not establish.

## Invariants and scope

Every publication checks the same live owner before and after computation.
No partial output survives failure; later transactions can recover. No public
force capability changes. The dense export has explicit numeric bounds and a
small domain. TensorIR executes all density weights and scientific reductions
on CUDA. Native ECP quadrature remains the already validated generated provider.
Its host export and repeated final-state validation are reported boundaries;
source-kernel counters are not totals for the ECP provider.

## Evidence and revisit condition

`tests/python/test_ecp_stationary_cuda.py` is the explicit real-device gate:
four LDA/PBE RKS/UKS analytic/finite-difference comparisons, raw CPU-oracle
comparisons, same-core/different-parameter identity, stale/closed-owner rejection,
budget and failure recovery, and four all-electron regressions. Exact commit,
library hashes and measured errors are retained with PR qualification evidence.
Revisit dense staging only when a device-resident ECP source-consumer interface
has complete endpoint/work/resource and independent numerical qualification.

Refs #171, #163; CPU prerequisite #584. Public forces are a separate capability.

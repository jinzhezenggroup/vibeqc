# Decision: share migrated D4 mathematics without claiming standard DFT-D4

Status: implemented
Date: 2026-09-19

## Problem

xTBloom has native CPU/CUDA D4, but copying its entire runtime would import
GFN2 SCC state, periodic infrastructure, device ledgers and a separate
allocation/error framework. Changing only Mulliken charge input to EEQ also
fails: reference charges and reference polarizabilities are charge-model
specific.

## Decision

Extract the existing molecular CN, reference interpolation, charge scaling,
pair/ATM and first-derivative mathematics into a single host/device
qualification baseline. Keep the GFN2 reference profile explicit and reject
EEQ. Retain exact source provenance, notices and losslessly packed C6 data.
Use caller-owned linear scratch and commit outputs only after validation.

The source's charge-scaling derivative has a 0/0 at q+Zeff=0. The extracted
helper uses the continuous saturated limit (scale=exp(3), derivative=0).
Signed s8 is accepted; a positive-damping check valid only for the GFN2
parameter set must not become a generic dispersion restriction.

## Rejected alternatives

- Copying the full CPU and CUDA files would duplicate thousands of lines of
  unrelated SCC/periodic/state machinery.
- Calling libdftd4 in production would not be a native migration.
- Calling the fixed-charge gradient a complete force would omit charge response.
- Marking CPU/CUDA parity as an independent oracle would preserve shared bugs.

## Invariants

No new public method/gradient/Hessian capability is enabled. Energy and
Cartesian/charge derivatives remain explicit. Parameter semantics and table
profile must stay aligned. No release or benchmark archive is published.

## Evidence

Native CPU and real CUDA tests cover energies, pair/ATM separation, multiple
finite-difference steps, translation/permutation symmetry, all 86 table
elements, the saturated charge boundary, empty/singleton members and errors.
The CUDA tests include concurrent packed unequal-size members and isolation
of a poisoned member. Independent upstream fixture provenance is retained
with the oracle-generation tooling.

Qualification run: GCC 11.4 CPU CTest passed; native sm_120 CUDA 12.9.86
CTest passed on RTX 5090. GCC ASan/UBSan passed. The independent five-case
DFT-D4 4.2.0 fixtures bound observed CPU errors by 1.7e-19 Eh (energy),
1.4e-19 Eh/bohr (gradient), and 2.2e-19 Eh/e (charge derivative); corresponding
native CUDA maxima were 4.4e-19, 9.9e-19 and 3.3e-19. These are errors on the
small fixed-charge fixture set, not general method accuracy or performance
claims. Seventeen provenance/ownership Python tests passed; full-repository
QC regression and production D4 endpoints were not run or enabled.

## Consequences and revisit conditions

The bounded scalar CUDA schedule is intentionally not performance promoted.
The handwritten migrated mathematics is an explicit reference/qualification
exception to #159's compiler-first rule, not a second permanent production
derivative stack. Production integration must lower reusable correction
primitives through #396 and the shared scalar/TensorIR/implicit-solve rules,
use independently qualified EEQ tables and charge response, and validate
method-specific manifests. Retain this baseline as an oracle or retire it
when generated execution has equivalent independently validated coverage.

## References

Refs #493; downstream #172; related #396 and #492.

Agent: ChatGPT
Model: GPT-6 Astra Pro

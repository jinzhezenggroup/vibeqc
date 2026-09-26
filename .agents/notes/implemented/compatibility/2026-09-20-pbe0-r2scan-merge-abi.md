# Decision: preserve merged r2SCAN interfaces when integrating PBE0

Status: implemented
Date: 2026-09-20

## Problem

PR #618 and the merged r2SCAN registration independently allocated public method
IDs 10/11 and incompatible private `vibeqc_xc_point_batch_v2` signatures. Selecting
one side of these conflicts would silently relabel methods or corrupt the ctypes
call layout. The shared UKS evaluator and final-state model also evolved while
PBE0 introduced explicit semilocal and full-range exchange coefficients.

## Decision

Keep r2SCAN IDs 10/11 and append PBE0 IDs 12/13 in the data manifest. Regenerate
all three native/Python registration artifacts. Preserve the original nine-value
v1 point bridge and the merged eleven-value, tau-aware v2 bridge. Add v3 for
explicit semilocal X/C scales; v2 forwards with unit scales. Reject scaled LDA or
r2SCAN until independently qualified.

Thread immutable X/C scales through the shared RKS/UKS evaluator callbacks and
use the numeric functional family in the final-state identity. Preserve PBE0's
MethodIR/exact-exchange provenance and the ECP snapshot versions (CPU/CUDA v4/v5,
CPU hybrid v6, and hybrid+ECP v7).

## Boundaries and rejected alternatives

Do not hand-edit generated registration outputs or repurpose published IDs.
Do not reinterpret the tau-aware v2 symbol as the old branch's scaled-PBE ABI.
r2SCAN remains energy-only at the public stationary-gradient consumer boundary.
The newly merged native CPKS path remains unscaled CPU LDA/PBE only; PBE0 must
fail closed rather than using a pure-PBE kernel or Coulomb-only response.

## Evidence

`tests/python/test_pbe0_r2scan_integration.py` covers both registrations, native
snapshot family IDs, v1/v2 wire equivalence, scaled-PBE component linearity, and
CPKS rejection of PBE0. Existing native DFT/UKS/final-state tests and Python
stationary-gradient/options regressions cover the shared numerical paths.

## Integration correction (2026-09-20 final review)

The preceding allocation described the earlier merge. Current master also
publishes RCCSD as ID 12, so final integration preserves r2SCAN IDs 10/11 and
RCCSD ID 12, and appends PBE0 RKS/UKS as IDs **13/14**. Generated registration
artifacts and provider-set tests follow that manifest; no published ID is reused.

The energy-only r2SCAN statement above is also superseded by the subsequently
qualified stationary-gradient paths. Preserve those paths and snapshot support.
The native point bridge exposes the AO kinetic coefficient `vtau/2` under
`kinetic`, not a raw `tau` derivative. Wire compatibility tests use that contract.

The response lease is now shared by RKS and UKS. Its existing selector gate owns
spin admission; the added unit-X/C, zero-K gate must apply to both spins rather
than accidentally restricting the shared lease to RKS. Pure LDA/PBE RKS and UKS
remain valid, while PBE0, r2SCAN, and custom PBE50 response remain fail-closed.

Final CPU evidence: 44/44 native tests and 233 Python tests pass, including
independent PySCF and reconverged finite-difference checks. Four explicitly
GPU-gated tests are skipped in the CPU allocation, not scientific oracle tests.

Refs #618, #620, #649, #165, #164.

Agent: ChatGPT
Model: GPT-6 Astra Pro

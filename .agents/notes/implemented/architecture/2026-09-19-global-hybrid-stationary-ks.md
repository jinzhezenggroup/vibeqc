# Decision: execute global-hybrid KS and first gradients from one MethodIR composition

Status: implemented
Date: 2026-09-19

## Problem

The fixed-density #531 slice proved that `ExactExchangePrimitive` could lower to the
common J/K provider, but self-consistent KS still rejected every K term and the
stationary-gradient plan only admitted a semilocal primitive. Adding PBE0-specific
SCF or force formulas would have made energy, Fock and geometric derivatives own
separate copies of the hybrid coefficients, defeating #396's method-composition
boundary.

The native KS options ABI also had to carry the resolved scientific composition
without changing the established LDA/PBE prefix or silently enabling unqualified
CUDA hybrid execution.

## Decision

Use the resolved MethodIR as the scientific coefficient authority and lower its
supported global-hybrid composition into three explicit native coefficients:
semilocal exchange scale, semilocal correlation scale and the raw Fock K
coefficient.

For PBE0 these are `0.75`, `1.0`, and `-1/8` for RKS or `-1/4` for UKS.
The K coefficient is derived by `ExactExchangePrimitive.fock_coefficient`, the
same rule used by the fixed-density execution path. Native CPU KS consumes those
values through the common `FockBuildSpec/FockPlan`; it does not inspect the PBE0
name when performing scientific arithmetic.

The `vibeqc_ks_options` ABI keeps the version-one prefix and appends a version-two
composition suffix. Pure LDA/PBE can retain the old prefix. A named PBE0 C-API
request must provide the explicit audited composition suffix rather than letting
C++ reconstruct 0.75/0.25 constants. CUDA rejects non-unit/scaled/global-hybrid
composition until separately qualified.

The stationary-gradient compiler admits an optional full-range
`ExactExchangePrimitive` and adds one `exact_exchange` source. For each ordered
ERI `(ab|cd)`, the generated source weight contracts same-spin
`D[a,c] D[b,d]` with `cK/2`; alpha/beta cross-spin exchange is absent by
construction. CPU runtime only binds tuple-indexed state and the existing
four-center ERI first-derivative provider.

CPU composition snapshots use wire version 6 for an all-electron non-unit composition
and version 7 when the same composition is paired with an ECP-bound state; both record
the actual X/C/K coefficients. Pure all-electron CPU snapshots retain wire version 2
and the pre-existing ECP CPU snapshot remains version 4. This prevents a PBE0 stationary state from being reinterpreted as pure
PBE during derivative assembly.

## Rejected alternatives

- A PBE0-specific KS driver or force assembler: duplicates coefficients and blocks
  the second-global-hybrid extension criterion.
- Scaling the complete PBE XC value by 0.75: incorrectly scales PBE correlation.
- Reusing total-density Coulomb derivative weights for K: introduces wrong AO
  index pairing and, for UKS, spurious cross-spin exchange.
- Inferring PBE0 coefficients inside native C++ from the method enum: duplicates
  the audited MethodIR manifest and lets the C API drift from compiler semantics.
- Advertising CUDA/global-hybrid forces because the shared source graph can emit
  CUDA code: source generation is not endpoint qualification.
- Pretending B3LYP is a PBE-based composition. The repository does not yet contain
  audited B88/LYP/VWN primitives, so B3LYP remains fail-closed until those
  expressions and independent fixtures are added.

## Invariants

- MethodIR owns physical hybrid fractions; Fock providers receive only resolved
  raw J/K coefficients.
- Energy, Fock and exact-exchange first-gradient weights derive from the same
  `ExactExchangePrimitive`.
- PBE exchange and correlation scales are separate throughout the native point
  model and RKS/UKS integration.
- UKS exact exchange contracts same-spin density matrices only.
- Legacy pure LDA/PBE KS option and CPU snapshot prefixes remain valid.
- A non-unit/global-hybrid composition is CPU-only until a CUDA endpoint is
  independently validated.
- Public DFT force properties remain disabled until #163's endpoint qualification;
  the complete PBE0 gradient here is an internal stationary diagnostic.
- B3LYP must not be registered until its exact versioned semilocal primitives are
  implemented and audited.

## Evidence

Fresh CPU Release build with tests enabled compiles the complete tree.

Independent PySCF 2.14.0 comparisons use the same AO basis, explicit native atomic
grid and full grid response:

- asymmetric-water PBE0 RKS: energy error `4.3e-14 Eh`, maximum total-gradient
  error `5.7e-12 Eh/bohr`;
- asymmetric open-shell PBE0 UKS: energy error `1.4e-14 Eh`, maximum
  total-gradient error `1.34e-11 Eh/bohr`;
- both RKS and UKS gradients recover translation to approximately `1e-14
  Eh/bohr`;
- reconverged changed-geometry directional finite differences are gated at
  `1e-6 Eh/bohr`;
- independent TensorIR tests verify RKS `-a_x/4` and UKS `-a_x/2` ordered-ERI
  derivative coefficients and explicitly detect UKS cross-spin contamination.

- a data-only PBE50 extension test (0.5 HF + 0.5 PBE-X + PBE-C) runs through
  the same CPU SCF, snapshot and compiled stationary-gradient path and matches
  an independent PySCF calculation at the established energy/gradient gates,
  without a new named-method scientific branch.

The C ABI regression also verifies that a named PBE0 request without the v2
composition suffix fails closed and that a wrong restricted raw-K coefficient is
rejected.

## Consequences

A second global hybrid can reuse the same SCF/J-K/stationary-gradient execution
path once its semilocal primitives are available; it does not need another hybrid
driver. The remaining #165 work is therefore primarily the audited,
explicitly-versioned B3LYP semilocal inventory plus its independent SCF/gradient
qualification, and later CUDA/public-force promotion through their owning gates.

## Revisit when

Revisit this split when CUDA global-hybrid SCF/forces are independently qualified,
when a density-fitted hybrid derivative is promoted, or when B88/LYP/VWN enter the
audited scalar-XC inventory.

## References

- #165
- #396
- #531
- #163
- `tests/python/test_stationary_gradient_plan.py`
- `tests/python/test_dft_complete_cpu.py`
- `tests/native/test_dft_api.cpp`


## Integration correction (2026-09-20 final review)

The mathematical ownership decision above is unchanged. Its historical statements
about the missing B3LYP scalar inventory are superseded: audited B3LYP MethodIR
and B88/LYP/VWN-RPA scalar components now exist. Remaining native KS SCF/geometry
lowering and complete endpoint qualification are separate from this PBE0 slice.
Likewise, the public-force restriction here is specific to the unpromoted hybrid
endpoint, not a withdrawal of subsequently qualified semilocal force paths.

Resolved KS options must retain their MethodIR when passed from Calculator to
resource planning. Rebinding only the descriptive PBE selector loses a custom
hybrid composition. Re-resolution now preserves and revalidates the resolved
graph and canonical functional identity. Named-selector idempotence and actual
budgeted PBE50 RKS/UKS SCF are regression-tested.

Agent: ChatGPT
Model: GPT-6 Astra Pro

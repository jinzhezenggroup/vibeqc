# Decision: value-only range exchange in the common native CPU Fock provider

Status: implemented internal slice; public RSH endpoints remain unavailable
Date: 2026-09-20

## Ownership and operator convention

The common exact CPU provider now binds a full-range, short-range, or long-range
operator and its inverse-Bohr omega to immutable prepared integral storage.
Ordinary Coulomb remains full-range. A J/K plan may borrow full-range ERIs for
J and a separate range tensor for K; the provider must not reinterpret a single
tensor as two different operators. Geometry, normalized basis, spin, operator
and omega participate in compatibility checks. The prepared owner retains the
range tensor and includes its capacity in observed CPU storage.

SR and LR values use direct positive-interval radial evaluation. In particular,
SR is not formed by subtracting nearly equal full and LR tensors. For the first
CAM-B3LYP SCF consumer, MethodIR generates the semilocal formula and coefficients;
exchange is assembled as `a_SR K_full + (a_LR - a_SR) K_LR` with the existing
RKS occupation-two and UKS same-spin conventions. The positive/negative Fock
sign is owned by the common Fock coefficient, not an extra method-side sign.

## State and energy boundary

Both primary and correction plans must describe the current molecular source.
The CAM consumer requires its generated operator/omega/coefficient identity;
changed geometry, grid or omega cannot reuse an incompatible prepared source.
The physical residual and final-state machinery remain shared with native KS.
Hartree and exact-exchange diagnostics are stored separately, avoiding a label
that previously counted exchange as Hartree. Proposal orbitals remain a solver
mechanism rather than the final physical-state proof.

## Rejected shortcuts and capability limits

Do not use `full - LR` to approximate small SR integrals, silently change the
omega, substitute full-range derivatives for range derivatives, or infer CUDA
capability from the CPU registration. Range Coulomb, fitted/range providers,
range derivatives through this common provider, and CUDA RSH remain rejected.
The private explicit-grid CAM demonstration is not production-grid, public
Calculator, analytic-force, open-shell-domain or performance qualification.
Independent reference engines are tests, not production dependencies.

## Evidence and revisit conditions

The review rebuilt matching Release CPU sources: 46 native tests passed,
including direct SR+LR identity, LR contraction, RKS/UKS closed-shell consistency,
geometry/omega rejection and warm replay. The selected fixed-density/Fock/range
suite passed 121 tests; 16 device-only cases were skipped. These counts are not a
new GPU campaign or independent full molecular CAM force comparison.

Expand the boundary only after the new operator/provider and physical endpoint
have their own independent numerical, resource, replay and scaling gates.
Do not reuse the tiny explicit-grid internal test as production-grid evidence.

References: #167, #166, #570, #723; `src/scf/fock_provider.hpp`.

Agent: ChatGPT
Model: GPT-6 Astra Pro

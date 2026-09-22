# Independent atomic guess for halogen DFT force references

Status: implemented for PBE reference convergence; full #872 qualification remains open
Date: 2026-09-21

## Observation

For the pinned BrH+ PBE-UKS fixture, the independent one-electron guess did not
converge under the original physical-gradient criterion. A 600-cycle shifted and
damped run still had an orbital residual near 1e-4; a bounded second-order trial
also failed. Neither experiment is adopted. An independent atomic guess converged
to the same energy reached by independent minao and HF seeds and passed the
unchanged analytic-gradient and two-step energy-difference checks.

## Decision and limits

Allow the shared reference helper to accept keyword-only initial-guess and finite
iteration-budget controls, preserving its defaults for every existing caller.
Only the new halogen open-shell wrapper requests the atomic guess and 600 cycles.
The reference never receives the VibeQC density. The basis, ECP, molecular charge,
spin, moving grid, functional, 1e-13 energy and 1e-10 orbital-gradient criteria, and
force/finite-difference tolerances are unchanged. Retain the explicit independent
convergence assertion and record both controls in per-case evidence.

Eight CPU PBE analytic cases (Br/I, RKS/UKS, Cartesian/spherical) pass; four shared
reference/default regressions pass. The remaining matrix has ten passing cases
and six failures: four native LDA-UKS convergence failures and two iodine PBE-UKS
recovery failures. This is not complete halogen or new GPU qualification.

Agent: ChatGPT
Model: GPT-6 Astra Pro

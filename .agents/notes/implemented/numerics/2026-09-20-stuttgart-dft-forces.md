# Decision: qualify complete DFT forces with Stuttgart RLC Na/K

Status: implemented
Date: 2026-09-20

## Problem

Issue #171's complete public DFT force gates originally used LANL2DZ Na.
Raw ECP and complete HF validation of another family does not establish
method-consistent DFT gradients, XC grid response or prepared-batch behavior.
The existing Stuttgart RLC Na/K records provide a distinct physical case with
an exactly zero local residual and active s/p/d nonlocal projectors.

## Decision

Reuse the pinned independent Stuttgart fixture and the same complete CPU/CUDA
public endpoint gates. Test LDA/PBE RKS/UKS for Cartesian/spherical NaH/KH,
including serialized spherical basis records. Verify independent full-grid-
response PySCF gradients, two reconverged finite-difference steps and translation
invariance. Preserve method tolerances rather than weakening them for a family.
PBE exact-budget mixed batches exercise cold/warm execution, changed geometry,
invalid-item isolation and subsequent recovery. The NVIDIA execution explicitly
disables CPU scientific fallback entrypoints.

Production algorithms and capabilities are unchanged. CPU cases become routine
CI; opt-in GPU cases require real-device evidence. Do not inherit CUDA d-shell
support from these s/p orbital fixtures. Nonlocal d projectors are a distinct
operator domain from d orbitals.

## Invariants

- Na removes 10 core electrons and K removes 18; both ionic charges are +1.
- Neutral hydrides have two valence electrons; +1 doublets have one.
- The zero local residual does not remove effective-charge attraction or the
  nonlocal contribution from the complete DFT energy/gradient plan.
- Compare the same discrete 24 x 8 x 16 XC quadrature, without claiming continuum
  grid or basis convergence.
- Read physical coefficients only from the pinned external test installation.

## Evidence and consequences

The committed suite contains 32 full analytic/FD cases and 16 prepared-batch
cases across both backends. The retained report records the tested Git revision,
native binary identity, GPU/toolchain, observed errors, work and resource bounds.
The CUDA Na/K doublet spherical replays also pass Compute Sanitizer memcheck
with zero errors. All 48 new cases, 24 existing CUDA cases, 5 native ECP CTests
and 90 adjacent Python cases pass without skips on the retained build.

[Measurements and reproduction](../../../../benchmarks/results/ecp-stuttgart-dft-171/README.md)
describe the exact accepted physical slice. A single endpoint time is not a
performance comparison. No new compiler or native scientific formula is needed.

## Rejected alternatives and revisit conditions

Passing raw matrices or generic all-electron DFT tests alone is insufficient for
complete ECP forces. Broad family promotion from two elements is unsupported.
Extend this matrix only with independent force evidence for the additional
states/parameters; retain resource and unsupported-domain rejection.

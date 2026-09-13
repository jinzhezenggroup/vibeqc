# PR #305 spin-boundary and returned-state corrections

The review baseline was `126071574f2a05cbbeb38369e9cf685054e36247`.
The PBE integrator now uses the energy-consistent point domain documented in
[`xc_scf_domain.md`](../../../docs/xc_scf_domain.md). UKS returns the density
whose physical energy and residual were evaluated, on both convergence and
iteration-budget exhaustion; it no longer advances that density after its
convergence decision.

## Regression evidence

- The added integrated empty/near-empty-spin continuity check failed against
  the review baseline: `PBE energy jumps between empty and nearly empty spin
  densities`. It passes with the new PBE point dispatch.
- The added returned-state test, linked against the baseline library, failed:
  `UKS returned energy and density from different iterations`. It passes for
  LDA/PBE asymmetric H3, with both one iteration and a converged run. Energy,
  spin populations and physical commutators are reconstructed from the
  returned alpha/beta density matrices.
- Extreme nonzero minority densities exposed a second exchange underflow in
  `d0b5862`: the expanded independent fixture fails with `valid SCF domain
  input was rejected`. Analytic reduced-gradient/reciprocal exchange forms
  fix this without changing the functional or its spin extension. Both zero
  and nonzero spin gradients, the smallest positive FP64 minority density,
  and the numerical branch connection are covered.
- All 23 CPU native CTest cases passed, including the point-fixture case.
- Linear OH/STO-3G at 1.8 bohr exhausted 200 LDA iterations on `8f24e9a`
  despite stationary energy/residual, with density RMS fixed at `1/6`.
  The expanded returned-state regression fails on that library and passes
  after occupation stabilization. The corrected public LDA endpoint converges
  in 17 iterations. Both OH LDA/PBE tests rebuild the unshifted energy and
  residual and verify spin traces plus `DSD=D`; all original gates remain.
- Related Python validation: **146 passed, 3 skipped, 25 deselected** using
  `test_calculator.py`, `test_xc_integration.py`, `test_grid_cpu.py`, and
  `test_xc_expressions.py`, with `-k 'not cuda and not gpu'`.
- All 87 point energy/derivative records passed after regenerating the
  independent Libxc 7 / 450-digit mpmath fixtures. Fixture SHA-256:
  `d143ea506a7dbc2648327a86d790e32f2684b89f065bc49a8fba5601e6dc919f`.

## Matching-grid SCF endpoints

`tools/validate_uks_endpoints.py` used PySCF 2.14.0 / Libxc 7.0.0, STO-3G,
the same 49,152-point GridSpec-v1 grid, CPU conventional J and energy-only
execution. The native physical residual below is evaluated at the returned
density, before any subsequent density proposal is accepted.

| Case | Method | Absolute energy error (Eh) | Native physical residual RMS |
| --- | --- | ---: | ---: |
| H2- doublet | LDA UKS | 1.034e-13 | 1.334e-11 |
| H2- doublet | PBE UKS | 1.344e-13 | 8.776e-12 |
| H2+ fully polarized | LDA UKS | 2.409e-14 | 2.776e-16 |
| H2+ fully polarized | PBE UKS | 1.058e-9 | 5.551e-17 |
| OH doublet | LDA UKS | 9.948e-14 | 3.356e-11 |
| OH doublet | PBE UKS | 2.700e-13 | 3.134e-11 |

All six native and independent endpoints pass the `1e-8 Eh` energy and
`1e-9` physical-residual gates. The current clean-source run is bound to
`cf256aecd528a7f434754f7996468827ce1144d2`, with the retained raw JSON at
`/home/jzzeng/codes/vibeqc-pr305-final-endpoints-20260914.json` and SHA-256
`9669839b95612dad3f45ea7cb30a04a32a753e7143e185a774fa6632783fee9f`.
Validator SHA-256:
`0c675aae2ff28ecd3c6e2d7adaad8ea91e1d66e2f71270afe33979ef995fb9e3`.
The earlier remote run at `d0b5862` remains documented in
[`cpu-uks-endpoints-20260913.md`](cpu-uks-endpoints-20260913.md) as lineage;
the newer run additionally validates the extreme-spin and OH corrections.
The high precision point oracle explicitly
differentiates the documented spin extension; Libxc uses its own empty-spin
screening convention.

The OH independent consumer uses PySCF's exact `C2v` molecular/grid subgroup
with its own `minao` seed to resolve pi orientation; no native density or
Fock is supplied. Unconstrained PySCF PBE DIIS stalled in the nearly flat pi
rotation and did not meet the reference residual gate. The accepted symmetry
run is checked against the **full unrestricted AO commutator**, including
the omitted rotations: independent residuals are `1.114e-12` (LDA) and
`4.524e-12` (PBE). Thus no energy or residual gate is loosened to accept the
reference. The native endpoint has no imposed point-group constraint.

The current matched-grid acceptance run used a GCC 11.4 Release CPU build,
Python 3.13.9, NumPy 2.5.3, PySCF 2.14.0 and Libxc 7.0.0. The point-fixture
regeneration used Python 3.13.9, NumPy 2.5.3 and mpmath 1.4.1. These
corrections add no CUDA execution, gradient, density-fitting, batch,
grid-convergence or performance claim. Issue #162's remaining prepared CUDA
and batch scope remains open.

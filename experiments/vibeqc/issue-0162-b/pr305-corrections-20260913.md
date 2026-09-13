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
- All 23 CPU native CTest cases passed (22 library/integration cases and the
  separately run point-fixture case).
- Related Python validation: **146 passed, 3 skipped, 25 deselected** using
  `test_calculator.py`, `test_xc_integration.py`, `test_grid_cpu.py`, and
  `test_xc_expressions.py`, with `-k 'not cuda and not gpu'`.
- All 72 point energy/derivative records passed after regenerating the
  independent Libxc 7 / 450-digit mpmath fixtures. Fixture SHA-256:
  `d91a3959ca67835b464711c98ff2654710e7c8b4594fbb7207f7389bc8d1d723`.

## Matching-grid SCF endpoints

`tools/validate_uks_endpoints.py` used PySCF 2.14.0 / Libxc 7.0.0, STO-3G,
the same 49,152-point GridSpec-v1 grid, CPU conventional J and energy-only
execution. The native physical residual below is evaluated at the returned
density, before any subsequent density proposal is accepted.

| Case | Method | Absolute energy error (Eh) | Native physical residual RMS |
| --- | --- | ---: | ---: |
| H2- doublet | LDA UKS | 1.034e-13 | 1.334e-11 |
| H2- doublet | PBE UKS | 1.351e-13 | 8.776e-12 |
| H2+ fully polarized | LDA UKS | 2.387e-14 | 2.776e-16 |
| H2+ fully polarized | PBE UKS | 1.058e-9 | 0.000e+0 |

All four native and independent endpoints pass the `1e-8 Eh` energy and
`1e-9` physical-residual gates. The current clean-source run is recorded in
[`cpu-uks-endpoints-20260913.md`](cpu-uks-endpoints-20260913.md); its durable
raw JSON has SHA-256
`a9a38e9527ddaee26b4ddda75a30484ad3d9fe53e802224e4350905be1067441`.
The previous `build/pr305-uks-endpoints.json` was a local correction probe and
is not the acceptance artifact. The high precision point oracle explicitly
differentiates the documented spin extension; Libxc uses its own empty-spin
screening convention.

The current matched-grid acceptance run used a GCC 11.4 Release CPU build,
Python 3.11.16, NumPy 2.2.6, PySCF 2.14.0 and Libxc 7.0.0. The point-fixture
regeneration used Python 3.13.9, NumPy 2.5.3 and mpmath 1.4.1. These
corrections add no CUDA execution, gradient, density-fitting, batch,
grid-convergence or performance claim. Issue #162's remaining prepared CUDA
and batch scope remains open.

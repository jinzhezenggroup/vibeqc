# Decision: qualify r2SCAN empty-spin potentials on identical inputs

Status: implemented
Date: 2026-09-23

## Problem

Review of the FP32 AO candidate exposed a failure in the unrelated strict-FP64
r2SCAN default-grid UKS test. The H2 fixture has exactly zero beta density and
49,152 grid points. Its minority potential element was 2.410941123885864 on
CUDA and 2.4109061845583866 on CPU. The preserved pre-candidate library from
the compiled-execution integration reproduced the same failure. LDA/PBE,
ordinary polarized r2SCAN, energy, electron counts, and majority potentials
did not exhibit this discrepancy.

The Libxc-derived boundary mathematics floors the work density before forming
`zeta = (rho_a - rho_b) / (rho_a + rho_b)`. At an empty spin, cancellation in
`1 - zeta` amplifies harmless rounding of the occupied density. This is a
conditioning limitation of the retained Libxc 7 boundary contract, not an AO
precision regression or a CPU/device point-emitter disagreement.

## Evidence

A review-only probe captured the production CUDA AO/features from its borrowed
arena and reconstructed the CPU AO-pair contraction. The maximum relative
density difference was 9.01e-16. Independent Libxc 7.0.0 C API evaluations on
each captured input reproduced that backend's integrated minority potential.
On identical features, CPU/device point emitters and Libxc agreed at roughly
1e-14 in the integrated potential.

One actual grid point has occupied density 0.04603137593786226, sigma
0.0023299235150585004 and tau 0.029349040301134496, with all minority features
zero. Independent Libxc gives minority `v_rho = 60.04172865208638`. Changing
only rho to its next FP64 neighbor, 0.046031375937862266, changes that derivative
to 60.14451563784888. Perturbing every occupied grid density by one ulp changes
the integrated potential by approximately 3e-5 to 4e-5, independently of
VibeQC. The original endpoint discrepancy was 3.4939327477445659e-5.

The original five Libxc fixtures remain unchanged. Three additional fixtures
store the point above and both adjacent occupied densities, generated directly
with `xc_mgga_exc_vxc` for `MGGA_X_R2SCAN` plus `MGGA_C_R2SCAN` in polarized
mode. Both emitters use the existing 5e-12 relative / 1e-12 absolute gate;
CUDA additionally checks spin exchange. No PySCF density clipping is applied.

## Decision

For the exact-empty-spin r2SCAN default-grid test only, capture one full tile
of device AO/features in the test fixture. Verify AO against CPU evaluation
and features against a separate double AO-pair contraction before evaluating
the host r2SCAN point function on those identical inputs. Integrate a reference
potential from those coefficients and compare the ordinary bounded 257-point
CUDA endpoint at its unchanged `2e-11 + 2e-12 * abs(reference)` gate. Exercise
both spin orientations. Energy, electron counts, occupied potentials, vacuum,
and all other functional/spin cases retain ordinary CPU endpoint comparisons.

This changes qualification only. Production mathematics, precision defaults,
storage, admission, and transfer behavior are unchanged. The FP32 AO candidate
continues to reject r2SCAN and response execution.

## Rejected alternatives and revisit conditions

A globally relaxed potential tolerance would conceal real assembly errors.
Requiring backend equality after different floating-point feature contractions
would demand more reproducibility than the independent Libxc contract supplies.
Replacing `1 +/- zeta` by density ratios might improve conditioning, but changes
the explicitly retained Libxc boundary values; that requires a separate
scientific compatibility decision and independent high-precision qualification.
Revisit the same-input test if that boundary contract is deliberately replaced.

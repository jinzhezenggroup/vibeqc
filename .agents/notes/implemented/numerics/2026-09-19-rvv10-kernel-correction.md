# Decision: distinguish revised rVV10 from reparameterized VV10

Status: implemented
Date: 2026-09-19

## Problem

The [initial slice-A note](2026-09-19-vv10-rvv10-methodir-reference.md)
incorrectly treated rVV10 as the original VV10 pair kernel with b=6.3.
Comparing that path against PySCF `_vv10nlc` checks reparameterized VV10,
not the revised kernel in Sabatini, Gorni and de Gironcoli, Physical Review B
87, 041108(R) (2013), DOI: 10.1103/PhysRevB.87.041108, Eqs. (4)-(6).

## Decision

Keep original VV10 unchanged. For rVV10 define q_i=omega_i/kappa_i and
z_i=1+q_i*R_ij^2. The paper absorbs kappa_i^(-3/2) into each density.
With the existing ordinary weighted-density interface, restore both factors
and evaluate the effective kernel

`phi_ij = -3 / [2 (kappa_i*kappa_j)^(3/2) z_i z_j (z_i+z_j)]`.

This differs from VV10 at unequal densities even when b and C are identical.
The same local scales and homogeneous-density correction remain. The revised
kernel convention changes the rVV10 scientific identity so the old arithmetic
cannot share its cache identity. The concurrently added fixed-density potential is preserved and corrected too.
No SCF, nuclear-gradient or CUDA capability is enabled.

Reject complex arrays before FP64 conversion. Trap invalid/divide/overflow
arithmetic and reject nonfinite outputs instead of publishing NaN or silently
dropping imaginary coordinates, weights, densities or gradients.

## Evidence and preserved history

Eleven new wrong-kernel/complex/nonfinite regressions fail before repair. The
corrected fixed-grid rVV10 energy is 0.0018146943496255286 Eh, versus
0.0018146943496255283 Eh from an independent 60-digit scalar Decimal evaluation
of the published density-rescaled equation. The original VV10 fixture remains
0.002000475442872441 Eh. The historical 0.0018147083462731767 Eh value remains
explicitly tested as VV10 at b=6.3, and is rejected as the rVV10 reference.

The new tests compare tiled results with the separate Decimal equation and
require rVV10 to differ from equal-parameter VV10 on unequal-density pairs.
The original symmetry, permutation, tiling and weighted-density checks remain.
These are fixed-grid equation/oracle checks, not a production periodic FFT,
SCF, force, performance or external rVV10 implementation qualification.

## Rejected alternative

Merely changing the stored rVV10 number would preserve the wrong equation and
validate it against itself. Reusing `_vv10nlc` with a different b is retained
only as the negative control, not as an independent revised-kernel oracle.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Concurrent fixed-density potential integration

The energy-only repair was not pushed over the later potential implementation.
For z_i=1+omega_i R²/kappa_i, define a_i=1/z_i+1/(z_i+z_j).
The corrected local derivatives use dphi/domega_i=-phi R² a_i/kappa_i and
dphi/dkappa_i=phi [-3/2+(z_i-1)a_i]/kappa_i, then the existing rho/sigma
chain rule and total-density AO assembly. Both the feature and AO potential
finite-difference gates must pass against the corrected energy. Historical
potential comparisons against the wrong rVV10 energy do not qualify this repair.

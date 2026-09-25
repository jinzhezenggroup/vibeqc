# Decision: Stable WB97M-V empty-spin Stoll evaluation

Status: implemented
Date: 2026-09-24

## Problem

The WB97M-V CUDA UKS endpoint failed its unchanged beta AO potential tolerance
on a 49,152-point grid with an empty physical beta spin. The pinned Libxc work
driver floors its spin densities at `1e-13`, but the Maple-expanded density
screen reconstructed that floor through rounded `rs` and `zeta`. After fixing
the screen, the derivative of the opposite-spin Stoll correlation still lost
precision: it subtracted three terms of order `1e-2` to obtain a term of
order `1e-13`, then differentiated it through `rho_b^(-5/3)`.

## Decision

Screen on the actual work spin densities before differentiation. For a spin
fraction at most `1e-3` and above Libxc's zeta screen, evaluate the pinned
PW/Stoll antiparallel correlation
as differences that vanish analytically at full polarization. Use `log1p` and
`expm1` for the spin interpolation and the PW radial/logarithmic increments.
The parallel PW energy at nominally pure spin still contains
`opz_pow_n(-1, 4/3) = zeta_threshold^(4/3)` in the pinned Maple source;
retain its contribution separately so floating-point addition cannot discard
it before differentiation.
Leave the original pinned Maple expression as the bounded fallback elsewhere;
apply the same scalar Graph substitution to CPU and CUDA generation. Maintain
the original physical-to-work density, sigma and tau flooring policy.

## Rejected alternatives

- Relaxing the beta potential tolerance or skipping empty-spin points would
  conceal a genuine loss of significant digits.
- Fixing only the density screen reduces a branch mismatch but leaves
  cancellation in the Stoll antiparallel factor.
- Attributing the single-tile captured reference versus bounded CUDA
  endpoint difference entirely to point math ignores feature rounding caused
  by distinct tiling and contraction order.

## Invariants

The modified PW coefficients remain those of pinned Libxc 7.0.0
`lda_c_pw.mpl`; the structural Stoll substitution fails closed if the imported
Maple subtree changes. Re-evaluate against an independent wide-precision
source whenever adjusting the spin threshold or algebra. Keep the full
CPU/device E/V and variational gates at their original acceptance tolerances.

## Evidence

`tools/qualify_wb97mv_tail.py` extracts the original polarized Maple C from
the Libxc 7.0.0 tarball with SHA-256
`8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77`
and evaluates captured work features with libquadmath. On captured point
11113, its minority tau derivative is `-42743.40113431417`; the stabilized
shared Graph returns `-42743.40113431414`, whereas the preceding double Graph
returned `-42745.57241680191`. An initial stable rewrite without the
pure-spin zeta floor returned `-42743.401130193684`; its `8.10e-7` integrated
AO error exceeded the unchanged `1.90e-7` gate, so that rewrite was rejected.
On the full captured grid the largest absolute difference between device and
wide reference beta density derivative is `5.09e-15`, and between twice the
device beta kinetic coefficient and the wide tau derivative is `2.64e-8`.
Integrating the independent reference with captured AO and weights yields
beta AO `[0,0] = -94960.0100331282`; the identical-feature device coefficients
integrate to `-94960.0100331283` (difference `1.02e-10` versus acceptance
`1.90e-7`).

The Release/sm_120 `vibeqc_dft_cuda_tests` passed the complete default-grid
E/V and state gates on the Slurm-scheduled RTX 5090 with its original
257-point production tile. The pointwise golden test checks both spin
orientations against the independent wide reference.

## Consequences

Near full polarization the generated CPU and CUDA point kernels evaluate an
additional bounded algebraic branch; ordinary spin mixtures and points at the
zeta screen retain the pinned Maple path. No public CUDA WB97M-V method
admission is implied: range-separated
exchange and VV10 composition retain their independent qualification gates.

## Revisit when

The pinned Libxc PW coefficients or Stoll decomposition change, or independent
wide-precision grids expose an error outside the retained endpoint tolerances.

## References

Supersedes `../../proposed/2026-09-24-wb97mv-cuda-empty-spin-qualification.md`.
See `tests/python/test_wb97mv_semilocal.py` and
`tests/data/xc/wb97mv-tail-oracle.cpp.in` for regression inputs and oracle.

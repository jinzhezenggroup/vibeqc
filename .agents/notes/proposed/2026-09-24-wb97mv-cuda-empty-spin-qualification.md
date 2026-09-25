# Investigation: WB97M-V CUDA empty-spin potential is not qualified

Status: proposed
Date: 2026-09-24

## Problem

PR #1155 at `b6a71b2c991da439a950a6c1e314f800cc2036af` fails the
Release/sm_120 `vibeqc_dft_cuda_tests` on RTX 5090. The 49,152-point default
grid with zero beta density gives UKS beta AO potential `[0,0]` of
`-94934.062332062837` on CUDA versus `-94934.152212486413` in the independent
CPU endpoint. The unchanged acceptance gate is `2e-11 + 2e-12*abs(reference)`.
Energy and electron counts pass; the failure is in the minority potential.

## Diagnosis

Capturing a full test-only tile of device AO, density features, and point
coefficients separates three effects. The ordinary bounded 257-point CUDA
endpoint differs from the host point evaluator integrated on *identical device
features* by `-0.024705616422579624` in that potential element. The remaining
roughly `0.065` of the original discrepancy comes from separate CPU/device
feature and evaluation paths. The test verifies captured AO against host AO
and device features against independent AO-pair contractions; it does not
mistake a point-math discrepancy for a quadrature/tile assembly defect.

At point 44031 the device alpha density is `3.9473117250526825e-12`,
alpha sigma `4.1017746243813355e-22`, and alpha tau
`1.2989367752344422e-11`; for identical features, beta `v_rho` is
`0.00053194897928781292` on host and `0.00067723167670816708` on device.
This is an especially sensitive Libxc work-MGGA boundary: the physical beta
features are zero, but the pinned evaluator differentiates at the floored
work beta density `1e-13`, sigma and tau, without differentiating those floors.

The Maple-expanded per-spin density screen reconstructs `rho_b` through
`rs` and `1-zeta`, then compares the rounded result with `1e-13`. That is
not a stable test of equality at the floor: original Libxc's work driver
compares its spin density directly. An experimental shared-Graph rewrite to
use `rho_a`/`rho_b` for the two screens before differentiation removed one
branch discrepancy but did **not** pass the complete gate: the identical-input
AO potential differed by `-0.024347877551917918`. Replacing `1+zeta` and
`1-zeta` with direct spin fractions as well left `-0.024666628436534666`.
Turning off host FMA contraction did not improve the point discrepancy.
These experimental rewrites were reverted; do not claim their pointwise
improvement qualifies the complete consumer.

An independent, diagnostic 113-bit probe was compiled from the original
Libxc 7.0.0 `maple2c/mgga_exc/hyb_mgga_xc_wb97mv.c` polarized `func_vxc_pol`
with libquadmath, using the pinned archive SHA-256
`8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77`.
For work inputs `rho=(1.5728682303591006e-07,1e-13)`,
`sigma=(3.0838755830330897e-13,0,pow(pow(1e-13,4/3),2))` and
`tau=(2.4513496827871423e-07,1e-20)`, it returns beta `v_rho` of
`0.027480251915270293852441453140308316`. The original device gives
`0.027480251915401661`, while the original host gives
`0.0273350739258158`. The separately pinned Libxc 7.0.0 double C API gives
`0.02748025155033081`. One-point proximity is not a full-grid acceptance
gate: pointwise rounding and branch choices can vary across the entire grid.

## Next gate

Before changing production algebra, establish an independent wide Libxc
reference across the occupied and empty-spin point domain, including both
work-driver floor equality and adjacent floats. Then compare both generated
backends against that oracle **and** retain the complete CPU/device E/V,
variational, graph-replay, failure-isolation, and canary gates with unchanged
tolerances. Do not hide the discrepancy by loosening the AO potential tolerance,
skipping the default grid, or replacing the CPU production reference with a
same-device-input reference alone. The public CUDA WB97M-V method remains
disabled pending range-separated exchange and VV10 composition as well.

## Subsequent qualification

Superseded by
`../implemented/numerics/2026-09-24-wb97mv-empty-spin-stoll-stability.md`.
The original same-input comparison above combined a 49,152-point reference
capture (one tile) with the bounded 257-point production tiling. The different
AO/feature contraction order also changes rounded inputs; the approximately
`0.024` residual must **not** all be attributed to point formula error.
The later full-grid independent libquadmath reference and the stable Stoll
rewrite resolve the numerical qualification without relaxing the endpoint gate.

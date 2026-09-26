# Decision: stabilize the M06-2X Stoll perpendicular correlation

Status: implemented; full endpoint qualification still pending
Date: 2026-09-26

## Problem

The M06-2X correlation Maple source forms `lda_stoll_perp(f_pw, rs, zeta)`
by subtracting two O(0.03) parallel PW contributions from its O(0.03)
total. At empty-beta work density `1e-12`, the remainder and its sigma
derivative depend on a difference of roughly `1.2e-12`. A one-ULP change
to majority rho therefore perturbed the generated derivative by ~1e-5
relative. Substituting only `1 +/- zeta`, or precomputing rational powers
at the work floor, left this cancellation intact.

## Decision

For polarized M06-2X correlation only, identify the *shared* Stoll
perpendicular factor in the source-derived `m05_fperp` and `vsxc_fperp`
expressions. Verify both are reachable, shaped as source-derived products,
and use exactly the pinned modified PW parameters. Replace that common
factor before AD with an analytic rearrangement in `fraction = minor/total`.
Its radial/logarithmic differences use `log1p`/`expm1` instead of subtracting
nearly equal O(1) PW energies. The related omegaB97M-V rewrite in
`python/vibeqc_compiler/xc/wb97mv_maple.py` independently establishes this
identity; keep the different Libxc density thresholds in each adapter.

When *both* work spins are at or below the inclusive correlation density
screen, retain the source's bounded original Stoll branch: neither parallel
term exists there. Reject altered source bindings or a one-sided structural
change; never alter the generic Libxc importer or other methods' identities.
The production adapter retains raw work-input vxc and physical-density
energy weighting. This does not introduce a CPU fallback or runtime oracle.

## Evidence and remaining gates

On 50 original host points, the source-derived ordinary Libxc reference
passes with maximum normalized error 0.000252. All six exact/adjacent empty
spin points pass the independent pinned 113-bit Maple source with maximum
normalized error 8.99e-8. Slurm RTX 5090 repeats all point, finite-difference
and restricted-spin gates: its maximum wide normalized error is 4.50e-8.
The compiled binary64 Libxc exact-empty diagnostic remains intentionally
failed (18.656): the unmodified two reference intervals are disjoint, per
`2026-09-26-split-hybrid-wide-boundary-authority.md`.

Current reports retain `dirty_tracked_sources=true` because the repair is
uncommitted. A clean tested head, matched-grid RKS/UKS complete endpoint,
Compute Sanitizer and semantic work/performance gates must still pass before
any scientific qualification claim or publication. Revisit the source
rearrangement if the pinned PW parameters, spin floors, or generated
math identities change.

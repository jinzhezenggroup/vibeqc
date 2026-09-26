# Decision: split-hybrid work-boundary screening and reference policy

Status: implemented; full M06-2X and endpoint acceptance still pending
Date: 2026-09-26

## Problem

The generated split-hybrid work adapter reproduced the per-component Libxc
density/sigma/tau floors but not the compiled reference's global Fermi-hole
curvature option. Its Maple spin screens reconstructed a floor-limited spin
density from rs and zeta. At the exact floor, cancellation and FP64 rounding
could activate a branch that Libxc screened, producing enormous spurious
minority `vsigma` in M06-2X and MN15.

## Decision

Keep the generic Libxc bulk importer and its source identities unchanged.
For the polarized split-hybrid programs only, replace the exact source-derived
spin-density subgraphs with the already-floored work rho before differentiation.
Require both matching source graphs or no screen at all; a one-sided source
change fails instead of silently lowering a different method. The program
identity includes the adapter source and policy. The pinned Libxc 7.0.0 oracle
used through PySCF 2.14.0 globally enforces FHC at initialization, even when
individual registration flags omit it. Pin that compiled policy explicitly in
generated adapters and verify it against the independent oracle at runtime.

This supersedes the component-flags-only FHC assumption in
`2026-09-26-split-hybrid-admission-and-work-domain.md`. Neither workaround
changes the caller's original-density energy weighting or differentiates
through work clipping. The public qualification metadata remains pending.

## Evidence and remaining boundary

Independent Libxc 7.0.0 host probes with 50 original points per method show
that the corrected MN15 point gate passes (maximum normalized discrepancy
0.028), including FHC and the formerly failing low-density tail. M06-2X
passes its interior, FHC, and low-density tail checks, but **still fails** the
exact-empty-spin first-derivative gate: correlation `vsigma` differs by roughly
5.6e-6 relative, against the existing 3e-7 relative criterion. No tolerance
was relaxed and no point was skipped. The full matched-grid endpoint, sanitizer,
and performance gates remain pending as specified in
`docs/maintainer/hybrid_cuda_acceptance.md`.

Replacing `1 +/- zeta` by direct spin ratios perturbed the independent MN15
binary64 boundary result instead of restoring parity. Emitting the root zeta
as a single division also worsened the MN15 empty-spin comparison. Both were
discarded; a future numerical change needs separate high-precision oracle
evidence, like the r2SCAN precedent in
`2026-09-23-scan-stable-spin-fractions.md`, rather than a looser gate.

## Independent wide-reference follow-up

The pinned Libxc 7.0.0 archive (SHA256
`8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77`)
also supplies independent M06-2X polarized Maple E/vxc formulas. The point
validator now compiles them in GCC 113-bit arithmetic, while retaining Libxc's
binary64 work floors, global FHC clipping, and source parameter values. Both
the exact-empty points and the immediately adjacent majority-density doubles
must pass the *unchanged* E/vxc tolerance in addition to the original Libxc
binary64 gate. Archive, source and generated-probe hashes are reported.

At the exact empty-beta point, the host generated `vsigma_bb` is
`6.589313973682683e16`, versus the wide source `6.589321868264429e16`:
the normalized error is **3.994**, not a pass. The original binary64 source
reports `6.589350853277056e16` and a normalized error of **18.656**.
For the adjacent majority value one ULP below the original, the generated
output misses the wide source by **33.319** normalized units on both the
host C++ and Slurm RTX 5090 probes. MN15 passes both ordinary independent
point gates. Replacing
`1 +/- zeta` with direct spin fractions *only* in M06-2X correlation did not
improve the worst channel and was reverted. Matching Libxc at a neighboring
input is thus insufficient as an acceptance rule. M06-2X, clean-head point
validation, and the complete endpoint remain blocked; do not claim
qualification until an independently demonstrated conditioning repair passes
both reference gates (or a scientifically justified revised gate supersedes
this policy).

The simultaneous-reference requirement for the exact-empty M06-2X points
was superseded by
`2026-09-26-split-hybrid-wide-boundary-authority.md`: its two reference
intervals are mathematically disjoint under the unchanged tolerance. Both
binary64 and 113-bit outputs remain in the acceptance report.

The pending M06-2X *point* gate was subsequently resolved by the
source-audited Stoll cancellation repair in
`2026-09-26-split-hybrid-stoll-conditioning.md`. Complete-endpoint
qualification remains pending.

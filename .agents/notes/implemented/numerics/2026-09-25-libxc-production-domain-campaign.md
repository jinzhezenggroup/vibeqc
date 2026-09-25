# Decision: produce fail-closed Libxc production-domain campaigns

Status: implemented
Date: 2026-09-25

## Problem

The bulk importer already had a versioned admission profile and exact receipt
schema, but no profile-driven producer for the full numerical rho/sigma/tau
matrix. Existing boundary fixtures covered selected endpoints and r2SCAN-specific
production semantics. They could not be promoted into generic bulk evidence
without inventing missing cases or conflating the interior runtime with a
production boundary continuation.

## Decision

Add deterministic physical coordinates for every non-control v2 matrix row and
a single-functional evidence campaign tool.

The coordinate owner stores densities, Cartesian density gradients, and kinetic
densities. Compact sigma coordinates are derived from those gradients so
polarized Gram matrices are physical by construction. Uniform-gas and
iso-orbital tau rows use their defining physical limits.

The optional campaign uses the exact imported bulk order-2 candidate for the
observed energy/vxc/fxc vector and PySCF 2.14.0 / Libxc 7.0.0 `eval_xc(deriv=2)`
as the independent oracle. Candidate rejection, oracle non-finiteness, output
mismatch, and numerical mismatch become explicit failed receipt rows.

The two generic control rows remain `not-run`; they are not fabricated as
numeric probes. A valid campaign therefore produces a complete identity-bound
receipt without falsely granting production admission before #1120 B3 defines
their shared semantics.

## Rejected alternatives

- Reusing arbitrary sigma triples would admit non-physical polarized Gram
  matrices and weaken the oracle comparison.
- Bypassing the current bulk runtime validator to force zero-density evaluation
  would test an unowned Graph interpretation rather than the candidate contract.
- Marking unresolved control rows as passing would manufacture evidence.
- Making every blocked campaign return a process error would prevent B2 from
  collecting a broad admitted/blocked inventory; `--require-pass` provides the
  hard-gate behavior explicitly.

## Invariants

- No production path imports PySCF or runtime Libxc.
- The independent oracle never uses the VibeQC Graph to generate expected values.
- Every campaign row corresponds to the exact versioned cases-by-spin profile.
- Missing boundary policy stays visible as fail/not-run evidence.
- A campaign file is evidence input, not capability by itself; promotion still
  goes through the receipt and stage-evidence validators.

## Evidence

Unit tests cover exact numerical/control coverage for LDA, GGA, and tau-MGGA,
finite physical coordinates, polarized sigma Gram validity, spin-aware
unpolarized coverage, ingredient-driven runtime feature layouts, and exact
vacuum coordinates.

The authorized node3 worker was offline while this stack was authored, so the
branch does not claim an out-of-band pytest run; repository CI is the executable
validation authority.

## Consequences

B1 can now enumerate and diagnose the real numerical blockers for every eligible
imported semilocal registration one at a time. The current interior-only runtime
is expected to reject some v2 boundary rows until B3 adds versioned production
continuations; those failures are now machine-readable instead of implicit.

## Revisit when

- B3 supplies generic control-case semantics and stable boundary continuations;
- B2 needs a catalog-wide scheduler/retention format on top of the
  single-functional campaign; or
- the production-domain profile advances beyond v2.

## References

- #1118
- #1120
- #1280
- #1314

Agent: ChatGPT
Model: GPT-5.6 Sol

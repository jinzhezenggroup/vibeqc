# Decision: qualify Libxc density boundaries with pinned outer screening

Status: implemented
Date: 2026-09-25

## Problem

The first qualification candidate admitted physical zero sigma but still rejected
zero spin densities, exact vacuum, and zero tau before the imported functional
could be tested. That prevented the production-domain campaign from distinguishing
a real endpoint singularity from a boundary that Libxc's own work driver screens
or passes to a source-level density branch.

Applying an arbitrary universal density/tau floor would be incorrect: different
functionals own different source mathematics, and the automatic importer must not
turn one curated PBE/r2SCAN continuation into policy for the full Libxc catalog.

## Decision

Add a second qualification-only runtime domain,
`libxc-bulk-production-candidate/v2`.

- rho, sigma and tau are structurally nonnegative rather than strictly positive;
- negative values, nonfinite inputs, and non-PSD polarized sigma Gram matrices
  still fail before mathematics;
- each candidate binds the pinned registration's
  `p_a_dens_threshold` from the imported catalog;
- points whose total density is below that threshold are returned as exact zero
  energy/vxc without evaluating the interior Graph;
- active empty-spin and zero-tau channels are passed to the imported Graph
  unchanged, with no clipping or synthetic derivative;
- the ordinary interior domain and v1 zero-gradient candidate remain unchanged;
- the single-functional and sharded catalog campaigns explicitly select v2, so
  its domain and threshold are captured by the existing execution/receipt
  identity.

The array evaluator now initializes screened lanes to zero and evaluates only the
active subset. For older domains the active mask is all true, preserving their
execution behavior.

## Rejected alternatives

- Reusing the curated r2SCAN work-MGGA floors for every MGGA would silently
  promote method-specific production semantics to unrelated functionals.
- Flooring every empty spin/tau channel to a small positive number would change
  the functional and could hide a genuine endpoint singularity.
- Sending exact vacuum through the interior Graph would test an implementation
  domain that the Libxc work boundary does not necessarily evaluate.
- Treating any structurally admitted endpoint as a pass would bypass the
  independent PySCF/Libxc oracle.

## Invariants

- This is a qualification candidate, not a public runtime domain.
- Candidate admission never grants production-domain capability by itself.
- The exact pinned density threshold is part of candidate semantic identity.
- Active endpoint algebra is never regularized merely to make a campaign pass.
- Negative/nonfinite/indefinite physical inputs remain fail-closed.
- Existing interior/v1 candidate behavior remains explicit and independently
  selectable.

## Evidence

Focused regressions cover exact-vacuum screening, mixed active/screened batches,
legacy-domain zero-density rejection, empty-spin/zero-tau structural admission,
negative rho/tau rejection, and exact threshold binding to the pinned catalog
record.

Repository CI is the executable validation authority for this stacked slice.

## Consequences

The catalog campaign can now classify vacuum and fully polarized rows instead of
blocking them at the generic positive-interior validator. Remaining failures are
more informative: they represent active imported endpoint algebra or independent
oracle disagreement rather than a blanket structural rejection.

## Revisit when

- a broader source-derived Libxc work-driver abstraction replaces the current
  qualification candidate;
- a functional requires additional work-driver state beyond rho/sigma/tau and
  density screening; or
- retained campaign evidence shows a shared, algebraically exact endpoint
  continuation that can be promoted without method-specific branching.

## References

- #1118
- #1120
- #1325
- #1327
- #1329

Agent: ChatGPT
Model: GPT-5.6 Sol

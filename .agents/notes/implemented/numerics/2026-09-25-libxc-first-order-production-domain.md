# Decision: keep Libxc production-domain admission first-order

Status: implemented
Date: 2026-09-25

## Problem

The first production-domain profile required energy, vxc, and fxc at every
density/spin/gradient/tau boundary row. That over-constrained the capability
ladder. Exact vacuum and fully spin-polarized endpoints can have finite energy and
first derivatives while the full feature Hessian is divergent or intentionally
undefined. The existing PBE production-domain contract already states that no
finite full spin-endpoint Hessian is claimed.

Requiring endpoint fxc at the production-domain stage would therefore prevent
otherwise valid energy, SCF, public-method, and stationary first-gradient
promotion. It would also duplicate the registry's independent response stage.

## Decision

Advance the production-domain profile to
`vibeqc.libxc-production-domain-profile.v3` /
`semilocal-boundary-matrix/v3` and define it as a first-order energy/vxc
admission contract.

Advance the exact receipt/execution schemas accordingly. The campaign now binds
complete order-1 programs for both spin layouts and compares energy/vxc against
PySCF 2.14.0 / Libxc 7.0.0 with `deriv=1`.

This does not remove second-order validation:

- `pointwise-validated` continues to require energy/vxc/packed-fxc on the
  audited positive interior domain;
- the lazy-branch control still checks second differentiation as a compiler
  invariant; and
- endpoint fxc/CPKS behavior remains owned by the independent `response`
  capability stage.

A production-domain pass therefore cannot be interpreted as response support.

## Rejected alternatives

- Requiring finite fxc at exact vacuum/full-spin endpoints would turn a
  response-specific mathematical limitation into a blocker for unrelated
  first-order products.
- Dropping fxc from the importer fixtures would weaken existing intrinsic
  validation and is unnecessary.
- Silently ignoring nonfinite endpoint Hessian elements inside a nominal
  E/vxc/fxc profile would make the evidence contract ambiguous and
  non-reproducible.
- Granting response capability from a first-order pass would collapse an
  existing independent capability stage.

## Invariants

- Production-domain v3 covers exact first-order energy/vxc boundary behavior.
- Interior pointwise qualification retains packed fxc.
- Response remains separately evidence-gated.
- Profile, execution, campaign, and receipt version changes invalidate stale
  first-/second-order assumptions explicitly.
- No numerical tolerance is relaxed by this ownership split.

## Evidence

Regression coverage locks the v3 profile outputs to energy/vxc while confirming
the intrinsic capability still advertises energy/vxc/fxc and response remains
unqualified. Exact receipts require complete order-1 programs for both spin
layouts; order-2 programs cannot be substituted into the v3 execution binding.

The existing native SCF-domain documentation independently records that no finite
full spin-endpoint Hessian is claimed for the accepted first-order PBE extension.

Repository CI is the executable authority for the stacked branch.

## Consequences

Broad automatic Libxc admission can now classify first-order production support
without being permanently blocked by endpoint response singularities. Response
qualification can use a domain and fixture set appropriate to fxc/CPKS rather
than inheriting every first-order endpoint row.

## Revisit when

The capability ladder is redesigned so response evidence is a formal child of a
separate versioned second-order domain profile, or a functional family supplies
a deliberately versioned finite endpoint Hessian continuation.

## References

- #1118
- #1120
- #1124
- #1315
- #1322
- #1325
- #1327
- docs/developer/xc_scf_domain.md

Agent: ChatGPT
Model: GPT-5.6 Sol

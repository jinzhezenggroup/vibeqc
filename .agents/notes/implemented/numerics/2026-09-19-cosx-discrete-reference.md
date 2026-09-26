# Decision: Pin COSX v1 to a shared analytic ESP recurrence and explicit discrete model

Status: implemented
Date: 2026-09-19

## Problem

Issue #246 needs a correctness oracle before GPU screening, fitting, task scheduling,
or provider selection can be optimized. COSX combines numerical collocation with an
analytic electrostatic-potential (ESP) integral. Implementing a second ESP recurrence
for COSX would create two scientific definitions for the same Coulomb operator, while
using an XC grid implicitly would make the approximation identity ambiguous.

## Decision

COSX reference version 1 takes explicit points and weights as mathematical inputs. It
uses the existing normalized contracted Gaussian Hermite/Boys Coulomb recurrence from
the one-electron nuclear-attraction oracle, factored into a unit positive
1/|r-R| primitive. Nuclear attraction remains -Z times that shared primitive, so
the refactor does not create a second recurrence or change derivative coordinates.

The reference forms the one-sided seminumerical exchange contraction, then explicitly
symmetrizes it. Version 1 is unscreened and has no overlap/S fitting. Requests for
screening or fitting are rejected rather than silently changing the model. Probe
coordinates are fixed external coordinates in this value-only slice.

## Rejected alternatives

- A separate COSX-specific ESP formula or recurrence: rejected because it duplicates
  scientific mathematics and makes cross-path drift likely.
- Calling PySCF/libcint from the reference path: rejected because the repository CPU
  oracle must be self-contained and production code must not acquire a hidden external
  dependency.
- Treating the existing XC GridSpec as the COSX method identity: rejected because
  the first reference accepts explicit quadrature and COSX grids may later differ from
  XC quadrature.
- Adding fitting or chain-of-spheres screening in the first slice: rejected because
  arithmetic correctness, quadrature/model error, and approximation-policy error need
  separate acceptance gates.

## Invariants

- ESP values use the positive unit-charge convention
  <mu|1/|r-R_probe||nu>; nuclear attraction remains the negative ionic-charge sum.
- RHF spin-summed exchange energy is -1/4 Tr(D K[D]); a single spin block uses
  -1/2 Tr(D K[D]).
- The one-sided discrete contraction and explicit symmetrization remain separately
  testable.
- Quadrature/model error is not folded into the arithmetic tolerance.
- GPU/native production work may tile, reorder, screen under a separately versioned
  policy, or generate the recurrence, but must reproduce the same v1 model when those
  extra approximations are disabled.

## Evidence

vibeqc_cosx_reference_tests checks the compact contraction against an explicit
four-index discrete sum on the identical points, RHF/UHF spin factors, analytic ESP
against independently integrated real-space quadrature, and grid refinement toward
analytic direct exchange. The complete CPU CTest suite passes 33/33 after the
nuclear-attraction recurrence refactor, including Cartesian and spherical integral
tests.

## Consequences

The reference intentionally materializes point-by-AO-by-AO ESP data and is not a
performance path. That makes the oracle simple and auditable while leaving bounded
GPU AO/ESP tasks, screening, fitting, derivatives, and crossover measurements to
later #246 slices.

## Revisit when

A later COSX variant adds fitting, screening, range separation, derivatives, or a
different grid prescription. Such a variant needs an explicit version/identity and
its own matched reference gates rather than mutating v1 semantics.

## References

- #246
- docs/cosx_reference.md
- tests/native/test_cosx_reference.cpp

Agent: ChatGPT
Model: GPT-5.6 Sol

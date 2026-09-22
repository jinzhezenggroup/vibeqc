# Decision: expose source-level stationary HVP integral blocks

Status: implemented
Date: 2026-09-21

## Problem

`StationaryHVPPlan` described the LDA/GGA second-order source inventory but had
no executable algebra. The complete molecular endpoint is still gated by
native moving-grid/partition consumers and shared CPKS state, so adding a
second Hessian implementation would either overclaim support or conflict with
the #179 response owner.

## Decision

Add bounded `HVPIntegralBlock` programs for the one-electron, Coulomb and
overlap/Pulay sources. The block reuses the existing
`StationaryGradientPlan.integral_block` energy/weight DAG, generates an exact
demand-driven JVP of those weights for a directional density/weighted-density
response, and combines

`d(weight) * dI + weighted_d2I(direction)`

from first-integral directional tiles and the already-weighted output vector
from #178 `weighted_hvp`. RKS and UKS use the same source algebra; UKS carries
explicit spin rows and the Coulomb derivative preserves both left/right
response terms.

## Rejected alternatives

- A numerical finite-difference HVP would hide missing response terms and is
  outside the analytic #180 contract.
- A new CPKS solver or resident response owner would duplicate #179 and create
  incompatible state/lifetime semantics.
- XC AO/grid/partition HVP code was not added before those native geometric
  directional providers were independently qualified.

## Invariants

- No shell tensor, full coordinate Hessian, or molecular result is materialized
  by these programs.
- Native providers remain responsible for state identity, shell/center recovery,
  resource budgets and second-integral execution.
- Missing geometric sources and missing response/second-integral feeds fail
  closed; no lower-order or CPU oracle fallback is implied.

## Evidence

- `tests/python/test_stationary_hvp_plan.py`: 27 tests pass, including RKS/UKS
  independent scalar weight oracles, response and second-term contraction,
  replay, source omission and missing-feed failures.
- `tests/python/test_stationary_gradient_plan.py` plus the HVP tests: 62 pass.
- Ruff check/format and `git diff --check` pass.

## Consequences

The compiler can now execute and inspect the integral part of a directional DFT
HVP while preserving the existing public capability gate. A later native
consumer can bind these programs to #178/#179 owners without reimplementing
the source weights or response derivative.

## Revisit when

Add XC/grid/partition blocks only after native second geometry directions have
independent finite-difference, raw-symmetry and failure-path evidence.

## References

- Issue #180 and compiler planning slice #810.
- `python/vibeqc_compiler/method/stationary_hvp.py`.

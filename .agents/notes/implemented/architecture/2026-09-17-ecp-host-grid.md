# Decision: Generate the production ECP host grid

Status: implemented
Date: 2026-09-17

## Problem

The projector and AO/weight consumers were compiler-owned, but production CUDA
still consumed nodes, harmonics and component coefficients assembled by native
scientific formulas. Sharing those formulas with the independent CPU oracle
also obscured which setup mathematics a generated-versus-CPU test exercised.

## Decision

`integral/ecp_grid.py` lowers Legendre recurrence/derivative/weight expressions,
radial mapping and Jacobian, real s/p/d harmonics, sphere weights/coordinates and
Cartesian component coefficients through the common scalar DAG. Finite Newton
and grid traversals are compiler-owned host schedules, emitted in the existing
ECP header. Host sin/cos calls remain direct standard-library operations; this
slice does not expand the generic algebra with unused trigonometric derivatives.

The production grid wrapper delegates to the generated helper. The public CPU
integral path retains its independent recurrence, nodes, harmonics and molecular
normalization implementation. Including a generated header in that translation
unit does not route the CPU oracle through generated arithmetic.

## Rejected alternatives

Do not replace the CPU oracle's own setup: otherwise a normalization or grid bug
could affect both sides of the primary integration comparison. Do not introduce
precomputed quadrature tables or change Newton tolerance/re-evaluation policy in
an ownership migration. Those are numerical changes requiring their own gates.
Do not label the entire CUDA adapter as runtime: its two-grid convergence policy
and basis-expansion metadata remain explicit scientific/method contracts.

## Invariants

Preserve FP64 storage, ascending Legendre/radial order, polar-major azimuth order,
the existing harmonic slots, radial 16..512 and polar 8..96 bounds, append behavior,
64 Newton iterations and the 2e-15 delta threshold. The derivative used in the
weight is from the last iteration, matching the previous implementation. Keep
160/32 versus 224/44 convergence gates, one-radial-shell device staging, physical
center scatter and arbitrary real fixed-weight force semantics unchanged.
Unsupported component powers throw before execution. No DFT, high-l or new
element-family capability is inferred from this migration.

## Evidence

Native tests use polynomial moments through degree 2*n-1, standard-library
Legendre root residuals, harmonic Gram matrices and the addition theorem,
analytic radial Gaussian moments and gamma-function Cartesian normalization.
Odd/even and minimum/maximum orders are covered. These oracles do not use the
generated expression to compute expected answers. Python tests check the
independent rational map/Jacobian and harmonic channel theorem, alongside
stdlib-only deterministic source generation. Complete CPU/Libcint and CUDA
all-center/HF comparisons remain mandatory retirement gates.

[Retained qualification](../../../../benchmarks/results/ecp-grid-171/README.md)
passes 30 CPU native tests, 45 CPU Python tests (10 GPU skips), 55 CUDA Python
tests, two CUDA native tests and sanitizer with zero errors. Complete RHF/UHF
force errors remain below 5.54e-13/4.65e-16 Eh/bohr. An initial UHF timing
outlier prompted a predefined ABBA follow-up; pooled endpoint medians changed
by less than 0.1%, with unchanged planned peaks and kernel resources. Both the
initial and follow-up samples are retained; no statistical speed claim is made.

## Consequences and revisit conditions

The generated header grows, while handwritten production host setup is removed.
Native device scheduling and storage are unchanged. Generic molecule expansion
metadata and complete-method convergence selection remain outside this slice.
Revisit higher angular coverage or alternate quadratures only with independent
raw derivatives, complete method endpoints and bounded-resource evidence.

## References

The orbital-domain boundary is subsequently extended through f by the
[orbital-f decision](../numerics/2026-09-17-ecp-orbital-f.md); the grid ownership
and independent oracle rationale above remain applicable.

- Issue #171; PRs #371 and #400.
- [Current ECP contract](../../../../docs/ecp.md).
- [Prior AO/weight decision](2026-09-16-ecp-ao-weight-consumers.md).

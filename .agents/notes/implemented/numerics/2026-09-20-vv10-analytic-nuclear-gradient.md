# Decision: split VV10 analytic nuclear gradients by geometric source

Status: implemented
Date: 2026-09-20

## Problem

Issue #491 C requires the first nuclear derivative of the VV10/rVV10
double-integral energy. Reusing semilocal XC geometry rules is incorrect:
the nonlocal kernel depends explicitly on pair distances, and each quadrature
weight appears on both legs of the symmetric pair integral.

The gradient must also compose with #163 without hiding VV10 work inside the
existing semilocal AO/grid/weight sources.

## Decision

Keep three VV10-specific stationary sources: `nonlocal_ao`,
`nonlocal_grid`, and `nonlocal_weight`. The AO source owns density and
density-gradient response to basis-center motion. The grid source owns both
feature motion under point translation and explicit pair-distance motion.
The weight source owns the complete molecular partition-weight response.
Each source is consumed exactly once by the shared stationary-gradient plan.
For
`E = beta sum_i A_i + 1/2 sum_ij A_i A_j phi_ij`, with
`A_i = w_i rho_i`, the exact fixed-feature weight derivative is

`dE/dw_i = rho_i [beta + sum_j A_j phi_ij]`.

It is intentionally not `rho_i * epsilon_i`: the latter contains only half
of the pair contribution and would silently undercount the weight response.

The explicit point derivative uses `d phi / d r_ij^2` from the audited
VV10/rVV10 kernel and contracts `2 (r_i-r_j)`. Feature derivatives continue
to use the already-qualified `vrho`/`vsigma` definition from slice B.
No second nonlocal scientific algebra is introduced.

The compiler ownership boundary is preserved: `dft` owns the fixed-density
nonlocal partials and replay identities, while the runtime DFT-gradient adapter
combines those partials with `xc.grid_response`. A direct `dft -> xc` import
was rejected by `tools/check_compiler_structure.py` and is not part of the
implementation.

## Provenance and replay

A `NonlocalGeometry` snapshot records basis, density, source-grid, explicit
quadrature, and kernel identities. Physical nuclear-component replay
revalidates basis, density and both grid identities. This prevents an old
VV10 geometric partial from being combined with a changed SCF density or
changed molecular grid.

`NonlocalIntegral.grid_identity` retains the caller-visible source-grid
identity. The materialized explicit-grid identity is recorded separately as
`quadrature_identity` so adding MolecularGrid support does not redefine the
existing provenance field.
## Rejected alternatives

- Reuse semilocal `rho * epsilon` as the weight partial: misses one pair leg.
- Fold pair-distance motion into `xc_grid`: obscures ownership and permits
  accidental double counting when semilocal and nonlocal terms coexist.
- Recompute finite differences per nuclear coordinate in production: useful
  only as an oracle and not an analytic-gradient implementation.
- Materialize a full grid-pair matrix: unnecessary; the reference derivative
  retains the bounded tiled pair loop.

## Evidence

Both VV10 and rVV10 explicit point/weight derivatives match independent
three-step central differences to near FP64 roundoff on the four-point oracle.
On native H2 AO fixtures, simultaneous independent AO-center, point and weight
motions converge to the analytic directional derivative. Rebuilt atom-centered
MolecularGrid finite differences independently validate the complete
`nonlocal_ao + nonlocal_grid + nonlocal_weight` nuclear contribution.
Rigid translations cancel, RKS/UKS total-density layouts agree, and stale
grid or density identities fail closed.

The focused DFT/VV10 regression set passes 194 tests with ruff clean.

## References

- Issue #491, slice C.
- `2026-09-19-vv10-fixed-density-potential.md`
- `2026-09-19-generated-becke-grid-response.md`
- `python/vibeqc_compiler/dft/nonlocal_integration.py`
- `python/vibeqc_compiler/method/stationary_gradient.py`

Agent: Agent D
Model: GPT-5.6 Sol

## Review correction: bind replay to the current primitive

The geometry record's kernel hash was initially only recorded, not checked
against the current method. The runtime resolver now requires the current
`NonlocalCorrelationPrimitive` and passes its kernel specification and exact
coefficient to the compiler-owned replay validator. The geometry snapshot also
retains the explicit coefficient. VV10/rVV10, b/C, and coefficient changes reject
before geometric sources can be consumed. The compiler `dft` package depends
only on the common specification type, not on MethodIR or public runtime.

The pre-fix independent review reproduced VV10 partials being accepted by an
rVV10 diagnostic assembly on the same basis/grid/density. Added regressions
exercise both variant directions, parameter changes, non-unit coefficient replay,
and rejection when the expected primitive is omitted. No derivative equation,
finite-difference threshold or public capability is changed.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Integration with the merged global-hybrid plan

Master f529b85f includes #618's independently qualified global-hybrid gradient
plan. The integration keeps both that plan and the semilocal/nonlocal plan.
Exact exchange is selected by primitive type, not by assuming every second
primitive is exact exchange: a nonlocal correlation node must never become K
weights. Two spin-mode regressions protect both inventories. A combined
hybrid-plus-nonlocal plan remains explicitly unqualified rather than being
promoted accidentally by resolving this merge. Existing exchange coefficients,
ERI derivative weights, nonlocal kernel binding and independent gates remain.

Agent: ChatGPT
Model: GPT-6 Astra Pro

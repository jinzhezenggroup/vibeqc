# Decision: separate stationary DFT state from generated XC geometry partials

The state-authorization and #163-A completion claims below are superseded by
[the native handoff decision](2026-09-16-stationary-native-handoff.md). The
fixed-density oracle rationale remains applicable.

Status: implemented
Date: 2026-09-15

## Problem

The generated semilocal XC contractions own exact derivatives of one discrete
fixed-density energy, but they cannot prove that their density belongs to a
successful current KS solution or infer how an atom-centred molecular grid moves.
Treating those partials as a molecular force would silently omit stationarity,
Pulay, other energy terms, and partition response.

## Decision

Keep local LDA/GGA expression and AO-jet pullback ownership in
`vibeqc_compiler.xc.ContractionProgram(..., "geometry")`. Add a private DFT
method boundary in `vibeqc._dft_gradient` that first validates the complete
stationary KS state and exact geometry, basis, grid, topology, functional,
provider, and generation identity. It binds the generated centre, point, and
weight partials to that state and contracts each independently declared motion
source exactly once with gradient sign.

Grid motion is explicit and accepted only while the caller asserts the same
topology. The A slice provides no inferred partition motion, complete molecular
assembly, force conversion, or CUDA gradient lowering.

## Rejected alternatives

- Put SCF state and stationarity into the XC compiler contract: that would make a
  fixed-density mathematical owner responsible for method and solver semantics.
- Handwrite a second LDA/PBE derivative in the method layer: it would duplicate
  the generated expressions and weaken provenance.
- Differentiate the SCF/DIIS trajectory: an ordinary variational first
  derivative depends on the accepted final physical state, not solver history.
- Publish the partial XC result as a force: it has neither all energy components
  nor the complete physical grid response.

## Invariants

- Density, physical Fock, orbitals, occupations, orbital energies,
  energy-weighted density, overlap, identities, and generations describe one
  successful converged physical state before any partial is consumed.
- RKS/UKS spin and LDA/GGA functional identity match the generated discrete
  energy contract exactly.
- AO-centre, point, and quadrature-weight sources are independent and appear
  exactly once in a directional gradient.
- Shape, finiteness, identity, and stable-topology checks fail closed.
- Public DFT force capability remains unsupported until complete method assembly
  and its CPU/CUDA acceptance gates exist.

## Evidence

`tests/python/test_dft_stationary_gradient.py` covers LDA/PBE RKS/UKS state
acceptance, asymmetric UKS occupation, stale and inconsistent state, isolated
and combined source directions, rigid translation, topology rejection, and
component omission/sign negative controls. The independent helper in
`tools/vibeqc_validation/dft_gradient.py` rebuilds native AO collocation after
displacing centres, points, and weights and compares several central-difference
steps without calling the generated geometry pullback.

On qz CPU at implementation revision `6a6dfbf`, the focused stationary/XC/DFT
suite passed 172 tests with 20 environment-dependent skips. This establishes the
CPU contract and oracle slice, not a full DFT-gradient or GPU-gradient result.

## Consequences

The method layer gains an auditable input for future #163 component assembly
without changing public method capability. Callers must carry more exact
identity and motion metadata, and topology-changing grid logic remains explicit
future work rather than an implicit approximation.

## Revisit when

Revisit this boundary when #163-B supplies complete atom-centred grid and
partition motion plus all remaining molecular-gradient components, or when
#163-C adds bounded CUDA gradient execution and the public capability gate.

## References

- Issue #163, slice A
- Issue #236 generated XC geometry partials
- `docs/user/methods.md`
- `2026-09-16-stationary-native-handoff.md`

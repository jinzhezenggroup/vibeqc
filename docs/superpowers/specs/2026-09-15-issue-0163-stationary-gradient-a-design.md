# Issue 163-A stationary DFT derivative design

Implementation status: the A slice is implemented. The live #162 state/source
handoff now feeds the exact SCF-domain point energy/differential into the
compiler-generated AO/density geometry pullback, so no `interior-v1` relabeling
is used. The private point bridge is covered by the 97 independent Libxc/mpmath
SCF-domain references; the fixed-density multistep oracle and directional tests
remain independent of the generated AO pullback. Complete grid/partition motion,
molecular gradients, CUDA derivative lowering and public forces remain B/C.
See [current method behavior](../../user/methods.md) and the
[native handoff decision](../../../.agents/notes/implemented/numerics/2026-09-16-stationary-native-handoff.md).

## Scope

Deliver the Issue #163 A slice: a typed stationary LDA/GGA derivative contract,
reuse of the generated explicit XC geometric partials from Issue #236, and an
independent CPU oracle with component and directional tests. The slice prepares
complete RKS/UKS gradient assembly without publishing a molecular gradient or
force capability.

This slice does not implement the full moving atom-centred grid and partition
response, assemble all one-electron/Hartree/Pulay/nuclear terms, lower the
gradient to CUDA, add CPKS, or change the public energy-only DFT capability.
Those remain in Issue #163 B/C and later method tasks.

## Chosen approach

Add a DFT-method-owned stationary derivative boundary above the existing
compiler-owned XC contraction programs. Do not extend the Issue #236 fixed-D
contract until it becomes a molecular-method owner, and do not introduce a
second handwritten LDA/PBE derivative implementation.

The alternatives rejected for this slice are:

- assembling a complete fixed-grid RKS gradient now, which would mix the A and
  B acceptance boundaries and prematurely own integral/Pulay components;
- adding stationary state to `xc.DiscreteEnergyContract`, which would confuse
  fixed-density XC partials with the constrained derivative of a complete KS
  energy;
- differentiating the SCF or DIIS trajectory, which is unnecessary for an
  ordinary variational first derivative and creates an invalid solver-history
  dependency.

## Ownership boundaries

The derivative path has one scalar point primitive and one generated geometric
owner. The native SCF point evaluator owns the exact versioned LDA/PBE
energy/first-differential domain already used by #162; its 97-point independent
reference suite prevents a second formula from drifting. The compiler layer
owns the compact geometric contractions:

- LDA/GGA ingredient and Cartesian point-coefficient conventions;
- AO spatial jets and their generated pullback into AO-centre and grid-point
  sources;
- explicit quadrature-weight derivatives;
- output-pruned generated CPU programs for the requested geometric observable.

The stationary DFT layer owns method and state semantics:

- proof that density, orbitals, occupations, orbital energies,
  energy-weighted density, physical Fock and overlap describe one successful
  current KS state;
- exact geometry, basis, grid, functional, regularization, spin and Coulomb
  provider identity;
- the mapping from independently generated geometric sources to a declared
  nuclear displacement direction on one stable grid-topology branch;
- component accounting, gradient-versus-force sign and rejection of stale or
  incomplete state.

Issue #163 B owns complete physical grid/partition motion and final molecular
component assembly. Issue #163 C owns bounded CUDA execution and public method
capability.

## Stationary derivative contract

Introduce an immutable contract representing one supported derivative request.
It records:

- functional family and exact functional identity;
- RKS or UKS spin semantics;
- geometry, orbital basis and overlap identities;
- grid specification, owner-atom topology, mask and regularization identity;
- resolved Coulomb/provider identity;
- solve epoch and density, Fock and orbital generations;
- the availability and shape of `D`, physical non-DIIS `F`, `C`, orbital
  energies, occupations and `W = C diag(f epsilon) C^T`;
- a stable-topology derivative policy and `gradient` sign convention.

The contract accepts only converged successful final-state snapshots produced
by the Issue #162 boundary. Matching dimensions, hashes copied from another
state, a previous successful epoch or a warm density seed are insufficient.
Validation is deterministic and completes before evaluating any derivative
component.

For the A slice, supported requests are real all-electron LDA/GGA RKS/UKS on an
explicit stable grid branch. Hybrid, meta-GGA, ECP, Hessian, topology-switching
and public force requests fail explicitly.

## Generated XC source consumption

Reuse `xc.ContractionProgram(..., "geometry")` and its native equivalent as the
scientific owner of `GeometryPartials`:

- `centers[A,k] = partial E_xc / partial R_Ak` from AO-centre motion at fixed
  density, points and weights;
- `points[g,k] = partial E_xc / partial r_gk` from grid-point motion at fixed
  AO centres, density and weights;
- `weights[g] = partial E_xc / partial w_g`.

The stationary layer must not recompute these expressions. It validates their
contract identity and contracts them with separately declared motion fields:

```text
dE_xc = centers : dR_basis + points : dr_grid + weights . dw
```

Every source appears exactly once. The A slice provides explicit identity,
zero and rigid-translation motion adapters for testing. A complete molecular
partition-motion generator is intentionally absent until B.

## Independent CPU oracle

The oracle must not call the generated geometry pullback under test. It
re-evaluates the scalar discrete XC energy on independently displaced inputs at
fixed density and uses symmetric finite differences at multiple step sizes.

Independent displacement controls cover:

- AO centres with grid points and weights fixed;
- grid points with AO centres and weights fixed;
- quadrature weights with centres and points fixed;
- arbitrary simultaneous combinations of all three sources.

The oracle rebuilds AO collocation from displaced centres and evaluates the
audited scalar functional energy. It records the step sequence and requires a
stable finite-difference region rather than accepting one favourable step. The
generated primal may be reused as an energy value only in ordinary production;
it is not the sole derivative oracle in these tests.

## Error handling and topology boundary

Reject before publication when:

- the KS state is failed, nonconverged, stale or internally inconsistent;
- geometry, basis, grid, functional, regularization, provider or spin identity
  differs from the final state;
- required `W`, occupation or physical-Fock information is absent;
- AO-to-atom ownership is invalid or inconsistent with the basis;
- a requested motion changes grid membership, pruning, screening or task masks;
- a component has a nonfinite value or incompatible shape;
- the caller asks for a force while only partial gradients are available.

Changing coordinates while preserving an asserted topology is a differentiable
branch request. Crossing a pruning or partition branch is a separate control
event and is not silently treated as a smooth derivative.

## Tests and acceptance

Add focused Python tests covering:

- typed payload/identity stability and exact supported-domain rejection;
- state identity acceptance from an internally consistent RKS and UKS fixture;
- rejection of stale solve epochs and mismatched density/Fock/orbital/grid/
  functional/provider identities;
- independent AO-centre, point and weight finite differences for LDA and PBE;
- RKS and UKS spin factors, including an asymmetric spin case;
- arbitrary combined directions and multiple finite-difference steps;
- component omission and sign-reversal negative controls that fail the declared
  tolerance;
- translation consistency between AO-centre and point sources;
- invalid AO ownership, nonfinite directions and topology-change rejection;
- proof that public DFT force capability remains disabled.

The generated component directional derivatives must agree with the independent
oracle within a tolerance justified by the stable step-size region. Existing XC
energy, potential, response and geometry tests must remain unchanged and pass.

## Commit and PR structure

Use one reviewable PR with meaningful commits:

1. this approved design and its exact completion boundary;
2. the stationary contract and independent CPU oracle;
3. component, directional, identity and negative tests plus current-state
   documentation required by the implemented interfaces.

The PR closes only the #163 A slice. Its description must state that complete
moving-grid/partition response, final molecular gradients, CUDA lowering and
public forces remain open under #163 B/C.

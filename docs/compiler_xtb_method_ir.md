# GFN-family compiler MethodIR

Issue #503 adds a compiler-owned representation layer for GFN-family
semiempirical methods.  The first audited graph is GFN2-xTB.  This layer is
deliberately not a public xTB execution endpoint.

## Ownership boundary

The compiler owns:

- canonical method and parameter-set identity;
- required parameter-table families and supported elements;
- overlap, dipole, and quadrupole integral requirements;
- H0, ES2/ES3/AES2 SCC electrostatics, spin-polarization, and fixed-state Hamiltonian algebra;
- coordination-number, repulsion, and self-consistent D4 correction semantics;
- requested compiler products such as energy and nuclear-gradient ingredients.

The runtime owns:

- SCC iteration, mixing, warm starts, and convergence policy;
- occupations and finite-temperature state;
- generalized eigensolver implementation/library selection;
- per-system convergence state and admission to a public execution endpoint.

Accordingly, XtbMethodIR may require an SCC fixed point and a generalized
eigensolution, but it does not contain iteration counts, tolerances, Broyden or
DIIS history, or an eigensolver implementation choice.

Runtime fixed-point control is shared with mean-field SCF through
`src/scf/solver/self_consistent.hpp` (#581). A future GFN2 runtime adapter owns
its charge/multipole state, occupations, Hamiltonian construction and mixing
policy while reusing that method-neutral convergence driver; this compiler IR
still owns none of those policies.

## Canonical GFN2 graph

The initial graph is ordered as:

1. basis/atomic parameter requirements;
2. coordination-number model;
3. overlap integrals;
4. cumulative dipole/quadrupole integrals;
5. coordination-dependent H0;
6. ES2/ES3/AES2 SCC electrostatics over charge/dipole/quadrupole state, with the AES2 coordination-number dependency explicit;
7. shell spin polarization;
8. fixed-state Hamiltonian assembly;
9. geometry-only repulsion;
10. self-consistent D4 dispersion.

The graph is a requirement contract for #504 and #505.  It does not duplicate
their numerical parameter tables, pair kernels, Hamiltonian lowering, or
adjoints.

## Identity and provenance

Semantic identity includes:

- the XtbMethodIR schema version;
- model flavor and restricted/unrestricted reference;
- parameter-set identifier, revision, source, supported-element scope, and
  required table families;
- every primitive model/dependency/state/derivative contract;
- requested compiler products.

The descriptive method name is excluded from semantic identity so aliases of
the same audited composition can reuse compiler caches.  It is retained in
manifest_identity for result provenance.

The initial GFN2 parameter manifest records the published GFN2-xTB source DOI
and the H-through-Rn element scope, but stores only symbolic table requirements.
Numerical parameter data must be introduced by later audited lowering work.

## Capability boundary

A resolved GFN2 graph currently reports:

- compiler_representable = true;
- lowering_available = false;
- runtime_executable = false.

This prevents MethodIR construction from being mistaken for a working public
GFN2 endpoint while #504/#505 and the runtime SCC/eigensolver work remain
separate.

## GFN1 extension point

The schema recognizes gfn1 as a future family discriminator, but the resolver
rejects it explicitly because no audited GFN1 primitive graph/parameter
manifest is registered here.  Adding GFN1 later requires its own audited graph
and does not inherit GFN2 correction or parameter semantics by default.

## Dependency rule

Production VibeQC code in this layer depends only on VibeQC compiler modules.
It does not import xTBloom and does not invoke an external xtb executable or
library.  xTBloom may be used independently as a qualification oracle in later
implementation slices.

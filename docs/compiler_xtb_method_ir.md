# GFN-family compiler MethodIR

Issue #503 introduced the compiler-owned representation layer for GFN-family
semiempirical methods. #837 extends that layer with an audited GFN1-xTB graph
beside GFN2-xTB. This layer is deliberately not a public xTB execution
endpoint.

## Ownership boundary

The compiler owns:

- canonical method and parameter-set identity;
- required parameter-table families and supported elements;
- model-specific integral requirements: overlap for GFN1, and
  overlap/dipole/quadrupole for GFN2;
- H0, SCC electrostatics, spin-polarization, and fixed-state Hamiltonian
  algebra;
- coordination-number and repulsion semantics, plus GFN1 D3(BJ)/halogen or
  GFN2 self-consistent D4 correction semantics;
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
`src/scf/solver/self_consistent.hpp` (#581). A GFN runtime adapter owns its
electronic state, occupations, Hamiltonian construction and mixing policy while
reusing that method-neutral convergence driver; this compiler IR still owns
none of those policies.

## Canonical GFN2 graph

The GFN2 graph is ordered as:

1. basis/atomic parameter requirements;
2. coordination-number model;
3. overlap integrals;
4. cumulative dipole/quadrupole integrals;
5. coordination-dependent H0;
6. ES2/ES3/AES2 SCC electrostatics over charge/dipole/quadrupole state, with
   the AES2 coordination-number dependency explicit;
7. shell spin polarization;
8. fixed-state Hamiltonian assembly;
9. geometry-only repulsion;
10. self-consistent D4 dispersion.

The graph is a requirement contract for generated lowering. It does not
duplicate numerical parameter tables, pair kernels, Hamiltonian lowering, or
adjoints.

## Canonical GFN1 graph

The GFN1 graph is deliberately model-specific rather than a GFN2 graph with
different constants:

1. basis/atomic parameter requirements, including GFN1 shell-valence metadata;
2. exponential covalent coordination numbers;
3. overlap integrals only (no GFN2 dipole/quadrupole multipole requirement);
4. coordination-dependent extended-Hückel H0;
5. shell-resolved ES2 plus atom-resolved ES3 SCC electrostatics over explicit
   shell-charge and atomic-charge state;
6. shell spin polarization;
7. fixed-state Hamiltonian assembly;
8. geometry-only GFN1 repulsion;
9. the GFN1 halogen correction as its own geometry primitive;
10. non-self-consistent two-body D3(BJ).

The halogen primitive is now concretely composable with the compiler-owned
TripletIR lowering. Its method-bound program identity includes the resolved
GFN1 MethodIR identity, the exact primitive semantics, the pinned halogen
parameter identity, the triplet topology and the TensorIR equation. This does
not change the method-wide capability boundary below: the remaining GFN1
electronic/runtime graph is still unavailable as a public endpoint.

The canonical parameter-set revision is the SHA-256 identity of
`upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/gfn1.json`. The correction requirements additionally bind
the existing canonical GFN1 D3 reference-table digest rather than duplicating
that table in the xTB parameter product.

## Identity and provenance

Semantic identity includes:

- the XtbMethodIR schema version;
- model flavor and restricted/unrestricted reference;
- parameter-set identifier, revision, source, supported-element scope, and
  required table families;
- every primitive model/dependency/state/derivative contract;
- requested compiler products.

The descriptive method name is excluded from semantic identity so aliases of
the same audited composition can reuse compiler caches. It is retained in
`manifest_identity` for result provenance.

GFN2 retains its published method identity contract. GFN1 binds directly to
the canonical repository snapshot and D3 table digest introduced by #837 so
changing either scientific input changes compiler identity.

## Capability boundary

Resolved GFN1 and GFN2 graphs currently report:

- `compiler_representable = true`;
- `lowering_available = false`;
- `runtime_executable = false`.

This prevents MethodIR construction from being mistaken for a working public
GFN endpoint while generated lowering and runtime SCC/eigensolver work remain
separate. In particular, resolving `GFN1-xTB` means the compiler can identify
and type-check the audited scientific graph; it does **not** mean a
`Calculator(method="gfn1-xtb")` endpoint exists yet.

## Dependency rule

Production VibeQC code in this layer depends only on VibeQC compiler modules.
It does not import xTBloom and does not invoke an external xtb executable or
library. xTBloom may be used independently as a qualification oracle in later
implementation slices.

Agent: ChatGPT
Model: GPT-5.6 Sol

# Methods and long-term scope

VibeQC's long-term mission is to cover **all quantum-chemistry methods** in one
accelerator-native system. This is a roadmap commitment, not a statement of
current availability. Today, only RHF and UHF energies and analytic nuclear
forces are executable.

## Current method status

| Family | Method or capability | Status |
| --- | --- | --- |
| Hartree-Fock | RHF | Implemented: energy and analytic forces |
| Hartree-Fock | UHF | Implemented: energy and analytic forces |
| Hartree-Fock | ROHF, GHF, spinor HF | Planned |
| Density fitting | Two-/three-center integral oracle, first nuclear derivatives, metric conditioning, memory planner | CPU correctness foundation implemented; accelerator RI-J/K and SCF integration planned |
| Density functional theory | LDA, GGA, meta-GGA, hybrid, range-separated, nonlocal correlation | Planned |
| Perturbation theory | MP2 and higher-order variants | Planned |
| Coupled cluster | CCSD, perturbative triples, higher-rank variants | Planned |
| Configuration interaction | CIS, selected CI, truncated and full CI | Planned |
| Multireference | CASCI, CASSCF, internally contracted and selected-space methods | Planned |
| Excited states and response | TDHF, TDDFT, EOM-CC, linear response | Planned |
| Nuclear derivatives and properties | Gradients, Hessians, response properties, spectra | RHF/UHF gradients implemented; broader coverage planned |
| Environments and Hamiltonians | Periodic, embedding, relativistic, ECP, and finite-temperature methods | Planned |

“Planned” records intended architectural coverage, not a release promise or a
fixed implementation order. Method families will be split into independently
testable milestones as their numerical oracles and performance baselines are
defined.

## Acceptance standard

A method becomes supported only when all of the following are true:

1. Its public behavior and mathematical conventions are documented.
2. Energies and relevant derivatives agree with an independent implementation
   over representative systems and basis sets.
3. CPU/GPU execution boundaries and unsupported cases fail explicitly; there
   is no silent fallback to an unvalidated path.
4. Reproducible benchmark artifacts support any performance claim.
5. Batched execution preserves per-system ordering, diagnostics, and failure
   isolation where the method permits batching.

## Expansion strategy

The current HF foundation supplies basis handling, integral validation,
device-resident SCF, analytic gradients, and ragged fleet execution. The first
density-fitting milestone adds a CPU correctness oracle, metric conditioning,
and a memory-bounded tile planner; it does not yet change production SCF
dispatch. Near-term work extends this foundation with accelerator RI-J/K and
broader HF robustness.
Method capability discovery and prepared execution are now registry-driven:
the public API is independent of RHF/UHF dispatch, while each method family
owns its validation, options, retained state, and batch policy.
DFT grids and exchange-correlation response, followed by AO-to-MO transforms
and correlated tensor contractions, open the main DFT and post-HF families.
Multireference, excited-state, periodic, embedding, and relativistic methods
then build on those validated primitives rather than on reserved names alone.

Detailed implementation milestones live in the [roadmap](roadmap.md).

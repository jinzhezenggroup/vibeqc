# Methods and long-term scope

VibeQC's long-term mission is to cover **all quantum-chemistry methods** in one
accelerator-native system. Current executable method identity is registry-driven:
the canonical native names, aliases, declared properties, and batch capability
are generated from `manifests/public_methods.json` into the
[public method table](../public_methods.md). That generated table is authoritative
for public native discovery; backend-, basis-, grid-, and model-specific
execution constraints remain method-specific and fail closed.

## Current method status

| Family | Method or capability | Status |
| --- | --- | --- |
| Hartree-Fock | RHF | Implemented: energy and analytic forces |
| Hartree-Fock | UHF | Implemented: energy and analytic forces |
| Hartree-Fock | ROHF, GHF, spinor HF | Planned |
| Density fitting | Two-/three-center integral oracle, first nuclear derivatives, metric conditioning, memory planner | CPU oracle plus CUDA-native batched integral generation, RI-J/K, raw two-electron force-response contractions, and device-resident SCF integration implemented; streamed host tiles and provider-dependent Graph replay are documented acceptance-boundary modes |
| Density functional theory | LDA/PBE RKS/UKS | Public native energy selectors with batching; backend/grid qualification remains explicit |
| Density functional theory | r2SCAN RKS/UKS | Public native energy selectors with batching; meta-GGA execution uses the audited r2SCAN composition |
| Density functional theory | PBE0/B3LYP RKS/UKS | Public native energy selectors with batching; global-hybrid backend/grid constraints fail closed |
| Density functional theory | PBE-D4 RKS | Public native energy selector with batching and method-owned D4 composition |
| Density functional theory | wB97M-V and broader range-separated/nonlocal coverage | `wb97m-v` retains a reserved public identity but has no executable provider; broader coverage remains planned |
| Perturbation theory | Closed-shell MP2 | Conventional energy and analytic forces implemented on CPU/CUDA; RI energy implemented on CPU/CUDA; RI analytic forces remain C2 work |
| Perturbation theory | Open-shell, frozen-core, ECP and higher-order variants | Planned |
| Coupled cluster | RCCSD and RCCSD(T) | Public native energy endpoints are registered; qualified execution boundaries remain method-specific |
| Coupled cluster | Higher-rank and broader open-shell variants | Planned |
| Configuration interaction | CIS, selected CI, truncated and full CI | Planned |
| Multireference | CASCI, CASSCF, internally contracted and selected-space methods | Planned |
| Excited states and response | TDHF, TDDFT, EOM-CC, linear response | Planned |
| Nuclear derivatives and properties | Gradients, Hessians, response properties, spectra | RHF/UHF gradients implemented; broader coverage planned |
| Scalar Gaussian ECP | Local/nonlocal residuals and complete direct RHF/UHF gradients | Bounded CPU/CUDA s/p/d baseline; [contract](ecp.md) |
| Environments and Hamiltonians | Periodic, embedding, relativistic, and finite-temperature methods | Planned |

“Planned” records intended architectural coverage, not a release promise or a
fixed implementation order. Method families will be split into independently
testable milestones as their numerical oracles and performance baselines are
defined.

## Direct CUDA HF force state

Direct RHF/UHF force solves require the maximum physical AO commutator
`|F P S - S P F|` to meet `min(1e-8, density_tolerance)` in addition to the
energy and density-update criteria. The maximum includes both UHF spins;
nonfinite residuals cannot pass. Additional SCF updates remain within the
requested iteration limit and appear in the public iteration count.
An item's mixed-precision coarse stage keeps its previous stop; the exact FP64
refinement and exact-precision neighbors apply the physical force criterion.

Finalization projects the final physical-Fock orbitals to a determinant,
rebuilds its physical Fock, and rechecks that determinant's commutator before
publishing forces. Energy, forces and the returned warm state share this P/F(P).
The Pulay weight is `P F(P) P / 2` for RHF and `P_sigma F_sigma P_sigma` for
UHF. A failed final residual returns nonconvergence. Energy-only and detached
physical-reference execution retain their existing finalization contracts.

This force finalization adds one density projection, one physical Fock build,
four matrix products for residual validation and two for the Pulay weight.
Existing device scratch holds the products; a four-byte work counter is copied
at the existing completion fence. `VIBEQC_DF_PROGRESS_TRACE` records final
updates, physical Fock builds, residual checks and rejections separately from
iterative SCF updates. Complete endpoint costs include all this work. The
[decision record](../../.agents/notes/proposed/2026-09-17-consistent-direct-pulay-weight.md)
retains the independent diagnosis and qualification boundaries.

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

VibeQC expands method coverage behind the same registry-driven prepared
calculation interface. New public capabilities should reuse the existing basis,
integral, SCF, batching, resource-planning, and diagnostic contracts rather than
introducing method-specific API branches.

Near-term development focuses on completing scientific and backend coverage of
the method families already present in the registry: broader DFT execution and
analytic derivatives, density-fitted derivatives, correlated-method
qualification and derivatives, and wider basis/ECP/backend coverage. Compiler
and runtime work should remove duplicated handwritten execution paths while
preserving explicit numerical ownership and fail-closed unsupported cases.

A capability is promoted only after the [acceptance standard](#acceptance-standard)
is satisfied. Implementation details belong in the
[Developer Guide](../developer/index.md), performance qualification in the
[Maintainer Guide](../maintainer/index.md), and durable historical rationale in
`.agents/notes/`.

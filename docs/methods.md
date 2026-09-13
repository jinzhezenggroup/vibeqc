# Methods and long-term scope

VibeQC's long-term mission is to cover **all quantum-chemistry methods** in one
accelerator-native system. This is a roadmap commitment, not a statement of
current availability. RHF and UHF provide energies and analytic nuclear forces;
closed-shell MP2 and CPU LDA/PBE RKS/UKS slices provide energy-only execution.

## Current method status

| Family | Method or capability | Status |
| --- | --- | --- |
| Hartree-Fock | RHF | Implemented: energy and analytic forces |
| Hartree-Fock | UHF | Implemented: energy and analytic forces |
| Hartree-Fock | ROHF, GHF, spinor HF | Planned |
| Density fitting | Two-/three-center integral oracle, first nuclear derivatives, metric conditioning, memory planner | CPU oracle plus CUDA-native batched integral generation, RI-J/K, raw two-electron force-response contractions, and device-resident SCF integration implemented; streamed host tiles and provider-dependent Graph replay are documented acceptance-boundary modes |
| Density functional theory | LDA RKS | Implemented vertical slice: CPU energy only, closed shell, conventional J; independent matched-grid SCF endpoint accepted for H2 and He |
| Density functional theory | PBE RKS | Limited CPU energy-only slice: closed shell, conventional J, exact interior PBE and versioned LDA fallback tail; independent matched-grid SCF endpoint accepted for H2 |
| Density functional theory | LDA/PBE UKS | Limited CPU energy-only slices: independent spin densities, total-density conventional J and versioned spin-tail policies; matched-grid H2- doublet and fully polarized H2+ endpoints accepted |
| Density functional theory | meta-GGA, hybrid, range-separated, nonlocal correlation | Planned |
| Perturbation theory | Closed-shell MP2 | Conventional and RI energy implemented on CPU/CUDA; analytic forces planned |
| Perturbation theory | Open-shell, frozen-core, ECP and higher-order variants | Planned |
| Coupled cluster | CCSD, perturbative triples, higher-rank variants | Planned |
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
device-resident SCF, analytic gradients, and ragged fleet execution. The
density-fitting milestone adds a CPU correctness oracle, metric conditioning,
a memory-bounded tile planner, and CUDA RI-J/K with a device-resident SCF
iteration path (plus an explicit host fallback for provider limitations).
Near-term work extends this foundation with accelerator-native integral and
force-response kernels and broader HF robustness.
Method capability discovery and prepared execution are now registry-driven:
the public API is independent of RHF/UHF dispatch, while each method family
owns its validation, options, retained state, and batch policy.
The native LDA and PBE RKS/UKS paths compose versioned atom-centered grids,
generated XC, the common Coulomb provider and host SCF. Their small matched-grid
endpoints pass the independent PySCF/Libxc gates recorded under the issue-162-a
and issue-0162-b validation records. These records cover only CPU energy-only
closed-shell H2/He and open-shell H2-/H2+/OH slices; broader DFT still requires
representative systems, prepared CUDA, resource planning and gradients.
The public `Result.density_rms` retains its density-update convergence meaning.
The separate `Result.physical_residual_rms` reports the physical commutator
RMS for these KS methods; UKS combines the alpha/beta matrix entries in one RMS.
The additive C query `vibeqc_calculation_get_scf_diagnostic` returns both
measures without changing the existing result descriptor layout. The physical
measure is unavailable (`None` in Python) for methods that do not report it
and for older native libraries. Both measures participate in KS convergence.
AO-to-MO transforms and correlated tensor contractions open the post-HF
families.
The RI-MP2 endpoint uses an RHF reference built with the same thresholded
density-fitting Hamiltonian as its correlation integrals. The auxiliary Coulomb
metric uses a square symmetric inverse square root with eigenvalues at or below
`density_fitting_relative_threshold * lambda_max` removed. The implementation
forms no four-index AO ERI tensor or T2 amplitude tensor and rejects requests
whose reference or RI transformation capacity exceeds the configured budget.
Multireference, excited-state, periodic, embedding, and relativistic methods
then build on those validated primitives rather than on reserved names alone.

Detailed implementation milestones live in the [roadmap](roadmap.md).

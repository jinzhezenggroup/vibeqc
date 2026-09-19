# Methods and long-term scope

VibeQC's long-term mission is to cover **all quantum-chemistry methods** in one
accelerator-native system. This is a roadmap commitment, not a statement of
current availability. RHF and UHF provide energies and analytic nuclear forces.
Closed-shell conventional MP2 provides CPU/CUDA energies and analytic forces,
while RI-MP2 and CPU/CUDA LDA/PBE RKS/UKS slices provide energy-only execution.

## Current method status

| Family | Method or capability | Status |
| --- | --- | --- |
| Hartree-Fock | RHF | Implemented: energy and analytic forces |
| Hartree-Fock | UHF | Implemented: energy and analytic forces |
| Hartree-Fock | ROHF, GHF, spinor HF | Planned |
| Density fitting | Two-/three-center integral oracle, first nuclear derivatives, metric conditioning, memory planner | CPU oracle plus CUDA-native batched integral generation, RI-J/K, raw two-electron force-response contractions, and device-resident SCF integration implemented; streamed host tiles and provider-dependent Graph replay are documented acceptance-boundary modes |
| Density functional theory | LDA RKS | CPU/CUDA single-system and native ragged energy only, closed shell, conventional J; independent matched-grid H2, He and water gates |
| Density functional theory | PBE RKS | CPU/CUDA single-system and native ragged energy only, closed shell, conventional J; scaled PBE tail algebra and independent matched-grid H2, He and water gates |
| Density functional theory | LDA/PBE UKS | CPU/CUDA single-system and native ragged energy only, independent spins, conventional J; matched-grid H, Li and H2+ gates; explicit PBE spin boundary policy |
| Density functional theory | Meta-GGA, hybrid, range-separated, nonlocal correlation | Planned |
| Perturbation theory | Closed-shell MP2 | Conventional energy and analytic forces implemented on CPU/CUDA; RI energy implemented on CPU/CUDA; RI analytic forces remain C2 work |
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
The native LDA/PBE RKS and UKS paths compose versioned atom-centered grids,
XC, the common Coulomb provider and CPU/native CUDA SCF. The
[SCF point-domain contract](xc_scf_domain.md) specifies the exact compositions,
stable tail algebra and explicit PBE spin endpoint extension. The issue-0162-b
record retains the independent CPU H2-/H2+/OH UKS endpoints. Additional
CPU/CUDA matched-grid tests use two PySCF initial guesses; native tests enforce
physical residuals, spin populations and stale-input rejection. Native prepared
ragged batches preserve per-item status/order and last-good densities through
geometry rebuilds, failure, frozen updates and atomic seed import. CUDA batches
enqueue all active item streams before reading their scalar results. Public
KS resource plans cover preparation, execution and geometry rebuilds through
the common ledger; see [resource planning](resource_planning.md). Immutable
[KS options](ks_options.md) bind the functional, grid/radii and tile schedule
to native preparation and resource identity. Richer diagnostics and workload
evidence remain open parts of #162.
`tools/validate_dft_endpoints.py` records method-resolved cold, warm and changed
geometry phases against independent CPU energies and physical residual gates.
The resident owner replaces #307's host-controlled method path; its compiler
spatial-XC consumer remains available with the separate `interior-v1` contract.
Nuclear gradients remain #163. No DFT density-fitting or performance-leadership
claim follows from the tested CPU/CUDA energy endpoints.
The internal #163-A groundwork reads the actual #162 CUDA LDA/PBE RKS/UKS
snapshot through `StationaryKsState.from_native(batch, basis)`. It verifies the
native overlap, normalized AO basis and explicit grid, and checks the live
owner token whenever a stationary contract is validated. Batch replay (including
failed requests), owner replacement and closure revoke old snapshots. Directly
constructed array records provide no stationary authorization.

Issue #163-A now binds that live native state to generated XC AO-centre,
point and weight geometry pullbacks without relabeling the energy model. A
private CPU point bridge reuses the exact
`semilocal-scaled-v1/pbe-spin-c2-1e-18` SCF evaluator for per-point energy and
Cartesian density/gradient coefficients; the compiler-generated AO-jet pullback
owns the density/AO geometric chain. The bridge is checked against all 97
independent Libxc/mpmath SCF-domain reference points, while the existing
multistep fixed-density oracle and LDA/PBE RKS/UKS component/directional tests
continue to exercise the geometric chain. `interior-v1` remains a separate
diagnostic contract and cannot authorize a native state.

This completes only the #163-A stationary derivative contract / XC geometry
slice. Atom-centred grid/partition response, the other energy terms and Pulay
contributions, complete molecular gradients, CUDA gradient lowering, and public
DFT forces remain unsupported under #163 B/C. See the
[stationary XC ownership decision](../.agents/notes/implemented/numerics/2026-09-18-scf-point-generated-pullback.md).
The public `Result.density_rms` retains its density-update convergence meaning.
The separate `Result.physical_residual_rms` reports the physical commutator
RMS; UKS combines the alpha/beta matrix entries in both public RMS measures.
The additive C queries `vibeqc_calculation_get_scf_diagnostic` and
`vibeqc_batch_get_scf_diagnostic` return both measures without changing the
existing result descriptor layouts or batch array stride. The physical
measure is unavailable (`None` in Python) for methods that do not report it
and for older native libraries. KS internally gates each spin separately, so
an empty spin cannot dilute an unconverged channel.
AO-to-MO transforms and correlated tensor contractions open the post-HF families.
The conventional MP2 force endpoint adds a bounded adjoint, true-residual RHF
response, relaxed weights, shell-local CPU/CUDA derivative contraction, and
transactional single/batch publication. Exact support and evidence are
documented in [Canonical RHF-MP2](mp2.md). The RI-MP2 endpoint uses an RHF
reference built with the same thresholded
density-fitting Hamiltonian as its correlation integrals. The auxiliary Coulomb
metric uses a square symmetric inverse square root with eigenvalues at or below
`density_fitting_relative_threshold * lambda_max` removed. The implementation
forms no four-index AO ERI tensor or T2 amplitude tensor and rejects requests
whose reference or RI transformation capacity exceeds the configured budget.
Multireference, excited-state, periodic, embedding, and relativistic methods
then build on those validated primitives rather than on reserved names alone.

Detailed implementation milestones live in the [roadmap](roadmap.md).

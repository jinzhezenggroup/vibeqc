# VibeQC documentation

VibeQC is an experimental, accelerator-native quantum-chemistry engine. The
public executable registry currently spans Hartree-Fock, density-functional,
perturbation, coupled-cluster, and semiempirical method families. See the
[generated public method table](public_methods.md) for canonical names and
declared properties; method-specific backend and model constraints fail closed.

## Users

- [README](../README.md): build, install, supported features, and minimal use.
- [Batched HF](batched_hf.md): batch semantics, warm starts, backend labels,
  and verification.
- [Portable HF checkpoints](checkpoint.md): restart across processes and backends,
  strict scientific compatibility, corruption handling and per-item restore.
- [Cross-basis HF initialization](basis_projection.md): rectangular overlaps,
  occupied-space projection and complete source-plus-target cost accounting.
- [External basis data](external_basis.md): offline input, provenance, element/ECP
  bookkeeping, per-operator limits and prepared-state identities.
- [Scalar Gaussian ECPs](ecp.md): local/nonlocal integrals, complete direct HF
  gradients, supported domains and empirical quadrature convergence checks.
- [Methods](methods.md): implemented methods, planned method families, and the
  acceptance standard for enabling new capabilities.
- [Public native methods](public_methods.md): generated canonical method names,
  aliases, properties, batch capability, and availability.
- [Local autotuning](local_autotuning.md): optional workload-first CUDA tuning,
  compatible profile reuse, diagnostics, and homogeneous cluster export/import.
- [Density fitting](density_fitting.md): milestone-1 correctness and planning
  foundation, plus the production features that remain in issue #5.
- [Accuracy evidence](accuracy.md): resolved HF models, observable targets and
  explicit reference comparisons, independent of iteration convergence.

## Developers

- [Architecture](architecture.md): scientific and runtime design decisions.
- [Scientific compiler ownership](compiler_architecture.md): package boundaries,
  source generation, compatibility and dependency checks.
- [Electronic method orchestration IR](electronic_method_ir.md): shared HF/KS/RCCSD
  state, operator dataflow, iteration contracts, and lower-level IR identities.
- [Experimental OpenCL contracts](opencl_backend.md): optional compiler/runtime
  execution, queried capabilities and the boundary before native HF integration.
- [Shared Fock construction](fock_build.md): method-neutral J/K requests, exact
  HF consumers, capability limits, and prepared strategy identity.
- [Implementation roadmap](roadmap.md): detailed milestones and measured
  implementation history.
- [Shell code generation](shell_codegen.md): generated CUDA policy and
  correctness model.
- [One-electron value generation](one_electron_codegen.md): S/T/V DAGs, native
  contraction schedules, same-binary selectors and independent validation.
- [One-electron derivatives](one_electron_derivatives.md): generic weighted
  gradients, bounded CUDA contractions and Direct/DF HF adapters.
- [Density-fitting derivatives](df_derivatives.md): generic A/M responses,
  bounded HF weights, metric subspace response, and fused CUDA contractions.
- [DF-CCSD(T) gradient composition](df_ccsdt_gradient.md): factorized B
  cotangents, raw A/M pullback, fixed-rank metric response, and #158 boundaries.
- [Integral IR contracts](integral_ir.md): operator centers, bounded raw blocks,
  external weights, serialization, and backend capability boundaries.
- [TensorIR](tensor_ir.md): typed tensor equations, exact factors, symmetry-packed
  amplitudes, CPU interpretation, conservative rewrites, and replay.
- [TensorIR CUDA execution](tensor_cuda.md): prepared FP64 contractions, memory
  budgets, shape buckets, bounded tuning, and complete tensor endpoint evidence.
- [StationaryProblem composition](stationary_problem.md): explicit residuals,
  constraints, provider dependencies and generated first-order source plans.
- [Validation gates](validation.md): pinned conventions and references,
  independent oracles, tiered execution, and shared evidence registration.
- [F-shell validation](f_shell_validation.md): all-34 source/numerical gates,
  release resource tiers, actual f-containing endpoints, and promotion policy.
- [Benchmark evidence](../benchmarks/results/README.md): comparison boundary,
  reproducibility rules, gates, and archived results.

The README intentionally omits kernel history and internal scheduling details.
Those belong in the developer documents so the first page remains an accurate,
compact user entry point.

- [HF reference snapshots and bounded MO integral providers](posthf.md)
- [RCCSD state transport](state_transport.md): complete compatibility identities,
  orbital-frame diagnostics and exact T1/T2 rotations.
- [Conventional CPU RCCSD equations and solver](rccsd_bc.md)
- [Generated RCCSD Lambda equation actions](rccsd_lambda.md)
- [Localized occupied and pair-natural-orbital spaces](local_spaces.md)
- [Safe SCF proposals, local traces and replay](scf_proposals.md)
- [Shared orbital response and bounded Krylov solves](response.md)
- [Atom-centered grids and spatial AO/density jets](dft_grid.md)
- [Fixed-density LDA/PBE XC integration and AO potentials](xc_integration.md)

- [Fixed-amplitude GPU RCCSD validation](rccsd_gpu.md)

- [Scientific evidence retention and publication](evidence_retention.md)

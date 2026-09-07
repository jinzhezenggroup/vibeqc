# VibeQC documentation

VibeQC is an experimental, accelerator-native quantum-chemistry engine. The
current executable method scope is RHF and UHF; the long-term mission is to
cover all quantum-chemistry methods through a coherent batched interface.

## Users

- [README](../README.md): build, install, supported features, and minimal use.
- [Batched HF](batched_hf.md): batch semantics, warm starts, backend labels,
  and verification.
- [Methods](methods.md): implemented methods, planned method families, and the
  acceptance standard for enabling new capabilities.
- [Density fitting](density_fitting.md): milestone-1 correctness and planning
  foundation, plus the production features that remain in issue #5.

## Developers

- [Architecture](architecture.md): scientific and runtime design decisions.
- [Implementation roadmap](roadmap.md): detailed milestones and measured
  implementation history.
- [Shell code generation](shell_codegen.md): generated CUDA policy and
  correctness model.
- [Integral IR contracts](integral_ir.md): operator centers, bounded raw blocks,
  external weights, serialization, and backend capability boundaries.
- [TensorIR](tensor_ir.md): typed tensor equations, exact factors, symmetry-packed
  amplitudes, CPU interpretation, conservative rewrites, and replay.
- [Validation gates](validation.md): pinned conventions and references,
  independent oracles, tiered execution, and shared evidence registration.
- [F-shell validation](f_shell_validation.md): all-34 source/numerical gates,
  release resource tiers, actual f-containing endpoints, and promotion policy.
- [Benchmark evidence](../benchmarks/results/README.md): comparison boundary,
  reproducibility rules, gates, and archived results.

The README intentionally omits kernel history and internal scheduling details.
Those belong in the developer documents so the first page remains an accurate,
compact user entry point.

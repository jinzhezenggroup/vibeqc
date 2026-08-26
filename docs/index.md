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

## Developers

- [Architecture](architecture.md): scientific and runtime design decisions.
- [Implementation roadmap](roadmap.md): detailed milestones and measured
  implementation history.
- [Shell code generation](shell_codegen.md): generated CUDA policy and
  correctness model.
- [Benchmark evidence](../benchmarks/results/README.md): comparison boundary,
  reproducibility rules, gates, and archived results.

The README intentionally omits kernel history and internal scheduling details.
Those belong in the developer documents so the first page remains an accurate,
compact user entry point.

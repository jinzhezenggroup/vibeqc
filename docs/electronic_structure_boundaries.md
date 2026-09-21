# Electronic-structure architecture boundaries

VibeQC shares native infrastructure across HF, DFT, post-HF and coupled-cluster
methods. This document records the top-level dependency direction protected by
`tools/check_electronic_structure_boundaries.py`.

## Shared native owners

The following directories are method-neutral owners:

- `src/core/`
- `src/runtime/`
- `src/tensor/`
- `src/response/`

They may depend on other shared chemistry, integral and runtime primitives, but
must not include concrete method implementation headers from:

- `src/scf/`
- `src/dft/`
- `src/posthf/`
- `src/cc/`

The check resolves repository-relative include spellings, including `../`
paths, before applying the rule. Commented-out examples do not create edges.

Two existing reverse edges are carried as explicit debt ceilings rather than
silently exempted:

- `runtime/cuda_runtime.cu -> scf/aot_shell_registry.hpp`
- `runtime/host_component_trace.hpp -> scf/reference/observation.hpp`

They may disappear without a coordinated baseline edit; any additional
shared-to-method edge fails the check.

This cross-method gate complements, rather than replaces, the more detailed
`tools/check_scf_structure.py` and `tools/check_compiler_structure.py`
ownership checks.

## Known duplicate infrastructure debt

The current CC solver still owns local CPU DIIS, a small dense linear solve used
by that DIIS implementation, and a CUDA DIIS orchestration entry point. These
are tracked as explicit debt while the shared iterative-solver work proceeds.
The architecture gate allows the current count to shrink but fails if any of
those owners multiply.

The debt baseline is intentionally a ceiling, not a target. Removing an owner
does not require a synchronized baseline update to keep CI green.

## Metrics

Run:

```console
python tools/check_electronic_structure_boundaries.py --json
```

to obtain:

- shared-module dependency edges,
- source files/lines/bytes by top-level native area,
- generated-line counts separated from total lines, and
- locations/counts for the tracked duplicate solver infrastructure.

The metrics are inventory only. They do not impose arbitrary line-count gates;
architectural direction and duplicate-owner growth are the enforced invariants.

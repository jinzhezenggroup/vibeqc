# Electronic-structure architecture boundaries

VibeQC shares native infrastructure across HF, DFT, post-HF and coupled-cluster
methods. This document records the top-level dependency direction protected by
`tools/check_electronic_structure_boundaries.py`.

## Shared native owners

The following directories are method-neutral owners:

- `src/core/`
- `src/runtime/`
- `src/tensor/`
- `src/solver/`
- `src/response/`

They may depend on other shared chemistry, integral and runtime primitives, but
must not include concrete method implementation headers from:

- `src/scf/`
- `src/dft/`
- `src/posthf/`
- `src/cc/`

The check resolves repository-relative include spellings, including `../`
paths, before applying the rule. It scans `.cpp`, `.hpp`, `.h`, `.cu` and
`.cuh` files. Commented-out examples do not create edges.

The method-neutral electron-interaction interface lives in the otherwise mixed
`src/integrals/` area. The checker therefore protects
`src/integrals/electron_interaction_source.hpp` as an individual provider
contract rather than labeling every integral implementation as shared.

Two existing reverse edges are carried as explicit debt ceilings rather than
silently exempted:

- `runtime/cuda_runtime.cu -> scf/aot_shell_registry.hpp`
- `runtime/host_component_trace.hpp -> scf/reference/observation.hpp`

They may disappear without a coordinated baseline edit; any additional
shared-to-method edge fails the check.

Post-HF still has 20 direct dependencies on SCF-owned driver, derivative,
provider and type headers. Those exact edges are also explicit shrink-only debt:
removing one is accepted, while any new `src/posthf/ -> src/scf/` edge fails at
the include location. This prevents the shared-reference migration from moving
backward without pretending the current migration is already complete.

This cross-method gate complements, rather than replaces, the more detailed
`tools/check_scf_structure.py` and `tools/check_compiler_structure.py`
ownership checks.

The current Infrastructure A--F boundary mapping is:

| Slice | Shared owner | Enforced boundary |
| --- | --- | --- |
| A: iteration / DIIS | `src/solver/` | directory cannot depend on a concrete method; CC and SCF remain include-graph consumers |
| B: electronic reference | `src/core/electronic_reference.hpp` | `src/core/` cannot depend on a concrete method; SCF and DFT remain consumers |
| C: interaction provider | `src/integrals/electron_interaction_source.hpp` | the individual provider contract cannot depend on a concrete method; SCF and post-HF remain consumers |
| D: MethodIR / ExecutionIR | `python/vibeqc_compiler/method/` | `tools/check_compiler_structure.py` enforces compiler-package directions |
| E: linear response | `src/response/` | directory cannot depend on a concrete method; post-HF remains a transitive consumer |
| F: runtime / workspace | `src/runtime/` | directory cannot acquire new concrete-method edges; the two existing reverse edges remain explicit debt |

For the native facilities in this table, the JSON report derives direct and
transitive consumers separately from resolved repository include edges. A
missing canonical owner or required consumer fails in a full checkout. This
catches a method silently forking away from a shared facility without depending
on source-text formatting or a line-count threshold.

## Duplicate infrastructure inventory

CC host bounded iteration moved to `src/solver/` in #920, and host DIIS plus its
small dense linear solve moved there in #922. Their duplicate-debt ceilings are
now zero, so reintroducing any of those generic host owners under `src/cc/`
fails the architecture gate. One method-specific CUDA DIIS orchestration owner
remains in `src/cc/cuda_solver.cu`; its ceiling is one until a shared device-loop
contract is justified.

Duplicate declarations are located with a comment- and literal-aware C++ token
scan that distinguishes definitions from calls, forward declarations, template
parameters and inheritance. The one CUDA exception is restricted to
`src/cc/cuda_solver.cu`; moving that identified owner fails even if the total
count stays at one. Failures name every matching `path:line`.

This named-declaration inventory is not a proof that arbitrary renamed code is
semantically DIIS. Canonical owner/consumer checks prevent CC from silently
replacing the shared host facilities, while infrastructure PR review remains
responsible for identifying an additional or fundamentally different duplicate
algorithm. The debt baseline is a ceiling, not a target: removing the remaining
CUDA owner does not require a synchronized baseline update to keep CI green.

## Regression inventory

The report carries stable, method-specific test-file pointers for the required
regression stories:

- RHF/UHF energy, DFT energy/XC, and RCCSD energy/amplitude equations;
- DFT, post-HF and CC forces;
- shared GMRES plus UHF, RKS, CC orbital response and molecular HVP behavior;
- runtime workspace and CC numeric-capacity accounting; and
- compiler and native runtime dependency audits.

These entries are an ownership/consumer index, not a replacement for executing
the tests. In a full checkout, removing an indexed gate without updating the
architecture inventory is CI-blocking. Infrastructure PRs must still state the
affected endpoints, numerical tolerances or unchanged-behavior evidence, and
resource-accounting evidence relevant to their actual change.

## Metrics

Run:

```console
python tools/check_electronic_structure_boundaries.py --json
```

to obtain:

- shared-module dependency edges,
- canonical shared-infrastructure owners plus direct/transitive consumers,
- numerical/resource/dependency regression test pointers,
- source files/lines/bytes by top-level native area,
- diagnostic shared/HF-only/DFT-only/CC-only ownership rollups,
- generated-line counts separated from total lines, and
- locations/counts for the tracked duplicate solver infrastructure.

The size metrics are inventory only. They do not impose arbitrary line-count
gates; architectural direction, required owner/consumer relationships,
regression-gate presence and duplicate-owner growth are the enforced invariants.

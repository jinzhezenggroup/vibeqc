# Automatic Libxc capability claims

The bulk Libxc importer has a machine-readable qualification layer in
`vibeqc_compiler.xc.libxc_bulk_capabilities`.

It separates **representation**, **runtime qualification**, and **public
admission**. Every imported registration starts with two intrinsic claims:

- `graph-imported`: the pinned Libxc Maple owner lowers to the canonical Graph;
- `pointwise-validated`: polarized and unpolarized energy, `vxc`, and packed
  `fxc` agree with independent PySCF 2.14.0 / Libxc 7.0.0 fixtures on
  `libxc-bulk-interior/v1`.

C and CUDA source emitters are also available, but source emission is not
compilation or runtime evidence.

## Evidence-driven promotion

Higher stages are no longer inferred from the pointwise claim. They are
computed from identity-bound evidence envelopes and a fail-closed dependency
DAG:

```text
graph-imported
  -> pointwise-validated
       |-> compiled-cpu -----------|
       |-> compiled-cuda -> gpu-runtime
       |-> production-domain -----|-> molecular-scf
                                      |-> forces
                                      |-> response
                                      `-> public-method
```

`molecular-scf` requires `production-domain` plus either a qualified CPU
binary or a qualified CUDA runtime. `forces` and `response` are independent
branches. A public energy method therefore does not incorrectly require
response support.

Each non-intrinsic pass must use
`vibeqc.libxc-bulk-stage-evidence.v1`, bind to the exact capability identity,
name its stage, and point to a non-empty retained evidence reference. The subject
binds parameter bindings, pinned upstream/source digests, and the compiler source
inventory under stable logical paths. A source or binding change invalidates old
evidence; bulk queries hash the source inventory once and do not lower all graphs. Missing,
failed, malformed, cross-functional, or out-of-order evidence never promotes a
stage.

A `production-domain` pass has one additional hard requirement. It must attach
the exact `vibeqc.libxc-production-domain-profile.v3` qualification payload
computed from the registration family and required ingredients. The current
`semilocal-boundary-matrix/v3` profile requires both spin layouts and
first-order energy/vxc coverage. Density, gradient, tau, and control cases apply
to both layouts, while alpha/beta zero-spin and full-polarization cases apply
only to the polarized layout. The exact cases-by-spin matrix is part of the
profile identity, so dropping or moving a case, changing the matrix version, or
changing an ingredient invalidates the admission proof. Registrations requiring
ingredients outside the current generic `rho/sigma/tau` domain (for example,
Laplacian-dependent meta-GGAs) carry an explicit structural blocker and do not
appear as `production-domain` ready.

This first-order scope is deliberate. The intrinsic `pointwise-validated`
claim still includes packed fxc on the audited interior domain, but exact
vacuum/full-spin endpoints need not possess a finite full feature Hessian.
Production energy, SCF, public-method, and stationary first-gradient admission
therefore do not infer response capability. Endpoint fxc/CPKS qualification
remains an independent `response` stage with its own evidence.

## Query

```python
from vibeqc_compiler.xc.libxc_bulk_capabilities import (
    STAGE_EVIDENCE_SCHEMA,
    claimable_functionals,
    functional_capability,
)

base = functional_capability("GGA_X_PBE_SOL")
compiled_cpu = {
    "schema": STAGE_EVIDENCE_SCHEMA,
    "subject_identity": base.identity,
    "stage": "compiled-cpu",
    "status": "pass",
    "reason": None,
    "evidence": "ci://xc/GGA_X_PBE_SOL/compiled-cpu.json",
}
qualified = functional_capability(
    base.name,
    evidence={"compiled-cpu": compiled_cpu},
)

print(qualified.qualified_stages)
print(qualified.ready_stages)
```

With no attached higher-stage evidence, the current 221-registration inventory
remains exactly `pointwise-validated`. Queries for known higher stages return
an empty inventory rather than manufacturing a claim; unknown stage names still
fail explicitly.

`ready_stages` is the machine-readable qualification frontier. CI producers
can use it to decide which evidence jobs are meaningful next, while the registry
itself remains a pure, deterministic admission evaluator.

## Compiled CPU point evidence

`tools/qualify_libxc_compiled_cpu.py` produces the concrete
`compiled-cpu` evidence used by the automatic semilocal path. It builds the
polarized first-order `libxc-bulk-production-candidate/v1` Graph, binds it to
the native `SemilocalPointProgram` ABI, derives native domain version 2 from
that compiler-owned domain, then compiles and executes one C++ translation unit.

A pass binds all of the following into
`vibeqc.libxc-compiled-cpu-result/v1`:

- exact functional capability identity;
- `vibeqc.libxc-bulk-point-program-binding/v3` identity and payload;
- point-expression and emitted-artifact identities;
- compiler executable hash and version;
- generated translation-unit hash;
- compiled executable hash; and
- a deterministic native point smoke input plus expected/observed output vectors.

The smoke comparison is an **artifact execution** check against the exact bound
Graph. It is not the independent scientific oracle: boundary correctness remains
owned by the production-domain Libxc campaign. Conversely, successful
production-domain evidence is not proof that a C++ artifact compiled or ran.

Compilation failure, unavailable compiler, native metadata mismatch, nonfinite
output, or numerical mismatch remains explicit fail/not-run stage evidence.
Passing `compiled-cpu` alone grants neither production-domain, molecular-SCF,
force, response, nor public-method capability.

## Bulk inventory queries

```python
evidence = {
    "GGA_X_PBE_SOL": {
        "compiled-cpu": compiled_cpu,
    }
}

assert claimable_functionals("compiled-cpu", evidence) == ("GGA_X_PBE_SOL",)
assert claimable_functionals("compiled-cuda", evidence) == ()
```

The evidence inventory is keyed by Libxc registration. Unknown or blocked
registrations are rejected rather than silently ignored.

## Scope

This registry still does **not** generate or mutate the public method manifest.
A semilocal component may be runtime-qualified without being a complete DFT
method. Hybrid/range-separated/nonlocal/correction primitives must be qualified
at the MethodIR/provider layer before a method-level admission producer may
attach `public-method` evidence.

The existing bulk numerical test continues to cover every intrinsic pointwise
claim. The promotion tests additionally enforce that stages cannot jump their
prerequisites, CPU and CUDA runtime branches are alternatives for SCF execution,
and public admission never appears merely because source code exists.

## Boundary qualification

`vibeqc_compiler.xc.boundary` owns a deterministic physical probe suite for
zero-spin, near-zero-spin, zero-gradient and tail limits.  The probe
coordinates are compiler inputs only.  Independent expected values are
generated by `tools/generate_libxc_boundary_reference.py`, which calls the
Libxc 7.0.0 C API through PySCF and never imports a VibeQC Graph or generated
kernel.

Boundary admission is intentionally split into two facts:

1. the generated expression stays finite at the probe;
2. energy and physical-feature derivatives agree with the independent oracle
   over the exact versioned production-domain profile.

Only a retained evidence record covering the **entire required profile** may
satisfy the `production-domain` stage. A finite value by itself is not a
correctness claim. Oracle-nonfinite points remain unqualified instead of being
coerced into a pass.

The receipt additionally binds the exact first-order execution programs used by
the campaign for both spin layouts: executor kind, runtime domain, imported
source identity, expression identity, optimization mode, feature ABI, and the
complete energy/vxc output contract. This binding is content-addressed and
included in the receipt hash. A matrix cannot be retained as production-domain
evidence without naming the exact mathematical execution that produced its
candidate values. Backend compilation/runtime qualification remains a separate
stage; the current B1 campaign explicitly records the shared array-Graph
executor rather than pretending that interpreted evidence is a compiled-CPU or
CUDA result.

The B1 campaign uses the explicit
`libxc-bulk-production-candidate/v1` runtime domain. It differs from the default
`libxc-bulk-interior/v1` only by admitting physical zero sigma into candidate
evaluation. Negative sigma, non-positive density, non-positive tau, non-PSD
polarized sigma Gram matrices, and nonfinite inputs remain rejected. Admission of
zero sigma is **not** a generic pass: the imported first-order Graph must still
produce finite E/vxc and match the independent Libxc oracle for that exact
functional. Functionals whose imported algebra has a true or unresolved
zero-gradient first-derivative singularity therefore remain blocked by their
numerical matrix row.

Ordinary bulk runtime consumers continue to default to the original interior
domain; the production-candidate domain is selected explicitly by the evidence
campaign and is part of its execution identity.

`vibeqc_compiler.xc.production_domain_cases` instantiates every numerical
rho/sigma/tau row in the exact v3 cases-by-spin matrix from finite physical
density, Cartesian-gradient, and kinetic-density coordinates.
`tools/qualify_libxc_production_domain.py` evaluates those rows through the
generic first-order bulk candidate and compares energy and vxc against the
independent PySCF 2.14.0 / Libxc 7.0.0 oracle. The tool always writes a complete
identity-bound receipt; numerical rows rejected by the versioned candidate
remain explicit failures rather than being silently clipped or skipped.

The two generic control rows are evaluated by the shared
`production_domain_controls` owner. `control/invalid-nonfinite` requires every
nonfinite rho/sigma/tau feature to be rejected before functional mathematics.
`control/lazy-inactive-branch` verifies value, first derivative, second
derivative, array interpretation, and C/CUDA lexical branch emission on a
singular inactive branch. These controls are compiler/runtime invariants rather
than functional numerical coordinates; the surrounding receipt still binds
their pass to the exact functional capability identity.

Running the campaign still does not imply that an imported registration is
production-qualified: unresolved numerical boundary rows remain explicit
failures. Use `--require-pass` only when the caller intends a fully qualified
matrix to be a hard gate.

### Catalog campaign

`tools/qualify_libxc_production_catalog.py` applies the same single-functional
campaign to a deterministic sorted inventory or shard. Use an artifact/scratch
directory rather than writing raw runs directly under `benchmarks/results/`:

```bash
python tools/qualify_libxc_production_catalog.py \
  --output .artifacts/libxc-domain/shard-0 \
  --evidence-prefix artifact://libxc-domain/shard-0 \
  --shard-count 4 --shard-index 0
```

Each eligible registration gets its own campaign JSON with exact receipt and
execution identities. Structurally unsupported registrations are summarized
without a fabricated receipt. `summary.json` reports pass/fail/not-run,
structural blockers, runner errors, family counts, and counts of blocked matrix
case IDs. Sharding is deterministic by sorted registration name, so shards are
disjoint and reconstruct the same selected inventory.

The catalog command is evidence collection, not publication or admission.
`--require-all-pass` turns any non-pass selected registration into a nonzero
command result; without it, negative results are retained for diagnosis and
later B2 aggregation.

The canonical `tests/data/xc/r2scan-tail-reference.json` fixture has
machine-readable status `pass` for the compiled CPU FP64 production entry point,
including its shared Libxc work-MGGA boundary wrapper. Its acceptance oracle uses
the original Libxc 7.0.0 Maple formulas evaluated in 113-bit arithmetic, because
the raw binary64 Libxc empty-spin derivatives are cancellation-sensitive. CI checks
the canonical zero/near-zero-minority tail points and their spin permutations.
`tests/data/xc/boundary/r2scan-zero-minority.json` remains a retained binary64
Libxc diagnostic, not the production acceptance target. Bare derivative roots omit
the wrapper and cannot establish the production entry point's status. This bounded
qualification does not promote the bulk inventory's production-domain or
public-method capabilities.

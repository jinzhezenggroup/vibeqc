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
the exact `vibeqc.libxc-production-domain-profile.v2` qualification payload
computed from the registration family and required ingredients. The current
`semilocal-boundary-matrix/v2` profile requires both spin layouts and
energy/vxc/fxc. Density, gradient, tau, and control cases apply to both layouts,
while alpha/beta zero-spin and full-polarization cases apply only to the polarized
layout. The exact cases-by-spin matrix is part of the profile identity, so dropping
or moving a case, changing the matrix version, or
changing an ingredient invalidates the admission proof. Registrations requiring
ingredients outside the current generic `rho/sigma/tau` domain (for example,
Laplacian-dependent meta-GGAs) carry an explicit structural blocker and do not
appear as `production-domain` ready.

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

The receipt additionally binds the exact order-2 execution programs used by the
campaign for both spin layouts: executor kind, runtime domain, imported source
identity, expression identity, optimization mode, feature ABI, and the complete
energy/vxc/fxc output contract. This binding is content-addressed and included in
the receipt hash. A matrix cannot be retained as production-domain evidence
without naming the exact mathematical execution that produced its candidate
values. Backend compilation/runtime qualification remains a separate stage; the
current B1 campaign explicitly records the shared array-Graph executor rather
than pretending that interpreted evidence is a compiled-CPU or CUDA result.

The B1 campaign uses the explicit
`libxc-bulk-production-candidate/v1` runtime domain. It differs from the default
`libxc-bulk-interior/v1` only by admitting physical zero sigma into candidate
evaluation. Negative sigma, non-positive density, non-positive tau, non-PSD
polarized sigma Gram matrices, and nonfinite inputs remain rejected. Admission of
zero sigma is **not** a generic pass: the imported order-2 Graph must still
produce finite E/vxc/fxc and match the independent Libxc oracle for that exact
functional. Functionals whose imported algebra has a true or unresolved
zero-gradient derivative singularity therefore remain blocked by their numerical
matrix row.

Ordinary bulk runtime consumers continue to default to the original interior
domain; the production-candidate domain is selected explicitly by the evidence
campaign and is part of its execution identity.

`vibeqc_compiler.xc.production_domain_cases` instantiates every numerical
rho/sigma/tau row in the exact v2 cases-by-spin matrix from finite physical
density, Cartesian-gradient, and kinetic-density coordinates.
`tools/qualify_libxc_production_domain.py` evaluates those rows through the
generic order-2 bulk candidate and compares energy, vxc, and packed fxc against
the independent PySCF 2.14.0 / Libxc 7.0.0 oracle. The tool always writes a
complete identity-bound receipt; zero-density/zero-gradient points rejected by
the current interior-only candidate remain explicit failures rather than being
silently clipped or skipped.

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

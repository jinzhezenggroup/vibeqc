# Decision: source-owned bulk Libxc Graph imports

Status: implemented
Date: 2026-09-21
Agent: ChatGPT
Model: GPT-6 Astra Pro

## Problem

The small production source closure and a static call-target scan do not establish
full-tree functional support. Real Graph construction also needs C-owned default
parameters, feature conventions and the exact upstream worker contract. Counting
parameter variants as separate mathematical source files is equally misleading.

## Decision

Use `upstream/libxc-fulltree/7.0.0` for this non-production inventory. Do not
place it at `upstream/libxc/7.0.0`: existing production adapters already probe
that reserved flattened layout and would stop falling back to their valid
legacy sources. The broad regression suite caught this ownership collision;
a dedicated test prevents it from returning.

Keep the complete pinned 7.0.0 Maple inventory and corresponding C parameter
owners as an offline, byte-preserved source projection. Reuse the existing
scientific source registry; exclude generated Libxc C kernels from the projection
and never compile these upstream C owners into the production runtime.

Extract only bounded, statically evidenced homogeneous-double parameter layouts
whose registration uses `set_ext_params_cpy`. Bind by actual C struct order, not
by the human-readable external parameter labels. Unknown/custom setters, mixed
layouts, unsupported constants and nonstandard workers fail closed. In
particular, SCAN-VV10 and SCAN-rVV10 registrations are not promoted just because
their semilocal equation can be evaluated: their worker differs from the admitted
ordinary semilocal contract.

Preserve the single Maple frontend. A collision-checked transient basename view
implements upstream's shared include lookup without rewriting any source bytes
or changing the common importer. Include order, duplicate/redefinition checks,
parameter identity and Graph lowering remain owned by that frontend. Logical
upstream paths and all source hashes are retained in the bulk catalog; temporary
absolute paths never enter identity.

The generated catalog records every discovered registration's outcome. Successful
entries must construct and symbolically differentiate physical-feature energy
Graphs in both spin layouts, rather than merely pass an AST traversal. The
compiler-facing builder supports density, sigma, laplacian and tau feature
coordinates through the existing Graph differentiation and C/CUDA emitters.

## Rejected alternatives

- Static parse success as an import count: it misses missing scalar bindings,
  branch-specific requirements and evaluator semantics.
- Inferring parameter fields from display names: external names and C struct
  fields/array layout differ in real upstream sources.
- Treating every custom setter as direct copy: transformations such as lsPBE's
  modified mu would silently change scientific mathematics.
- Synthetic default parameters or handwritten fallback equations: neither is a
  valid source import.
- Promoting new public Calculator/MethodIR methods on point tests alone: endpoint,
  force/response, device execution, domain and performance gates remain separate.

## Invariants

- No runtime Libxc, Maple, PySCF, Torch, NumPy or GPU is needed for source generation.
- Bootstrap accepts only the pinned archive checksum; ordinary regeneration,
  loading, source verification and wheel operation remain offline.
- Source/parameter-owner tampering is rejected before mathematical construction.
- Existing production math and public admission are unchanged.
- One source may have many parameterized registrations; both counts remain visible.

## Evidence

The projection contains 285 Maple files, of which 268 carry energy-family type
markers. Their C owners expose 544 registrations. The admitted subset has 221
registrations from 108 distinct mathematical source files; 323 registrations
retain explicit blocker reasons.

Independent fixtures use PySCF 2.14.0's Libxc 7.0.0 C library, calling the native
`xc_{lda,gga,mgga}_exc_vxc_fxc` APIs without any VibeQC equations or bindings in the
oracle. They cover both spin layouts and two ordinary physical points each
(884 points), including every physical first derivative and packed feature
Hessian. These are point-level interior checks, not full scientific admission.

Tests also cover malformed/ambiguous C metadata, custom setters, integer constant
semantics, real C compilation, both existing emitters, deterministic regeneration,
source tampering, installation-path-independent identity and an isolated
`python -S` packaged build with socket access disabled.

## Consequences and revisit conditions

Coverage expansion is driven by shared unsupported parameter/Maple constructs,
not by adding per-functional formula adapters. Extend the metadata subset only
with source evidence and independent numerical gates. Extend public scientific
admission separately under #744/#745. The original small production closure can
be consolidated once those cutovers complete; it is not silently redirected here.

References: #739, #743, #744, #745, #800; `docs/libxc_bulk_import.md`.

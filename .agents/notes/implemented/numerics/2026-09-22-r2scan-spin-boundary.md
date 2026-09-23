# Decision: r2scan spin boundary

Status: implemented
Date: 2026-09-22

The later [stable spin-fraction decision](2026-09-23-scan-stable-spin-fractions.md)
supersedes the rounded double reference assumption below: disabling FMA did
not remove sensitivity to AO-contraction roundoff at the complete endpoint.
It preserves the thresholds and derivative conventions recorded here.

## Decision

Use scalar domains and boundary continuations to keep generated r2SCAN value and first derivatives finite when a minority-spin density vanishes. CPU and CUDA emitters consume the same expression policy.

## Invariants and rejected alternatives

Do not mask a nonfinite result after evaluation or change the interior functional to make a boundary test pass. Preserve independent Libxc comparisons and zero-spin derivative conventions.

## Evidence and remaining qualification

CPU expression, Maple adapter and generated-C++ boundary tests pass. RTX 5090 FP64 boundary qualification and coordination with #1040 remain merge gates.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.

## Integration with the shared compiler owner

The #1092 merge moved scalar differentiation into `xc.semilocal_codegen`.
The boundary-aware root construction and component-wise r2SCAN emitter now
live there too; both tool wrappers forward to that owner. The emitted
equations, constants and ABI match the original candidate; source-provenance
hashes reflect the intervening compiler ownership changes.

The native KS test helper previously narrowed functional ID 2 to a boolean,
so the supposed r2SCAN tests ran PBE. Preserving the integer ID exposed a
CUDA/CPU domain mismatch: the CUDA r2SCAN point shortcut/continuation could
hide a negative spin density. Match the CPU finite/nonnegative rho/tau and
finite-gradient gate before either continuation. Zero minority spin remains
valid; an indefinite physical density must still fail and preserve the
last-good seed. This does not change the interior functional or relax a gate.

## CUDA contraction policy

An independent RTX 5090 FP64 probe exposed a second boundary failure that the
CPU tests could not detect. Default nvcc FMA contraction changed the empty-spin
density derivative by 0.00530399 at rho=(0.073, 0), despite agreeing in energy.
The same generated body with `--fmad=false` passes the unchanged Libxc 7.0
energy/first-derivative gate (5e-12 relative plus 1e-12 absolute), including
spin-exchanged points. Contracting the nearly cancelling `1-zeta` expression
changes its rounded boundary argument and amplifies the derivative error.

Apply the existing compiler XC FP64 no-contraction policy to the native
generated grid translation unit and use that policy in the allocated-GPU
regression. This includes AO/grid contractions in that translation unit;
complete native LDA/PBE/r2SCAN endpoint qualification remains required and
there is no speedup claim. Do not change only the probe flags while leaving
production contraction enabled. A future narrower or stable algebra policy
must independently pass the same boundary and full-endpoint gates.

## Completed NVIDIA endpoint qualification and provider spelling

The source-matched RTX 5090 Release library passes the native LDA/PBE/r2SCAN
RKS/UKS suite, including physical-state closure, restart, resource and invalid
input gates. The public DF suite passes 17 applicable cases. Seven selected
r2SCAN tests pass together: CPU/GPU Libxc boundary fixtures, independent RKS
and UKS analytic gradients, reconverged directional finite differences, and
public RKS/UKS forces. The probe retains the original 5e-12 relative plus
1e-12 absolute derivative gate; these results do not extend that raw gate
to additional exploratory near-vacuum points.

CuMetal's pinned nvcc-compatible driver delegates compilation to Clang and
rejects NVIDIA's `--fmad=false` spelling. Select `-ffp-contract=off` for that
provider and retain `--fmad=false` for NVIDIA. Both request the same policy;
Apple execution remains covered by its provider CI, not by the NVIDIA run.

## Importer identity and offline inventory

The density-threshold selection in shared Maple exchange lowering changes its
semantics even though the existing interior functional values remain the same.
Record this as `libxc-maple-graph/v11`, regenerate the bulk graph inventory and
coverage snapshot, and refresh both source-admission and product hashes. The
catalog retains all 221 imported and 323 blocked registrations; only graph-node
counts and the importer version change. Leaving v10 and its old digests would
make offline source verification and catalog reproduction fail.

All 527 selected source-registry, bulk independent Libxc E/vxc/fxc, importer and
CPU boundary cases pass (one allocated-GPU-only skip, already covered by the
separate Slurm qualification). Complete generated CPU and r2SCAN CUDA sources
remain identical to the qualified boundary implementation except provenance
hashes. No new bulk production-domain admission follows from this update.

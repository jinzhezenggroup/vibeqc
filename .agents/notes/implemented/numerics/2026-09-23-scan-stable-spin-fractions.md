# Decision: form SCAN spin fractions before differentiation

Status: implemented
Date: 2026-09-23

## Problem

Issue #1105 reproduces an empty-beta r²SCAN potential mismatch in the complete
native H2/STO-3G device-buffer test: GPU 2.410941123885864 versus CPU
2.4109061845583866, a 3.49393e-5 error against a 2e-11 + 2e-12*abs(reference)
gate. Slurm 11382 reproduced it without the matrix schedule in #1106. The
fixture uses 49,152 molecular-grid points and an alpha density matrix with
diagonal (0.21, 0.1575) and off-diagonal 0.035; beta is zero.

Identical point features agree between the old generated CPU/GPU routines.
However, changing only majority density by one ULP at a contributing grid
point changes minority v_rho from 38.3890340026 to 38.3184503034. Reassembling
the beta potential changes a matrix element by approximately 1.23e-5.
The cancelling `1-zeta` coordinate loses relative accuracy in the continued
minority work density. Ordinary AO contraction roundoff exposes this even
with FMA disabled.

## Decision and invariants

Within the compiler's SCAN-family correlation adapter, replace exact DAG nodes
`1+zeta` and `1-zeta` by `2*rho_a/density` and `2*rho_b/density` before AD.
The identities require nonnegative admitted spin densities and positive total
density; production work floors already establish this. Unpolarized lowering
stays unchanged. This is an algebraic conditioning repair, not a new functional
or boundary continuation. Density/sigma/tau/zeta thresholds, exchange screening,
raw-work first derivatives, precision, and admission gates remain unchanged.

`Graph.replace_subexpressions` performs simultaneous exact-node substitution;
replacement subtrees are inserted verbatim. It preserves node operations,
payloads and selected branches and checks graph ownership. The scientific
adapter owns equivalence and domain reasoning. Shared CPU/CUDA generation and
all subsequent derivatives consume the resulting graph. The adapter provenance
includes `direct-spin-fractions/v1`; source-admission hashes are refreshed.
The generic Maple importer and bulk catalog equations have not changed.

## Independent reference and rejected alternatives

The old binary64 Libxc reference suffers the same cancellation; matching those
rounded minority channels would retain the bug. The five original double
diagnostics remain in the new fixture, with the original 5e-12 relative plus
1e-12 absolute gate unchanged. Their acceptance reference is replaced only
after comparison to **independent original Libxc Maple 2022 generated C E/vxc**
evaluated in 113-bit arithmetic using GCC/libquadmath. The reference generator
does not import VibeQC Graph, differentiation, Maple parsing, or runtime.

The generator verifies the exact Libxc 7.0.0 archive hash, extracts two raw
polarized E/vxc routines without altering their algebra, widens floating
constants and elementary functions, and records the original source/probe
hashes. Its adapter keeps the FP64 work-driver input floors and clipping,
physical-density energy, and raw-work derivative convention. The fixture has
50 points: nine zero-spin points and their neighboring majority floats, plus
20 finite-minority points spanning 1e-16 to 1e-6, and the three additional
master H2 grid/adjacent-density points. The CUDA gate also exchanges
spins. Regeneration needs only the pinned archive and GCC/libquadmath.

A separate diagnostic widening the old VibeQC program agrees, but is not used
as the independent acceptance oracle because it shares our AD. The stable
FP64 candidate agrees with the independent Libxc wide evaluation on the first
nine probes within the unchanged gate (maximum normalized error 0.146).
One-ULP integrated sensitivity falls to roughly 1e-15.

Disabling FMA alone was insufficient at the full endpoint. Loosening matrix
or point tolerances, changing work floors, or silently keeping the old double
reference would hide the conditioning problem. No extra runtime precision,
CPU fallback, oracle call, allocation, or performance claim is introduced.

## Evidence

- Slurm 11393: complete native device-buffer LDA/PBE/r²SCAN RKS/UKS energy,
  potential, resource and state gates pass, including the original n=2 tail.
- Slurm 11394: generated CPU and GPU probes pass the 47-point independent
  reference and exchanged-spin GPU gate (two Python tests).
- 92 selected host compiler/provenance/boundary tests pass; source registry
  verification and compiler ownership checks pass.
- Compiler adapter tests retain independent SCAN/r²SCAN E/vxc/fxc interior
  and boundary fixtures; general Graph substitution tests protect simultaneous
  replacement, branch behavior, subsequent differentiation and graph ownership.
- Local diagnostic artifacts: `.artifacts/r2scan-tail/`; committed recipe and
  oracle data: `tools/generate_r2scan_tail_reference.py` and
  `tests/data/xc/r2scan-tail-{oracle.cpp.in,reference.json}`.

## Revisit when

Revisit if admitted feature domains or continuation policy change, or if a
future backend permits contraction. Keep full AO-to-potential endpoint gates:
same-feature point agreement alone cannot detect this failure. Full96 SCF and
README benchmark publication remain outside this repair's qualification.

This supersedes the rounded-boundary-reference assumption in
[the earlier r²SCAN boundary decision](2026-09-22-r2scan-spin-boundary.md).

## Integration with the subsequent master tail diagnostics

Master #1065 adds three independent Libxc FP64 grid/one-ULP fixtures and
a native same-device-feature diagnostic. Retain all three input points and
their original double values in the wide-reference fixture. The initial PR
merge CI appended their old rounded reference values to the new fixture and
failed those six minority channels; regenerate their acceptance values from
the same independent 113-bit original Libxc formulas. The unchanged host
boundary gate passes all 50 points.

The native same-input diagnostic is retained as an **additional** comparison.
The stable coordinates also restore the complete independently computed
CPU/GPU empty-spin potential comparison, for both exchanged spins. This
supersedes the matching-input-only interpretation in the master rounding note.

## Complete molecular qualification

Slurm 11402 passes both generated-backend tests against the final 50-point wide
reference. Slurm 11414 passes the full native LDA/PBE/r²SCAN RKS/UKS E/V and
state suite and all 12 selected public/complete CUDA r²SCAN cases (44 unrelated
cases deselected; 190.27 seconds). These include independent analytic RKS/UKS
gradients, reconverged directional finite differences, public forces, ordinary
SCF and DF energy endpoints. No scientific gate was relaxed.

An earlier local run had five force failures because only `vibeqc` and the native
test target had been built. Slurm 11413 called the force helper directly after
a successful energy solve and exposed `FileNotFoundError: missing packaged
stationary CUDA artifact for r2scan_rks`. Building the separate
`vibeqc_stationary_r2scan_{rks,uks}_manifest` targets resolves that local build
omission; no mathematical or runtime change was needed. Keep these AOT targets
in complete endpoint qualification, since the public status alone maps this
exception to the generic numerical-failure category.

Tested code: `d5d7de5f`. Native library SHA256:
`d482db17833398c27f2c98b970055fe4d71ada203510a76fdc50ab9b1527f365`.
RKS stationary artifact SHA256:
`b0f2b68b40c6c51927e7fc43e9724acf4e351b9a091e2ec017fa67342ec66234`;
UKS artifact:
`438334f70fc114981f7d5afd2012e7314de404a096b670e40fa849480576de46`.
The complete source-head CI also passes. This qualifies the conditioning repair,
without claiming 96-atom SCF performance or resuming README publication.

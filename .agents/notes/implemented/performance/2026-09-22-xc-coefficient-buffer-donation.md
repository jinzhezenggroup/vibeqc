# Decision: Donate packed PBE scalar rows to coefficient storage

Status: implemented
Date: 2026-09-22

## Problem

The compiler-owned fixed-density polarized PBE CPU tile already emits one
contiguous 8-by-N scalar-XC row matrix: energy plus the seven complete feature
gradient rows. The following native coefficient producer also needs an 8-by-N
FP64 output, but previously allocated a second owner, then unpacked that owner
into separate rho/gradient arrays before AO assembly. This left #831's explicit
ProgramIR donation contract unused by a real native provider.

## Decision

The packed PBE `vxc` call now declares
`CallDonationBinding("vxc", "xc_rows", "coefficients")`.
`PreparedXCContractions` enables the matching runtime ownership transfer only
when the ProgramIR buffers have equal capacity and the generated packed
coefficient ABI has exactly one output row beyond its feature-gradient input.

The native coefficient loop preloads every referenced feature-gradient scalar
for a point before writing any output row. It is therefore safe for the output
base to overlap the input at the exact one-row offset used by `xc_rows[1:]`.

The normal unpacked/native path and packed callers that do not pass the explicit
donation selection retain separate owners. LDA and other unequal-row layouts
fail closed rather than forcing reuse.

## Rejected alternatives

- Hidden shape-based donation inside every packed scalar call was rejected
  because ownership transfer must be an explicit execution-plan decision.
- Reinterpreting or resizing layouts merely to create a donation opportunity was
  rejected because physical layout follows the scientific/provider ABI.
- Keeping direct feature-row loads while overlapping input/output was rejected:
  an earlier output store can overwrite a feature row needed by a later root.

## Invariants

- The scalar-XC energy row is reduced before the owner is transferred.
- Donation requires exact contiguous FP64 layout, equal capacity, the one-row
  input offset, compiler ownership and non-overlap with density-gradient input.
- Borrowed coefficient views are read-only and must share memory with their
  validated owner.
- Unsupported/opaque layouts keep the non-donating fallback.

## Evidence

For the 12-AO, 7-point ProgramIR regression, the compiler-visible pageable peak
drops from 10,044 to 9,596 bytes, exactly 448 bytes (one 8x7 FP64 owner).
A 13-point generated-native PBE tile reuses the same 832-byte owner pointer;
energy, electron count and every potential-matrix element are bitwise identical
to the independent Graph/interpreted contraction path.


## Consequences

The production packed PBE tile no longer needs a distinct compact-coefficient
owner at the scalar-to-Vxc boundary, and packed native coefficient results feed
AO assembly as views rather than being copied into per-field arrays. This is a
bounded ownership/lifetime optimization; it changes neither functional
mathematics nor quadrature/SCF policy.

## Revisit when

Requalify the transfer if coefficient row order/count changes, if the provider
becomes asynchronous, or if a new layout makes the input/output overlap differ
from the validated one-row offset.

## References

- #831
- #900
- #943
- #973

Agent: ChatGPT
Model: GPT-5.6 Sol

## Review qualification on current compiler ownership

After integrating master `41834864`, 100 ProgramIR/native XC/empty-tile and
contraction tests pass. The audience-documentation rename preserves this
storage contract under `docs/developer/compiler_architecture.md`.

A complete `PreparedXCContractions.execute` CPU E/V ablation compares donation
on/off on the same source, including AO collocation, features, scalar XC,
coefficient production, assembly and result publication. Both retained fixture
cases use 32 grid points and five tiles (7/7/7/7/4), with five scalar calls,
five coefficient calls and 30 logical matrix products. Donation removes 2,048
bytes of new coefficient-output allocation over the call and reuses those exact
bytes. All energy, electron and potential outputs are bitwise equal between
arms and retain the independent fixture gates. This is a donation ablation;
both arms retain the new zero-copy coefficient views.

Python/NumPy tracemalloc peaks and complete-call times were also retained in
local review evidence. The larger Python endpoint peak is not reduced by this
small buffer transfer, and these tiny fixtures do not establish an endpoint
speedup or a process/native allocator peak reduction. The qualified claim is
the actual removal of a distinct coefficient owner with unchanged work, not a
complete SCF performance improvement. Resource admission retains its existing
conservative complete bound.

Instrumented ABBA call-time medians (two samples per arm, tracing enabled) were
28.05/28.69 ms without/with donation for the two-AO H2 fixture and
28.43/28.44 ms for the 16-AO spherical-f fixture. These are small fixed-density
E/V endpoint diagnostics, not converged-SCF or throughput qualification.

# Decision: derive Libxc native domain versions from runtime identity

Status: implemented
Date: 2026-09-25

## Problem

The generated `SemilocalPointProgram` carries both a string-valued compiler
runtime domain and an integer `domain_version` consumed by native RKS/UKS
diagnostics and prepared-state identity. The binding helper previously accepted
the integer as an arbitrary caller argument. A caller could therefore bind
`libxc-bulk-interior/v1` with one integer and the production-candidate domain
with the same or another unreviewed integer, creating contradictory execution
metadata before compiled-CPU evidence is retained.

## Decision

Make the compiler the single owner of this mapping.

- `libxc-bulk-interior/v1` -> native domain version 1.
- `libxc-bulk-production-candidate/v1` -> native domain version 2.

`bind_runtime_semilocal_point_program` no longer accepts a caller-supplied
version. It derives the integer from `program.spec.domain`.

Advance the point-binding identity to
`vibeqc.libxc-bulk-point-program-binding/v3`. Direct construction still permits
synthetic/custom domains for focused tests, but known compiler-owned domains
reject a mismatched integer version.

## Rejected alternatives

- Keeping caller-selected integers would make native diagnostics unauditable and
  weaken artifact/receipt identity.
- Hashing only the domain string while ignoring the native integer would allow a
  compiled binding to disagree with the state carried into the SCF runtime.
- Reusing version 1 for the production candidate would hide a deliberate domain
  expansion from prepared-state identity.

## Invariants

- Every compiler-owned Libxc runtime domain maps to one stable native integer.
- Known domain string and native integer cannot disagree.
- Changing a domain policy requires a new compiler domain identity and native
  version rather than silently mutating an existing mapping.
- The mapping itself grants no compiled, production, SCF, or public capability.

## Evidence

Tests lock the interior/candidate mapping to 1/2, require distinct binding
identities, reject unknown binding-helper domains, and reject forged integer
versions for known domains.

## Consequences

The next compiled-CPU evidence producer can bind one exact point program without
accepting a separate domain-version parameter. The retained compiled artifact,
production-domain receipt, and native SCF diagnostics can therefore refer to one
consistent execution domain.

## Revisit when

A third production domain is introduced or the native ABI replaces the integer
version with a content-addressed domain identifier.

## References

- #1119
- #1121
- #1276
- #1325
- #1329

Agent: ChatGPT
Model: GPT-5.6 Sol

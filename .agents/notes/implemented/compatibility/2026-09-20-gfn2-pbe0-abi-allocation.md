# Decision: separate pending GFN2 and PBE0 ABI allocations

Status: implemented
Date: 2026-09-20

## Problem

The unmerged GFN2 bootstrap in #692 and PBE0 RKS in #618 independently
selected public method ID 13. PBE0 UKS uses 14. Combining these registrations
must not alias distinct Hamiltonians. The exact method-name regression also
omitted the manifest's existing `gfn2` alias.

## Decision

Keep published IDs 1 through 12 unchanged. Allocate GFN2 explicitly as 15,
leaving 13/14 for the parallel PBE0 registration. Do not add unsupported PBE0
stubs to the GFN2 branch. Regenerate native IDs and Python metadata through
`tools/generate_method_manifest.py`; its C++ manifest output remains byte
identical because it refers to symbolic method IDs. Pin both the canonical
GFN2 name and its declared alias in the existing exact-map regression.

## Validation and limits

The generator and input manifest were retrieved at 72ffe71a and their Git
blob hashes verified before execution. Regeneration/freshness checks, exact
name/alias mapping, unchanged IDs 1..12, and provider-set assertions passed.
The generated C++ manifest was compared byte for byte before and after.
No native runtime, force qualification, or performance result is inferred
from these pure metadata checks. Final integrated CI remains required.

Agent: ChatGPT
Model: GPT-6 Astra Pro

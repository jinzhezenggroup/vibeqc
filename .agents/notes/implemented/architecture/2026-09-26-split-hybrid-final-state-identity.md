# Decision: preserve split-hybrid identity through native final-state export

Status: implemented
Date: 2026-09-26

## Problem

Native admission assigned the generated MGGA selectors the curated r2SCAN
family as an implementation detail. The CUDA final-state validator, however,
accepted only curated functional codes, so successful M06-2X and MN15 SCF
could not export a matched-grid state. The generic version-1 diagnostic also
reported the r2SCAN work domain rather than the split-hybrid work policy.
Separately, the public four-component KS diagnostic omitted exact exchange
from its XC energy despite reporting the hybrid's complete total energy.

## Decision

Give generated split hybrids a distinct native work-domain version (4), while
retaining the generated functional code as their immutable selector. Accept
only registered codes in the final-state validator, only on CUDA with direct
FP64 J/K, the registered spin-dependent exact-K coefficient, and no added
range or nonlocal terms. The CPU build has no dependency on CUDA-generated
registry headers. Snapshot and diagnostic readers bind version 4 to the
resolved public split-hybrid method; no CUDA derivative capability is granted.

CUDA split-hybrid snapshots use wire versions 8 (all-electron) and 9 (ECP),
which carry the native semilocal X/C scales and actual spin-dependent K term.
Older CUDA v3/v5 snapshots cannot claim this provenance; the derivative CUDA
consumer still rejects v8/v9 rather than treating energy acceptance as force
qualification.

The existing four-component C diagnostic continues to expose the *complete*
physical energy: its XC field includes semilocal XC plus full-range exact K.
Internal native components remain separate, so existing source accounting is
not discarded. Both final and per-iteration diagnostic totals retain the same
meaning. No C ABI layout or Python result shape changes.

## Rejected alternatives

- Relabel the generated code as curated r2SCAN in the export: this loses the
  mathematical functional and could validate a different model's state.
- Reuse native domain version 1 and trust the caller's method name: identical
  numeric provenance would hide the different work-density floor and policy.
- Expose a five-component public C struct without an ABI version change: this
  would break existing consumers and is unnecessary for total-energy reporting.

## Invariants

The CPU build must not require the generated CUDA registry. A changed K
coefficient, noncanonical composition, unknown code, or changed domain must
fail before releasing a validated state. Independent fixed-density Libxc and
PySCF J/K plus matched-grid SCF remain required for scientific acceptance.

## References

See `tests/python/test_split_hybrid_endpoints_cuda.py` and
`docs/maintainer/hybrid_cuda_acceptance.md` for the reproducible endpoint gate.

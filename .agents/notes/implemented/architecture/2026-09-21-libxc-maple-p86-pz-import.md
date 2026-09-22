# Decision: admit P86/PZ through the common Libxc Maple frontend

Status: implemented
Date: 2026-09-21

## Problem

Issue #743 must qualify `LDA_C_PZ` and `GGA_C_P86` from the pinned Libxc 7.0.0
Maple sources. Their mathematics is already expressible in the canonical Graph,
but the source uses Maple's explicit backslash line-continuation syntax, which
the v7 frontend previously rejected before expression parsing.

## Decision

Admit only a backslash followed by optional horizontal whitespace and a newline
as a lexical line continuation. Normalize that token pair before parsing and
bump importer identity to `libxc-maple-graph/v8`; an interior backslash remains
unsupported and fails closed.

Import PZ with its pinned parameter block enabled and pin `lda_c_pz.c` as the
matching parameter owner. Import P86 with the source-defined PZ include graph,
pin both C parameter owners plus `util.mpl`, and bind only P86's Libxc default
parameters and the 3D `RS_FACTOR` value owned by the pinned utility source.
Production construction remains unchanged for #744.

## Invariants

- No generic Maple runtime or family-specific symbolic engine is introduced.
- Graph AD owns energy, vxc, and packed fxc derivatives.
- Existing physical density/sigma/tau and spin conventions are unchanged.
- Scalar C and CUDA continue through the common emitters.
- Source, include, define, parameter and importer identities remain explicit.

## Evidence

The existing `tests/data/xc/p86-hessian.json` retained oracle was generated with
PySCF 2.14.0 / Libxc 7.0.0. `tests/python/test_libxc_maple_p86_pz.py` checks both
spin layouts against that independent E/vxc/fxc fixture, against the existing
audited handwritten DAG, and through Scalar C/CUDA lowering.

Polarized full E+vxc+packed-fxc growth baselines:

- `LDA_C_PZ`: 291 reachable nodes; 396 emitted lines / 14,415 characters.
- `GGA_C_P86`: 782 reachable nodes; 840 emitted lines / 30,222 characters.

## Rejected alternatives

- Do not strip arbitrary backslashes. Only source-evidenced end-of-line
  continuation is admitted; malformed interior backslashes remain errors.
- Do not duplicate the PZ formulas inside the P86 adapter; keep the pinned
  Libxc include relationship intact.
- Do not turn `RS_FACTOR` into a new frontend primitive for one family; bind
  the pinned 3D utility-source value at the adapter boundary.
- Do not combine qualification with the #744 production cutover.

## References

- #739
- #743
- #744

Agent: ChatGPT
Model: GPT-5.6 Sol

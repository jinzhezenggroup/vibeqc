# Decision: import LYP without extending the Maple frontend

Status: implemented
Date: 2026-09-21

## Problem

Issue #743 must migrate `GGA_C_LYP` from the audited handwritten expression to
the common pinned Libxc Maple frontend without widening frontend semantics or
creating another family-specific formula translation.

## Decision

Import the pinned Libxc 7.0.0 `gga_c_lyp.mpl` source with the four default LYP
parameters bound from the matching pinned `src/gga_c_lyp.c`. The adapter calls
the source-defined `f_lyp_rr` entry with VibeQC's canonical physical features,
so no `RS_FACTOR` frontend intrinsic is required. Existing `Pi`, `exp`, and
`opz_pow_n` primitives cover the complete selected expression.

Keep production construction on the audited handwritten source for now; #744
owns source-of-truth cutover after qualification.

## Invariants

- The Maple formula and C parameter source are both hash-pinned provenance.
- Graph AD owns energy, vxc, and packed fxc derivatives.
- Polarized/unpolarized density and sigma conventions remain unchanged.
- Scalar C and CUDA use the existing emitters with no family-specific backend.
- Unsupported Maple constructs continue to fail closed.

## Evidence

`tests/data/xc/lyp-hessian.json` is generated independently with PySCF 2.14.0 /
Libxc 7.0.0 and covers eight physical points for both spin layouts.
`tests/python/test_libxc_maple_lyp.py` checks imported-vs-handwritten parity,
the retained independent E/vxc/fxc fixture, source/parameter identity, and
Scalar C/CUDA lowering.

For the polarized E+vxc+packed-fxc graph, the qualified slice has 993 reachable
canonical DAG nodes. Scalar C and CUDA each emit 959 lines / 33,978 characters.
These measurements provide the #743 growth baseline for later families.

## Rejected alternatives

- Do not add a generic `RS_FACTOR` intrinsic just for LYP; `f_lyp_rr` is already
  a source-defined public helper with the required mathematics.
- Do not copy the default LYP constants into importer code; pin their Libxc C
  owner and pass them as explicit bindings.
- Do not combine qualification with the #744 production cutover.

## References

- #739
- #743
- #744

Agent: ChatGPT
Model: GPT-5.6 Sol

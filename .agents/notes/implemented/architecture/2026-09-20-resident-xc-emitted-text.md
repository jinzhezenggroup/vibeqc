# Decision: emit resident XC contraction text from the compiler

Status: implemented
Date: 2026-09-20

## Problem

The resident semilocal XC consumer kept its density-product, feature, point-layout,
potential and scalar-reduction bodies in a native runtime header. Moving their
source ownership must not be confused with deriving every contraction from a
new typed grid/XC IR or with removing all maintained scientific text.

## Decision

`vibeqc_compiler.dft.ao_cuda` emits the existing maintained CUDA text for these
resident-only contractions. This is an intermediate source-ownership boundary,
not a new mathematical IR lowering. Gaussian/feature scalar policies and the
shared point/directional expressions remain their existing scientific authorities.
The native header retains validation, stream use and allocation-free launches.

Only `emit_grid_source(native_ks=True)` appends the resident block, before its
native launch header. Ordinary JIT grid generation does not include it and keeps
its existing source/ABI. Native compilation supplies the existing generated
r2SCAN header; no public runtime, native library or device probe is imported by
the source generator.

## Rejected alternatives

Copying or re-deriving the XC formulas during this move would mix an ownership
change with an independent numerical change. Calling this fully IR-generated
science would obscure the remaining maintained text. Removing native validation
or adding resident definitions to the ordinary JIT translation unit would change
unrelated compatibility and lifetime boundaries.

## Invariants and evidence

Eight moved function definitions and the unchanged native enqueue body were
compared byte-for-byte against the merge-base source. Existing generator tests
check the native/JIT distinction and that retired definitions do not return to
the native header. Reduction order, spin packing, both potential legs, tau's
one-half factor, vacuum/tail admission and launch policy are unchanged.
This ownership move claims neither a performance gain nor new method capability.

## Consequences and revisit condition

The ownership ledger's runtime-only classification applies to the native header;
it does not assert that this emitter contains no maintained scientific text.
Replace this representation only when typed grid/XC contraction lowering can
preserve or independently qualify the same coefficient layout, numerical gates,
error publication, resource ownership and complete endpoint behavior. Keep the
native/JIT compatibility test during that transition.

References: #168; #713; `docs/xc_native_cuda.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro

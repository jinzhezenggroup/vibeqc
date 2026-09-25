# Decision: generated force roots do not retire native Gaussian geometry

Status: rejected
Date: 2026-09-25

## Rejected change

PR #1262 initially classified four low-order Direct force headers as runtime-only solely because they call compiler-generated Weighted IntegralIR force roots. Review found that every header still constructs operator-specific Gaussian geometry: rho, ERI prefactors, Boys arguments and decay derivatives. Some also recover centers and compose density-weighted force contributions.

The generated roots are genuine scientific compiler ownership, but the native inputs and composition are not generic packing or queues. In particular, `geometry.decay = -2*mu*(A-B)` is coordinate-response algebra, not movement of already-computed data. The ownership policy explicitly keeps operator-specific traversal/contraction glue in the migration surface even when scalar arithmetic is generated.

## Decision and invariants

Keep all four files conservatively scientific and in the low-order force retirement family. Move the force-only psss adapter out of the Fock family. Keep the generated-root regression and add per-file ownership/overlay guards. Remove the new runtime-only C++ comments by restoring the exact pre-PR source blobs; numerical execution is unchanged.

No physical scientific-code retirement or performance improvement is claimed. The existing independent force, resource and complete-endpoint gates are unchanged.

## Revisit when

Reclassify only after the remaining geometry and derivative composition actually have qualified generated/shared scientific owners, or after reviewed exact source regions separate genuinely generic runtime code. Do not infer retirement from a generated-call substring or a green source-contract test.

Agent: ChatGPT — Even-PR Review
Model: GPT-6 Astra Pro

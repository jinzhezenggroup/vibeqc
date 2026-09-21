# Decision: preserve preprocessing selection and lexical binding in Maple import

Status: implemented
Date: 2026-09-21

## Problem

The bounded PBE-C file importer introduced three observable semantic failures.

- A source can test a define before defining it itself. Imports with different
  initial symbols then select different expressions but finish with the same
  symbol set. Hashing only that final set aliases different imported graphs.
- Rewriting `log(1+x)` and `exp(x)-1` by spelling alone ignores parameters that
  shadow `log` or `exp`, including higher-order function arguments.
- Overwriting a scalar/function definition may change a previously assigned
  value because this importer resolves assignment expressions lazily against
  the final namespace, unlike Maple's assignment-time RHS evaluation.

## Decision

Record sorted initial defines separately from final active defines and include
both in the transitive identity. Source hashes and ordered include edges remain
part of the identity. Checkout paths and input iteration order do not affect it.

Resolve each call target before applying cancellation-safe intrinsic lowering.
Only actual log/exp intrinsics become log1p/expm1. Ordinary parameter binding,
non-callable errors, and higher-order calls retain their semantics.

Retain the pinned PBE file's independent literal/list overrides and its final
function replacement. Reject redefinitions reachable from earlier captured
assignment expressions, including references through other assignments or
functions. Function parameters are excluded from global dependency traversal.
This is intentionally conservative: supporting general assignment-time capture
is a separate language extension, not part of this bounded importer.

Advance the importer semantic identity from v2 to v3. No PBE equation, feature
derivative, scientific domain, production source selection, or numerical
acceptance threshold changes.

## Evidence

The new preprocessing/shadowing regressions expose eight failures before the
repair. The new capture regressions expose six more. After the repairs, all
178 selected Maple importer/parser/comment/independent-oracle/XC-expression
tests pass, including full polarized/unpolarized PBE-C typical and boundary
energy/gradient/Hessian contracts. No GPU or production-endpoint qualification
is inferred from these host/compiler tests.

## Rejected alternatives

Do not hash only final defines, ban all higher-order parameter binding, remove
stable elementary functions, or silently generalize Maple's mutable namespace.
Those approaches respectively retain identity collisions, regress supported
syntax, worsen PBE cancellation, or misrepresent unimplemented semantics.

## References

- PR #748, issue #741.
- Maplesoft assignment reference: https://www.maplesoft.com/support/help/Maple/view.aspx?path=assignment

Agent: ChatGPT
Model: GPT-6 Astra Pro

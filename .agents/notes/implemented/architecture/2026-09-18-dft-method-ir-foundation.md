# Decision: separate canonical DFT method composition from executable KS capability

Status: implemented
Date: 2026-09-18

## Problem

The scalar XC compiler already owns audited semilocal expressions and feature
derivatives, while native KS execution supports a deliberately narrower LDA/PBE
energy domain and rejects exact exchange.  Adding global hybrids directly through
more `FunctionalSpec` metadata or named runtime branches would mix three different
questions: what a method means, which primitives can represent it, and which
runtime/provider combinations are executable today.

Issue #396 requires a method-level composition layer above the scalar XC compiler
without weakening those existing capability boundaries.

## Decision

Add a representation-only `MethodSpec -> MethodIR` layer under the sibling
`vibeqc_compiler.method` front-end package.

- `MethodSpec` is declarative data: audited semilocal component coefficients plus
  an explicit exact-exchange coefficient.
- Resolution combines duplicate semilocal fragments with exact rational
  arithmetic, removes exact cancellations, and emits at most one canonical
  semilocal primitive plus one full-range exact-exchange primitive.
- The semilocal node is a normal `FunctionalSpec` backed by the existing #161 XC
  compiler; exact exchange is a separate structural primitive rather than hidden
  in `FunctionalSpec.exact_exchange` metadata.
- Named `LDA_XC_PW`, `PBE`, and `PBE0` manifests are data entries.  The first
  hybrid representation therefore needs no PBE0-specific scientific driver.
- `MethodIR.identity` is semantic and excludes the descriptive method alias;
  `manifest_identity` includes the alias so provenance remains inspectable.
- Primitive derivative metadata describes what the primitive representation can
  request from lower layers; it is not a public method-capability promise.

Representability is intentionally weaker than executable capability.  In
particular, resolving PBE0 does not change the current native KS restriction that
rejects exact exchange.

## Rejected alternatives

- Extend `FunctionalSpec` so its existing exchange metadata becomes the method
  abstraction.  This keeps exact exchange as a magic side field and cannot scale
  cleanly to range-separated exchange or external corrections.
- Add `if method == "PBE0"` branches to the native KS driver before a common
  composition contract exists.  That would immediately create the duplication
  #396 is intended to prevent.
- Include the descriptive method name in the semantic cache identity.  Two
  aliases with identical primitives would then fragment generated/tuned artifacts
  despite identical mathematics.
- Claim gradient/Hessian capability from representation alone.  Semilocal feature
  derivatives and exchange providers are necessary ingredients but do not by
  themselves establish a complete stationary molecular derivative.

## Invariants

- Same semantic primitive graph, spin contract, expression version and source
  provenance produce the same `MethodIR.identity` independent of alias or input
  component order.
- Manifest provenance remains separately hashable and includes the requested
  identifier.
- Exact exchange is never silently folded into a semilocal XC primitive.
- Unknown method names, unsupported XC components, non-exact coefficients and
  unsupported spin domains fail before runtime allocation.
- Provider selection, SCF policy and public energy/gradient support remain outside
  the compiler-side MethodIR.
- The `method` owner may depend on `xc` and `common`; lower-level `dft` remains
  independent of method/XC policy.

## Evidence

`tests/python/test_dft_method_ir.py` covers PBE/PBE0 graph structure, exact
rational canonicalization, semantic-versus-manifest identity, data-only same-family
extension, explicit exchange separation, JSON-serializable manifests, and
fail-closed invalid/cancelled compositions.

The change adds no generated CUDA, no native runtime path and no performance
claim.  Existing native KS semantics are intentionally unchanged.

## Consequences

DFT method semantics now have a stable place to grow toward meta-GGA,
range-separated exchange and external corrections without naming each method in
the scientific driver.  The short-term cost is a second, explicit capability
boundary: callers must distinguish a represented MethodIR from an executable
runtime method.

## Revisit when

Extend the primitive registry when #164/#165/#167/#172 need tau-dependent XC,
range-separated exchange or corrections.  Revisit semantic identity only if a
provider parameter is shown to change mathematical semantics and therefore must
move from runtime identity into MethodIR.

## References

- Issue #396
- Issue #161
- Issue #163
- `python/vibeqc_compiler/method/spec.py`
- `python/vibeqc_compiler/xc/spec.py`
- `python/vibeqc/ks.py`

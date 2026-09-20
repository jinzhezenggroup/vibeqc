# Decision: keep the Array API frontend above TensorIR

Status: implemented
Date: 2026-09-20

## Problem

TensorIR is the compiler's typed mathematical representation, but direct
construction of every add/einsum/broadcast/reduce node makes ordinary
tensor-level scientific glue more verbose than the mathematics. VibeQC also
needs a future interoperability boundary that can resemble the wider Python
array ecosystem without losing AO/occupied/virtual/auxiliary identity,
representation, symmetry, exact coefficients, or generated AD.

The package ownership checker requires every top-level compiler subsystem to
have an explicit dependency direction. A frontend cannot be added as an
unclassified directory.

## Decision

Introduce `vibeqc_compiler.array_api` as a distinct compiler owner immediately
above `vibeqc_compiler.tensor`.

The owner may depend on itself and TensorIR only. It exposes a bounded symbolic
`VibeArray`/namespace surface and lowers captured expressions directly to the
existing TensorIR primitives and `Program`. There is no parallel array IR and
no frontend-only runtime node.

The first subset is intentionally smaller than the Array API standard and is
only Array-API-shaped. It does not implement `__array_namespace__` or claim a
version: the standard treats that method as a compliance-discovery signal and
requires the returned namespace to expose the standard top-level API. The
protocol is deferred until conformance is actually tested. Unsupported implicit
broadcasting, dtype promotion, dynamic shapes, Python control flow, and
approximate float scalar spelling fail explicitly. General einsum remains a
VibeQC extension.

Higher-level physics stays with its existing owners: MethodIR, IntegralIR,
ProgramIR, stationary/implicit solves, providers, SCF policy and eigensolvers
are not generic array operations. External PyTorch/JAX/CuPy integration and
DLPack are follow-on interoperability work, not frontend dependencies.

## Rejected alternatives

- **Replace TensorIR with an Array API graph.** This would discard or weaken
  scientific index spaces, symmetry/packing, exact rational factors, stable
  equation identity and existing generated JVP/VJP semantics.
- **Place the frontend inside the TensorIR owner.** That would obscure the
  one-way dependency and make a convenience syntax look like part of the
  mathematical IR contract.
- **Trace arbitrary Python bytecode/control flow.** The initial need is pure
  tensor expression capture, not a graph-break framework or a tape through
  SCF/CC solver iterations.
- **Adopt framework arrays as the internal IR.** A PyTorch/JAX/CuPy dependency
  would couple scientific identity and compiler availability to one external
  runtime.
- **Permit NumPy-like implicit coercions immediately.** Equal shapes do not make
  AO and orbital populations interchangeable, and approximate float spelling
  must not bypass TensorIR's exact-coefficient contract.
- **Expose `__array_namespace__` for the preview subset.** Array-consuming
  libraries use that attribute to detect compliant arrays, so returning an
  intentionally incomplete namespace would create a false interoperability
  claim.

## Invariants

- Lowered outputs are ordinary TensorIR `Node`/`Program` objects.
- TensorIR never imports `array_api`; the dependency is one-way.
- No array convenience may make an illegal TensorIR equation legal.
- Index-space domains, representation, symmetry, role/differentiability and
  exact coefficients are preserved or fail closed.
- Existing TensorIR optimization, serialization, JVP/VJP and CUDA lowering are
  reused rather than reimplemented by the frontend.
- Capture/trace is preparation work; prepared native execution does not call
  Python once per TensorIR node.
- Generic compiler code does not gain SCF/method policy or mandatory external
  framework dependencies through this frontend.
- `__array_namespace__` stays absent until a declared standard version passes
  a conformance matrix; explicit namespace import is the preview entry point.

## Evidence

PR #636 includes:
- structural/logical-hash equivalence between captured and hand-built TensorIR;
- execution parity on a captured elementwise/reduction program;
- reuse of existing TensorIR JVP generation without frontend AD rules;
- rejection of equal-size AO versus occupied domains;
- rejection of approximate float scalar spelling and Python control flow;
- an SCF density expression that canonicalizes to the same TensorIR identity.

The repository ownership hook enforces `array_api -> tensor` and rejects
undeclared compiler-owner edges. The normal compiler-wide type check remains
authoritative.

## Consequences

Common tensor formulas can gain a concise Python-array frontend while the
compiler keeps one mathematical IR and one AD/code-generation stack. The cost
is an additional frontend package and a capability table that must stay
deliberately narrower than any conformance claim.

New frontend operations should preferentially lower to existing TensorIR
primitives. A new TensorIR primitive still requires its own mathematical,
serialization, AD and lowering justification.

## Revisit when

Revisit the boundary if:
- a required standardized array operation cannot preserve VibeQC scientific
  domains without a materially new IR concept;
- cross-framework zero-copy execution requires a runtime owner rather than a
  pure compiler frontend;
- conformance testing supports advertising a specific Array API version; or
- real consumers show the frontend adds complexity without reducing manual
  scientific glue.

## References

- #633 — Array API frontend/interoperability tracker.
- #625 — core SCF algebra expressed in TensorIR.
- #145 / #151 — TensorIR and generated JVP/VJP foundations.
- #460 — separate ProgramIR ownership for cross-subsystem execution/lifetimes.
- PR #636 — first bounded frontend implementation.

---
Agent: ChatGPT
Model: GPT-5.6 Sol

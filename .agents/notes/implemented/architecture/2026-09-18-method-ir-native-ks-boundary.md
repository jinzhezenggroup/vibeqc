# Decision: resolve native semilocal KS through MethodIR

Status: implemented
Date: 2026-09-18

## Problem

The first #396 slice (#440) introduced canonical `MethodSpec -> MethodIR`
composition, but the existing LDA/PBE native KS orchestration still reconstructed
scientific semantics independently from method-name prefixes and direct
`FunctionalSpec` lookups. That left two sources of truth before the native
runtime boundary and made a future exact-exchange primitive easy to bypass with
another method-specific branch.

A direct substitution of the canonical MethodIR semilocal `FunctionalSpec` is
also unsafe for compatibility: MethodIR deliberately canonicalizes component
order, while the established XC catalog retains audited declaration order in its
`FunctionalSpec.identity`.

## Decision

Resolve each currently executable native LDA/PBE RKS/UKS method through the common
#396 `MethodIR` frontend before resource planning or native allocation.

- A small declarative native-method table supplies only the public method alias and
  spin/reference selection.
- `resolve_method` supplies the scientific primitive graph.
- Native semilocal KS accepts exactly one `SemilocalXCPrimitive`; any graph with
  exact exchange or another primitive fails before native allocation.
- `KsOptions` retains the resolved MethodIR and records its manifest plus semantic
  identity in the model payload.
- Snapshot and stationary-gradient semantics reuse the same resolver instead of
  independently dispatching LDA/PBE functionals by name.
- The current C/C++ ABI remains unchanged in this slice; existing LDA/PBE flags
  still select the already-validated native numerical path.

For compatibility, the runtime uses the established audited XC catalog
`FunctionalSpec` identity only after comparing the MethodIR node's semantic payload
with an independently canonicalized catalog primitive for the requested native
selector. This checks the original coefficients, component set, spin, and
provenance before restoring declaration order. Graph identifiers remain
descriptive. This keeps historical snapshot/cache identity stable while making
MethodIR the semantic authority.

## Rejected alternatives

- Pass the canonical MethodIR `FunctionalSpec` directly into the current runtime.
  Its sorted component order changes the existing `FunctionalSpec.identity`, which
  would invalidate stationary state/gradient bindings and fragment caches despite
  unchanged mathematics.
- Keep the old `method.startswith("pbe")` / `method.endswith("uks")` scientific
  dispatch. That would leave #396 representation disconnected from production
  orchestration.
- Add PBE0 execution in the same change. Exact-exchange provider selection and
  coefficient accounting are a separate capability slice and remain fail-closed.

## Invariants

- Existing LDA/PBE RKS/UKS numerical execution and native ABI are unchanged.
- The runtime cannot gain hybrid capability merely because a hybrid is
  representable in MethodIR.
- MethodIR is the source of scientific composition and required ingredients before
  the native KS boundary.
- Existing named `FunctionalSpec.identity` values remain stable for the current
  LDA/PBE runtime and stationary handoff.
- Any disagreement between the MethodIR semilocal node and the audited runtime XC
  catalog fails explicitly.

## Evidence

With a fresh CPU-only Release build from the same source tree:

- `tests/python/test_dft_method_ir.py`,
  `tests/python/test_ks_options.py`, and
  `tests/python/test_dft_stationary_gradient.py`: 72 passed, 8 skipped.
- The new negative gate substitutes a PBE0 MethodIR graph and verifies that native
  KS rejects it before execution.
- Review regression tests reject changed coefficients, missing/extra components,
  the wrong family, and the wrong spin through option resolution, Calculator
  construction, and resource planning before native loading. All four native
  mappings retain their catalog identity, including descriptively renamed graphs.
- Ruff and whitespace checks pass on the touched Python files.

## Consequences

The Python KS orchestration now consumes #396 MethodIR without changing the native
scientific kernels. The next #396 slice can introduce
`ExactExchangePrimitive -> existing J/K provider` at a common provider boundary
rather than adding a PBE0-specific Python driver.

The native C++ LDA/PBE selector remains an implementation detail to migrate only
when a typed method-plan ABI/runtime boundary is justified; this slice does not
pretend that the full native driver is already generic.

## Revisit when

Revisit the compatibility projection if XC catalog identity becomes canonical-order
independent, or when a typed method plan replaces the current LDA/PBE native
descriptor. Extend the accepted primitive set only together with the corresponding
validated provider/runtime capability.

## References

- #396
- #440
- #161
- #163
- #165
- `python/vibeqc/ks.py`
- `python/vibeqc/_ks_snapshot.py`
- `python/vibeqc/_dft_gradient.py`

# Decision: Keep provider registration execution-only

Status: implemented
Date: 2026-09-18

## Problem

Backend availability, public capability metadata, no-CUDA diagnostics and prepared
provider dispatch had separate declarations. A compiled-out provider could therefore
remain visible in a capability table, while repeated stubs encoded availability in
ad-hoc strings. Collapsing those paths into a universal numerical provider would
remove duplication but would also risk treating Direct, DF, XC, ECP or correlated
methods as interchangeable fallbacks.

## Decision

`runtime/provider_registry.hpp` owns only execution metadata: provider identity and
version, backend, build availability, prepared/resource requirements, fallback
classification, provenance and deterministic unavailable diagnostics. It wraps a
typed domain that remains owned by the scientific subsystem; runtime code does not
interpret that domain.

SCF Fock registers CPU/CUDA x exact/DF providers. `resolve_fock_build` uses the typed
Fock domain to validate the mathematical request without rewriting its selected
backend or approximation. `PreparedFockPlan` separately requires the selected
provider to be executable before source allocation. Thus a CPU-only build can still
represent a CUDA request, but preparation rejects it instead of substituting CPU.
Capability queries expose scientific support only for an executable registration.

The method registry uses the same execution metadata, and constructs public
`Capabilities::available`, `supported_properties`, and `supports_batch` from the
registered validation/prepare/batch entry points rather than independent booleans.
Reserved methods therefore cannot advertise executable properties without a
prepare function. ABI-stable CUDA stub entry points remain explicit; compatible
stubs share the common not-built diagnostic/status helper.

## Rejected alternatives

A single universal provider selector for Direct, DF, XC, ECP and post-HF methods was
rejected because those choices change the Hamiltonian, approximation or response
semantics. Automatic CPU fallback for a missing CUDA provider was also rejected:
backend substitution must be an explicit method policy, never a registry side
effect. Removing explicit stub translation units was rejected where their symbols
are part of the existing ABI/build contract.

## Invariants

- Provider registration never changes the requested scientific method, operator,
  approximation, spin/reference convention or derivative order.
- Not-built and reserved providers fail before numerical execution with their own
  identity in the diagnostic.
- Public capability metadata cannot claim executable properties without a
  registered executable prepare entry point.
- Prepared-state reuse continues to depend on exact geometry/basis/auxiliary,
  semantic strategy, device/resource controls and concrete provider ownership;
  registry identity does not replace those cache checks.
- Performance fallbacks remain explicit and subsystem-owned.

## Evidence

`vibeqc_fock_build_tests` covers executable registration metadata, unsupported
second derivatives, CPU-only CUDA not-built diagnostics and absence of silent
backend substitution. `vibeqc_fock_provider_tests` retains stale geometry, basis,
auxiliary-basis and metric-cutoff rejection. `vibeqc_native_tests` verifies reserved
method capability metadata has no executable properties. CPU-only ECP and direct
J/K stubs retain explicit entry points while sharing the not-built helper.

## Consequences

Adding a provider now requires one execution registration plus its subsystem-owned
typed capability domain. The small common descriptor is intentionally less
ambitious than a numerical provider interface; some explicit backend adapters and
stubs remain because they carry distinct ownership or ABI contracts.

## Revisit when

Introduce additional shared fields only when at least two scientific subsystems
need the same execution metadata without interpreting each other's method
semantics. Revisit generated symbol stubs when their signatures can be generated
without obscuring ABI ownership or diagnostics.

## References

- Issue #352.
- [Current Fock build contract](../../../../docs/developer/fock_build.md).

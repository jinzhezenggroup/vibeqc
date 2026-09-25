# Decision: isolate bulk point-program translation units

Status: implemented
Date: 2026-09-25

## Problem

The #1276 adapter emitted a global `bulk_xc_point` and external inline
`evaluate_point`/`kPointProgram` symbols shared by all registrations. Linking
two otherwise valid artifacts failed with duplicate definitions; changing only
the scalar function would still allow different descriptors to coalesce.

## Decision and invariants

Include system/native headers globally and put the unmodified scalar source,
evaluator, identity constants and descriptor in a translation-unit-local
namespace. Each consumer can still address `bulk_generated::kPointProgram`
within its own translation unit and expose it through its own registry entry.
The standalone scalar AOT source and emission identity are unchanged. The
point-binding schema is v2 so old wrapper artifacts cannot alias the repair.
No point formula, feature pullback, domain policy or scientific admission changes.

## Evidence and rejected alternatives

Two independently emitted synthetic registrations are compiled and linked at
-O0 and -O2 against the production point ABI declarations. The executable checks
distinct descriptors, identities, energies and density derivatives. Both link
steps fail before the repair and pass afterward. This is a host linkage test,
not new Libxc or molecular numerical qualification.

Keeping the shared external inline names or making only `bulk_xc_point` static
is insufficient: all registration-dependent implementation symbols must be
isolated. Revisit when the native registry supports explicitly named per-binding
symbols or multiple bindings within one generated translation unit.

Extends `2026-09-25-libxc-bulk-point-binding.md`. Refs #1276, #1119, #1121.

Agent: ChatGPT — Even-PR Review
Model: GPT-6 Astra Pro

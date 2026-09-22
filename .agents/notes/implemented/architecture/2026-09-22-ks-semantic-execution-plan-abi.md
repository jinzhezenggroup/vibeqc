# Decision: lower MethodIR through one semantic KS execution-plan ABI

Status: implemented pre-release breaking change
Date: 2026-09-22

## Context

The compiler already owns the complete KS scientific graph: semilocal XC components, full- or range-separated exact exchange, optional nonlocal correlation, spin, numerical domain, and grid policy. The previous native ABI re-encoded that graph as an append-only v1-v6 vibeqc_ks_options structure. Each new contribution therefore required another suffix and method-family code, even though MethodIR had already resolved the physics.

VibeQC has not been released, so preserving those intermediate layouts would freeze unnecessary compatibility debt.

## Decision

Replace the v1-v6 suffix chain with one current semantic layout. Python lowers the compiler execution plan directly into arrays of semilocal components, exact-exchange contributions, optional nonlocal correlation, and the exact numerical-domain/grid identity.

native_ks_options() performs only structural lowering. It has no WB97M-V/PBE/B3LYP/r2SCAN family selector and no ABI-version branch. Native preparation validates the semantic graph and maps only qualified graphs onto available lowerers; unsupported graphs fail closed.

The ABI intentionally accepts no old nested KS prefixes. The public query now returns schema 1 for this semantic layout. The top-level method descriptor's independent short-allocation compatibility remains unchanged.

## Consequences

Adding another composition built from already-supported primitive kinds no longer requires vibeqc_ks_options v7/v8 suffixes. New native lowerers may still require an admission rule, but transport itself remains generic. WB97M-V uses the same transport path as PBE0 and generic RSH compositions: MethodIR -> KS execution plan -> semantic C ABI -> native lowerer admission.

Refs #935 #491 #167 #396

Agent: ChatGPT
Model: GPT-5.6 Sol

# Generate B3LYP CUDA point algebra from MethodIR

Status: implemented
Date: 2026-09-25

## Decision

The native CUDA B3LYP semilocal point consumer is generated from the same canonical
polarized MethodIR graph and production-domain policy as the CPU implementation.
CUDA does not own a second handwritten B88/LYP implementation.

The generated device ABI is a GGA rho/sigma value-and-first-derivative consumer.
Full-range exact exchange remains a separate prepared Fock-provider primitive and
is composed by the shared CUDA KS path. The B3LYP family therefore carries four
grid features per spin and never inherits the meta-GGA tau layout.

## Admission

Only canonical versioned B3LYP with unit semilocal-family scaling, 20% full-range
exact exchange, direct J/K, device-fused XC and strict FP64 is promoted by this
slice. Host-unfused CUDA B3LYP, density-fitted hybrid execution, mixed precision,
range separation and nonlocal correlation remain fail-closed.

## Invariants

- CPU and CUDA B3LYP semilocal algebra share one compiler graph.
- The existing B3LYP production-tail continuation remains authoritative.
- Exact exchange is counted once by the common Fock provider.
- Family code 3 is GGA-shaped: rho plus three Cartesian-gradient features.
- Unsupported execution combinations fail before scientific work.

## Evidence

Native CUDA XC tests compare B3LYP RKS/UKS energy and potential against the CPU
implementation. Public native endpoint coverage compares complete converged
CPU_REFERENCE and CUDA B3LYP calculations for closed- and open-shell cases.

No source-matched GPU execution result is claimed by this note until CI or an
allocated GPU run records one.

Refs #1187.

Agent: ChatGPT
Model: GPT-5.6 Sol

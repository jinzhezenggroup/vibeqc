# Decision: Select CUDA schedules from explicit provider capabilities

Status: implemented
Date: 2026-09-20

## Problem

The one-electron CUDA value policy inferred whether the active execution provider
was CuMetal from the presence of `CUMETAL_ROOT`. That environment variable
describes developer/toolchain setup, not the provider embedded in the configured
VibeQC execution image. A configured but inactive CuMetal tree could therefore
change NVIDIA production scheduling, while an active compatibility provider had
no explicit capability identity.

## Decision

VibeQC now configures a CUDA execution-provider identity explicitly. The normal
CUDA build advertises `nvidia`; the CuMetal CI execution image advertises
`cumetal`. A runtime-owned capability record states whether that provider
supports the templated shell-warp one-electron value kernel.

Production auto-selection consumes only this capability record. NVIDIA keeps the
qualified shell-warp default. CuMetal keeps the pair-thread fallback until its
execution image advertises shell-warp support.

`VIBEQC_ONE_ELECTRON_VALUE_MAPPING` remains a diagnostic override. An explicit
shell-warp request on a provider that does not advertise that capability does not
force an unsupported launch: it resolves to pair-thread and records a visible
capability fallback. Prepared Fock execution variants retain both provider
identity and override/fallback provenance, so execution identity changes when
the effective provider policy changes.

## Rejected alternatives

- Continue using `CUMETAL_ROOT`: build/development setup is not runtime identity.
- Infer CuMetal from Apple/NVIDIA platform facts: platform is not provider
  capability and would make future compatibility providers ambiguous.
- Let a diagnostic override bypass missing capabilities: this can request a
  kernel the active provider did not register and violates safe fallback policy.

## Invariants

- Production schedule selection must not inspect `CUMETAL_ROOT`.
- Missing provider capability selects a documented executable fallback.
- Provider identity and effective policy are provenance-visible.
- Diagnostic overrides never authorize unsupported provider capabilities.
- Scientific one-electron formulas and FP64 numerical semantics are unchanged.

## Evidence

- Native policy tests cover configured-but-inactive CuMetal, CuMetal capability
  absence, unsupported shell-warp override fallback, and NVIDIA thread override.
- Python structural tests require explicit CMake/CuMetal-provider wiring and
  reject reintroduction of the environment heuristic.
- Existing Fock execution-identity tests cover frozen prepared-plan provenance.
- CUDA and CuMetal CI remain the device/compiler acceptance gates.

## Consequences

Adding a new CUDA compatibility provider requires an explicit capability record
and build/provider identity rather than an environment-name convention.

## Revisit when

Revisit the CuMetal fallback when its VibeQC execution image registers and
qualifies the templated shell-warp one-electron kernel. At that point only the
advertised capability should change; production policy should not acquire a new
provider-name special case.

## References

- Issue #595
- `src/runtime/cuda_provider.hpp`
- `src/scf/cuda/rhf_policy.cpp`
- `.github/workflows/cumetal-cuda.yml`

Agent: ChatGPT
Model: GPT-5.6 Sol

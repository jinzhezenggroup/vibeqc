# Decision: Keep CUDA target selection separate from measured tuning profiles

Status: implemented
Date: 2026-09-21

## Problem

The compiler already represented CUDA targets explicitly and provided portable
fallbacks, but several target-independent surfaces still treated the development
RTX 5090 / `sm_120` environment as the implicit meaning of CUDA. In particular,
CMake supplied architecture 120 when none was requested, benchmark scheduling
requested `gpu:5090:1`, and the DF value candidate manifest admitted only an
`sm_120` payload.

That coupling was unsafe for fat binaries and new devices: an architecture-specific
candidate mapping could be compiled into a target for which it had never been
measured.

## Decision

CUDA architecture selection is caller/toolchain policy. VibeQC no longer inserts
architecture 120 merely because CUDA is enabled. Explicit project/CMake
architecture settings remain supported, and the existing `sm_120` presets remain
explicit development/release choices.

Shared scheduler defaults request one generic GPU (`gpu:1`); clusters that need a
model selector provide it through the existing environment or call arguments.

DF value candidate data is architecture-keyed. Device code selects an exact
profile using `__CUDA_ARCH__`; targets without an entry use the exact generic
value evaluator. Host-side candidate lane scheduling uses an architecture profile
only for a single-architecture build whose CMake target is explicit. Multi-
architecture builds keep the host candidate schedule scalar because their active
device is a runtime property.

## Rejected alternatives

- Keeping `sm_120` as the generic fallback was rejected because measured
  scheduling is not portable evidence.
- Treating all Blackwell/Ampere-family targets as compatible without measurements
  was rejected for the same reason.
- Using `__CUDA_ARCH__` to select host launch policy was rejected because it is a
  device-compilation macro and does not identify the active device in shared host
  code of a fat binary.

## Invariants

- Architecture-specific tuning must never be borrowed by an unlisted target.
- Unknown/future normalized CUDA targets retain exact generic behavior.
- Historical benchmark artifacts keep their recorded RTX 5090 / `sm_120`
  provenance.
- A measured profile may be added for another target without changing the
  compiler's scientific algebra.
- Multi-architecture host scheduling stays portable unless a runtime device-aware
  selector is introduced and independently validated.

## Evidence

The compiler target catalog already covers `sm_80`, `sm_86`, `sm_89`,
`sm_90`, and `sm_120`, with a conservative target record for unknown
architectures. New tests exercise an explicit non-`sm_120` value manifest,
a future `sm_130` fallback, architecture-gated generated source, and the
model-agnostic scheduler default.

## Consequences

Single-target `sm_120` builds retain their recorded diagnostic candidate lane
schedule. Fat binaries and unprofiled devices prioritize correctness and
portability over borrowing that schedule.

## Revisit when

Add compatibility between architectures only after complete endpoint evidence
shows a measured profile is valid across those targets. A future runtime
device-aware host scheduler may restore per-device candidate lane tuning inside a
fat binary.

## References

- #136
- #444
- #487
- #784

Agent: ChatGPT
Model: GPT-5.6 Sol

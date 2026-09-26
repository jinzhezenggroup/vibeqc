# Decision: compiled-CPU Libxc evidence binds and executes the native point program

Status: implemented
Date: 2026-09-25

## Problem

The automatic Libxc capability registry has a `compiled-cpu` stage, and #1276
already emits a native `SemilocalPointProgram` translation unit, but there was
no retained producer proving that one exact binding actually compiles and
executes. Source emission alone cannot establish compilation, and compiler
success alone cannot prove that the linked native ABI still represents the
intended expression/domain identity.

## Decision

Define `vibeqc.libxc-compiled-cpu-result/v1` and
`vibeqc.libxc-compiled-cpu-qualification/v1`.

The producer builds the polarized first-order
`libxc-bulk-production-candidate/v1` program and its v3 point binding. It then:

1. records the exact compiler executable hash/version;
2. emits one self-contained C++20 translation unit;
3. compiles and links it against the repository native headers;
4. executes the resulting binary with a deterministic physical rho/gradient/tau
   point;
5. requires the binary to report the exact binding identity and native domain
   version; and
6. compares every native `SemilocalPointValue` coefficient with the expected
   value reconstructed from the bound Graph.

The receipt content-addresses compiler, translation unit, executable, binding,
smoke input, and expected/observed outputs. Compilation/runtime failures are
retained as fail/not-run evidence.

## Rejected alternatives

- Treating emitted C/C++ source as `compiled-cpu` would confuse representation
  with compilation.
- Accepting object/executable creation without running it would miss ABI and
  domain-version mismatches.
- Using PySCF/Libxc inside this stage would duplicate the independent scientific
  production-domain oracle and blur capability ownership.
- Compiling the historical interior domain would not match the candidate domain
  that current production-domain evidence evaluates.

## Invariants

- `compiled-cpu` proves artifact compilation/execution, not scientific boundary
  correctness.
- The exact binding uses production-candidate/v1 and native domain version 2.
- The native smoke must match the same bound Graph within an explicit FP64
  tolerance.
- Stored evidence revalidates binding schema/name/domain/version/features/mask and
  content hashes before promotion.
- No C++ compiler or generated executable is invoked by ordinary runtime import.

## Evidence

Focused tests cover pass/fail receipts, wrong-domain rejection, result tampering,
smoke tolerance enforcement, and nonpromoting compile failures. A real
non-curated GGA test compiles and executes the generated C++ point binding when a
C++ compiler is available.

Repository CI is the executable authority for the stacked branch.

## Consequences

The automatic KS qualification path can consume a real `compiled-cpu` fact
instead of fabricated test envelopes. The next integration step can tighten the
capability registry so only this exact qualification schema is accepted and then
cross-check it against production-domain execution identity.

## Revisit when

The native point ABI changes, a packaged persistent CPU artifact replaces local
compilation evidence, or CPU and CUDA share a common executable qualification
schema.

## References

- #1119
- #1121
- #1123
- #1276
- #1331

Agent: ChatGPT
Model: GPT-5.6 Sol

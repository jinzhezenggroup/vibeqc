# Decision: retire Direct-HF ssss force science through generated lowering

Status: implemented
Date: 2026-09-19

## Problem

Direct HF already had a production shell-class compiler, but low-order force code
still owned a second handwritten `ssss` derivative formula. That duplicated the
same scientific recurrence expressed by the compiler and kept native scientific
CUDA ownership higher than necessary.

Issue #356 requires migration to be evidence-driven: generated code is not
promoted merely because it exists, and a faster or safer native path remains
valid when the endpoint evidence justifies it.

## Decision

On the measured `sm_120` profile, `ssss` force is now a generated production
consumer. The dedicated `SsssWeightedGradient` type and
`contracted_eri_cartesian_source_ssss_weighted_gradient` implementation are
deleted. Portable or unprofiled targets retain only a scheduling fallback that
uses the shared order-zero quartet gradient rather than a second `ssss`
formula.

The production Direct scientific families are classified as follows:

| Route | Families | Classification |
| --- | --- | --- |
| Force, `sm_120` | `ssss dppp dpdp dddp dddd dpss dsds ddss ddpp ddds dpds ddps fpps ppps dpps dsps dspp pppp psps ppss dsss` | generated/default |
| Force, `psss` | `psss` | measured performance exception; native low-order/resident/paged consumers remain default |
| Fock, bounded streaming | `ssss psss dppp dpdp dddp dpss dsds ddss ddpp ddds dpds ddps ppps dpps dsps dspp pppp psps ppss dsss` | generated/default |
| Fock, fixed topology | `ssss psss` | unsupported fallback; generated fixed-topology route is not qualified |
| Fock, fixed or bounded | `dddd` | unsupported fallback; native exact recurrence remains because the generated value consumer is not production-reliable |
| Portable CUDA | all unqualified classes | unsupported fallback; the portable production profile is intentionally empty |
| Reference force | all supported reference classes | oracle only, never selected as a production replacement |

The manifest can still contain a generated `dddd` Fock artifact because its
force consumer is qualified. Runtime masks keep that value consumer out of
production while allowing the generated `dddd` force consumer.

## Rejected alternatives

Promoting `psss` force was rejected. Its generated fused kernel beats the
independent recompute oracle, but the committed complete endpoint measurements
remain slightly slower than the tuned handwritten path. Synthetic wins alone do
not justify changing the production default.

Deleting all fixed-topology low-order native Fock code was also rejected. The
streaming generated rows do not by themselves establish a qualified fixed-task
replacement, so those paths remain explicit rather than being silently routed
through a generic scan.

## Invariants

- Every generated production promotion must preserve RHF and UHF energy/force
  endpoints and must pass resource gates without spills.
- A class may have generated source without being a generated production
  consumer; runtime qualification is a separate decision.
- Portable correctness must not depend on the `sm_120` profile.
- Retained native scientific paths must have an explicit fallback, oracle, or
  measured-exception reason.

## Evidence

Historical implementation-worktree measurements on an RTX 5090 used 1,024
isolated `ssss` force tasks with two primitives per shell. The fused generated
kernel took 0.196294 ms versus 0.237885 ms for the recompute oracle (1.21188x),
with maximum force difference `5.20417e-18 Eh/bohr`. RHF/UHF persistent kernels
used 122 registers; ordinary kernels used 126; all four reported zero stack and
zero spills. These numbers are retained as development evidence rather than a
standalone reproducible benchmark artifact.

The maintained CUDA ownership report falls from 6,889 to 6,790 handwritten
scientific lines under `src/scf/cuda/direct_*`: a net reduction of 99 lines
without reclassifying the deleted formula as runtime or fallback code. The
subsystem inventory correspondingly falls from 7,090 to 6,991 Direct-integral
scientific lines while Direct runtime remains 1,644 lines.

After rebasing onto master `280754b9`, exact-head validation on the RTX 5090
completed successfully: the full `cuda-dev-fast` build linked `libvibeqc.so`;
the complete codegen/ownership Python suite passed 307 tests with 48 gated
skips; Direct/CUDA SCF structure checks passed 65 tests; five focused GPU RHF/UHF
and exact-vs-bounded-streaming regressions passed; and the native RHF, UHF,
mixed-precision and spherical/gradient executables all exited successfully.

## Consequences

`ssss` has one scientific owner for the promoted force mathematics: the
compiler lowering. Native code still owns dispatch, queues, screening, density
contraction, atomic accumulation, and the shared portable fallback schedule.

The remaining handwritten Direct science is now intentional rather than an
unclassified migration backlog: `psss` is performance-qualified as an
exception, `dddd` Fock is a correctness exception, and fixed low-order Fock
still lacks an independently qualified generated route.

## Revisit when

- a generated `psss` force route beats the native complete endpoint by a
  material margin without changing work or accuracy;
- generated `dddd` Fock passes independent numerical qualification; or
- a generated fixed-topology low-order Fock route is measured and qualified.

## References

- #356
- `docs/shell_codegen.md`
- `docs/weighted_eri.md`
- `python/vibeqc_compiler/integral/production_shell_classes.json`
- `docs/cuda_ownership_current.json`

# Decision: specialize admitted CUDA XC point consumers

Status: implemented
Date: 2026-09-23

## Problem

The immutable XC plan already knows its functional and physical/response
consumer, but the generated point kernel received both as runtime choices.
Consequently one device entry contained physical LDA/PBE/r²SCAN and signed
LDA/PBE response. The retained #1113 PBE96 capture measured 20,736 point calls,
2.211708581 seconds, two blocks of 128 threads, and 255 registers per thread.
A compile-only PBE physical ablation reduced register allocation to 148 and
removed static spills. Static resource counts alone did not demonstrate dynamic
spill traffic on the PBE branch or a complete-endpoint improvement.

## Decision

Propagate the admitted `(functional, response)` key into compiler-emitted point
consumer templates. There are exactly five launchers. The plan resolves one
launcher during preparation and keeps it for its lifetime; execution and graph
capture bind the same validated buffers to that entry. Spin remains a layout
argument. AO precision remains a separate producer policy, and every point
consumer still uses FP64. Consumer pruning is measured first at the unchanged
128-thread launch width. A separate ablation qualifies a 32-thread block only
for physical PBE; other consumers retain 128 threads. Point tiles and device
workspace do not change.

Specialization calls the existing scaled LDA/PBE point source and generated
r²SCAN source with constant facts. It adds no alternate equation or AD system.
NVCC can eliminate unreachable physical/response and functional branches before
register allocation. The existing emitted resident block remains maintained
scientific text; this is not a claim that it has become a complete typed XC IR.

The source hash changes deliberately because the native generated source now
contains this finite dispatch and specialized entry set. The JIT grid consumer
does not acquire these native entries. Existing immutable layout and compiled
artifact identity determine the consumer; no serialized function pointer or
new density/geometry cache is introduced.

## Invariants

- Preserve scalar expressions, boundary/empty-spin policies, AD, finite checks,
  FP64 storage, disabled FMA, and existing acceptance tolerances.
- Preserve all admitted spin/functional combinations, invalid-input rejection,
  numerical-error isolation, exact arena size and asynchronous execution.
- Preserve rejection of r²SCAN response and FP32 AO response/r²SCAN requests.
- Do not infer 96-atom SCF convergence from an unconverged diagnostic.

## Rejected alternatives

Increasing point tiles at the same time would conflate specialization with
parallelism and workspace effects. Specializing spin and AO precision as well
would multiply AOT entries without establishing the need. Hand-written PBE or
r²SCAN equations would create a second numerical owner. Runtime JIT of a known
five-entry domain would add preparation and cache complexity unnecessarily.

## Evidence

Qualification is in progress. Baseline source is master `6dbffc10`; Slurm 11441
passes the original native LDA/PBE/r²SCAN RKS/UKS E/V and state suite. Artifacts
are retained in ignored `.artifacts/xc-point-consumers/`. Promotion requires
production GPU numerical and endpoint checks; compile-only results are not a
performance acceptance gate.

Slurm 11442 passes the expanded native production-kernel suite, including
captured/replayed physical consumers for every functional/spin and signed
LDA/PBE RKS/UKS response checked against CPU physical-potential differences at
two steps. Twenty host tests and compiler/ownership/pre-commit checks pass.
The standalone native grid object grows from 4,472,224 to 5,329,864 bytes.
Observed native-object compile times were 23.257/19.713 seconds; these are
concurrent-build observations, not a controlled compile-time speedup claim.

| Point entry | Registers/thread | Stack bytes |
| --- | ---: | ---: |
| Original generic | 255 | 2128 |
| Physical LDA | 98 | 96 |
| Physical PBE | 148 | 96 |
| Physical r²SCAN | 255 | 240 |
| LDA response | 255 | 1888 |
| PBE response | 255 | 2224 |

Slurm 11443 compares saved composed v12/v13 libraries differing only in this
four-file specialization. Both use the #1106/#1109 contractions, 768 AOs,
2,654,208 points, tile 256, 20,736 point launches, two blocks of 128 threads,
and two reported **unconverged** iterations. Point time is
2.199711866/2.175901059 seconds; complete profiled execution is
21.359858826/21.380673349 seconds, with preparation
4.082813183/4.056991470 seconds. **No meaningful endpoint speedup is established.**
Lower register counts do not prove that register pressure dominates this small
launch. The next ablation must address point scheduling separately.

The v13 library SHA256 is
`e705b1c0bf632272799beb21dc2afbfaef20bc1db0252b74ed17207c8d98183d`;
its composed source archive SHA256 is
`f4b73948f9b2067ae448304bf3460981031d434c7842d81fdf67845537d56467`.
This older composed tree supplies PBE performance evidence only; standalone
latest-master gates own r²SCAN qualification.

### Separate point scheduling ablation

Slurm 11445 retains the same PBE96 semantic work and tile size but uses eight
32-thread point blocks. Point time falls to 2.014273851 seconds and complete
profiled two-step execution is 21.206064242 seconds (preparation 4.061018431 s).
This is a modest improvement: density/potential contractions still dominate.
The diagnostic remains unconverged and does not qualify full96 endpoints.

Complete PBE24 direct/DF energy endpoints use identical 16 cold/2 warm native
iterations and pass all four measured GPU4PySCF energy pairs at the unchanged
1e-8 Eh gate (Slurm 11444/11445). These are clean endpoint timings, separate
from the profiles above:

| Variant | Direct cold / warm seconds | DF cold / warm seconds |
| --- | --- | --- |
| Generic v12 | 20.0591 / 2.5211, 2.5454 | 10.5160 / 1.3731, 1.3681 |
| Specialized v13, 128 threads | 20.0195 / 2.5268, 2.5665 | 10.4515 / 1.3590, 1.3877 |
| Specialized PBE, 32 threads | 19.6856 / 2.4838, 2.5031 | 10.1142 / 1.3177, 1.3483 |

The experimental v14 library uses 32 threads for all consumers, but **only PBE
was performance-tested**. The production schedule therefore selects 32 only
for physical PBE and retains the prior width elsewhere. Its PBE specialization
is the same measured program. No spin/precision Cartesian product is emitted.
The experiment's library/archive SHA256 values are respectively
`0f049dc22ea0ec387573755fb883215b7d97426ec7863b468a575bdce69e7a08` and
`c1cb650a6e22ccc80086d07a9922894f99e83a9125d1049ebb27c452d0e883e5`.

The final PBE-only schedule passes the expanded standalone native suite and
compute-sanitizer memcheck with zero errors (Slurm 11446). All final PBE24 pairs
are within 9.10e-12 Eh (direct) / 4.17e-11 Eh (DF); the independently serialized
cold-energy differences are 9.10e-12 / 3.02e-11 Eh. Preparation is
0.390062403 / 0.608762536 seconds. These timings do not include forces.

Broader public tests on the old composed tree are **not an accepted gate**:
Slurm 11447 reports 26 failures, 8 passes and 1 skip across selected direct/DF
SCF and replay tests. Slurm 11448 reproduces the minimal direct H2/PBE failure
with both the pre-change v12 and specialized v13/v15 libraries, so at least
that failure predates this repair. Latest-master standalone endpoint gates are
required to isolate this integration limitation; do not weaken the tests or
use the passing PBE24 results to assert all small endpoints pass.

The final composed v15 library/archive hashes are
`1cb7c035e2f37ba23ff1f28edd0a0b32e06eb3910c777d2dd8efd91e2a8e2d3b` /
`48e600e431270edf982aa737e5f87a40bd827baddf6e464e7140384adbea04e1`.

## Revisit when

New admitted point roots or precision domains exceed this small fixed set, or
profiling shows that point launch geometry rather than consumer resource demand
dominates the remaining cost. A future typed point IR must reuse the existing
canonical equations and boundary qualification.

## References

- #1113: point consumer specialization and separate scheduling qualification.
- #1102: complete CUDA XC/SCF pipeline performance and 96-atom endpoints.
- #1108/#1114: independent r²SCAN boundary qualification.

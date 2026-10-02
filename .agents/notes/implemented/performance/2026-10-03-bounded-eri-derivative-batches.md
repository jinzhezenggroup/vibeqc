# Decision: batch weighted CUDA ERI derivatives across shell quartets

Status: implemented
Date: 2026-10-03

## Problem

The [completed force diagnosis](2026-10-03-cc-force-phase-diagnosis.md) found that
28-AO CCSD(T) derivatives consumed 22,210 separately synchronized primitive
batches per endpoint. Most launches left very little work on each GPU. Host
allocation reuse alone would retain that schedule; AO weight transformation was
only about 0.1 seconds of the roughly 6.2-second derivative stage.

## Decision

Let the conventional MO-to-AO derivative traversal feed an accumulating shell
consumer and finish it before publishing the gradient. The CPU/one-shell adapter
still computes and scatters twelve center derivatives synchronously. CUDA uses
`CudaEriDerivativeBatch`: a bounded host primitive stream plus output-tile atom
maps. The one-shell and batch paths share the same validation, normalized
Cartesian/spherical expansion, and primitive record producer. No integral
recurrence, precision, orbital response, or symmetry projection changes.

Keep at most 16,384 primitive records and 256 output tiles. Flush when either
capacity fills, and remap the current quartet if a primitive flush splits it.
Keep all four derivative slots distinct until the completed result is added to
its physical atoms. Several slots may share an atom and must all contribute.
Zero-weight quartets create no output tile. Drain the final partial buffer before
the final finiteness check; exceptions discard the method's unpublished gradient.

The existing unscreened CUDA Hermite/Dual3 consumer accepts multiple output tiles.
It launches one kernel for each angular order present in a batch: a batch count
is therefore not a kernel count. Each record still contributes once to its
matching-order kernel. There is no screening, recomputation of integrals, CPU
fallback, full derivative tensor, or retained state across prepared executions.

## Capacity and compatibility

Cap batch numeric storage by both the caller stage budget and the existing
8-MiB post-HF source scratch envelope. Charge host records, device upload storage,
host/device results, tile-to-atom metadata, maximum current shell expansion, and
the producer's temporary record. The STO-3G batch plan is 6,878,256 bytes. It fits
the already admitted force envelope; public force capacity/work fields do not
change. Record the batch plan separately in the optional derivative trace. As
in the existing weighted consumer contract, numeric payload accounting excludes
allocator metadata, CUDA context storage and implicit kernel stacks.

Allocate buffers lazily on the first ERI callback, after one-electron derivative
scratch has been released. This prevents accidental lifetime overlap with that
consumer. The system is borrowed and must remain stable until finishing; neither
weights nor gradient spans are retained. Lower budgets reduce primitive/tile
capacities. Below batch admission, keep the bounded one-shell CUDA consumer,
including explicit rejection when even one primitive cannot fit. The legacy
shell budget now also charges its shared producer's temporary record.

The derivative trace includes final draining in shell-consumer time; the separate
`two_electron_finalize_ns` observation is nested there. Do not sum it a second
time. The synchronous adapter now includes atom scatter in its shell callback.

## Evidence

`benchmarks/results/cc-derivative-batches-20261003/` retains all outputs, maximum
errors, disjoint/nested phase observations, work counters, paired samples, and
source/binary provenance. Apply its patch to the exact parent revision to
reconstruct the measured native code. Later notes and evidence do not change the
qualified library.

Retained records factor shared case/row defaults and column names without
rounding values. The JSON `row_encoding` field gives the exact reconstruction
rule, verified against every original row before publication. Work observations
similarly factor profile defaults. The only large retained file contains all 112
complete outputs; its exact bytes and rationale are recorded in the large-evidence
storage review. Aggregate and per-file repository caps remain unchanged.

n1 job 5462 uses one Slurm-assigned RTX 5090 for six adjacent parent/candidate
process pairs at each size, alternating AB/BA. Each process performs cold, warm,
warm-repeat and changed-geometry energy+force calls. Preparation is separate;
changed geometry moves atom 1 z by 0.01 Bohr. CUDA 12.9 sm_120 Release, fast compile
disabled, one CPU thread per numerical library, 8-GiB endpoint budget. All GPU
commands use finite 15-minute Slurm allocations and preserve device visibility.

| Size / workload | Parent mean | Candidate mean | Geometric mean speedup | Paired bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| 14 AO cold | 2.002 s | 1.679 s | 1.19x | 1.17–1.21x |
| 14 AO warm | 1.224 s | 0.895 s | 1.37x | 1.36–1.37x |
| 14 AO changed | 1.281 s | 0.949 s | 1.35x | 1.33–1.37x |
| 28 AO cold | 14.834 s | 10.210 s | 1.45x | 1.45–1.46x |
| 28 AO warm | 14.019 s | 9.505 s | 1.48x | 1.46–1.49x |
| 28 AO changed | 14.023 s | 9.467 s | 1.48x | 1.47–1.49x |

The two warm calls are averaged within each process; the bootstrap resamples
whole adjacent pairs, not individual warm calls. These intervals describe six
pairs on a shared node, not unmeasured hardware/bases. The evidence envelope is
accepted for numerical qualification: the stricter automated compiler promotion
schema also requires per-iteration solver history and measured compilation/full
memory costs, which this campaign does not claim to supply.

| Per 28-AO endpoint work | Parent | Candidate |
| --- | ---: | ---: |
| Canonical shell quartets | 22,155 | 22,155 |
| Primitive records | 7,137,963 | 7,137,963 |
| Completed CUDA consumer calls | 22,210 | 436 |
| Derivative kernels (independent Nsight count) | 22,210 | 1,307 |
| Primitive upload bytes | 1,484,696,304 | 1,484,696,304 |
| Result download bytes | 2,309,840 | 2,349,152 |

Splitting quartets at batch boundaries produces 22,588 output fragments instead
of 22,210; this slightly increases result bytes while collapsing transfers and
synchronizations. The 14-AO case keeps 516,132 primitive records and reduces
consumer calls from 1,546 to 32. All public CC work counters, hashes, iteration
counts and reported numeric capacities match the parent for every paired call.
The independent n2 Nsight run confirms kernel and transfer counts; its GPU/API
timings overlap and are not used for the paired performance ratios.

All 112 retained endpoint calls (96 paired plus qualification, memcheck and
profiling) pass independent PySCF 2.14 energy/triples/force/residual gates, using
exact repository primitives and corrected triples Lambda. Maximum errors are
1.43e-12 Eh, 1.06e-14 Eh and 6.67e-8 Eh/Bohr, below respective 3e-9, 2e-9 and 1e-6
gates. Response residuals remain below 1e-9.

Qualification includes 80 public CPU/CUDA CCSD, CCSD(T) and MP2 tests (including
batch forces), host MP2 contracts, and non-CUDA compilation of changed post-HF
sources. The focused native test covers Cartesian/spherical s/p/d/f components,
shared atoms, repeated quartets, zero weights, primitive/tile splits, final drain,
invalid shapes and 8-MiB/32-KiB/8-KiB/one-byte budgets. Primitive counts are
identical across admitted budgets and atom gradients match independent CPU shell
integrals. n2 memcheck reports zero errors for those native cases and all four
complete 14-AO force calls. The native `--eri-batches-only` mode excludes an
unrelated intentional cudaMalloc OOM test from sanitizer execution.

## Revisit when

Consider grouping angular orders or reusing the remaining consumer workspace
only after full endpoint evidence identifies a new material bottleneck. The
current batch retains bounded upload and output storage and does not require
identity-based caching. Native CPU triples response and Lambda now account for
larger fractions of force time and remain separate optimization targets.

The public CCSD(T) force domain remains at 28 AOs. This scheduling change does not
resolve the independently documented 56-AO degenerate orbital-response problem.

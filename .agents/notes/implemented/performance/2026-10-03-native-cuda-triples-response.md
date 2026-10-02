# Decision: execute the native standard-(T) response on CUDA

Status: implemented
Date: 2026-10-03

## Problem

After bounded ERI derivative batching, the 28-AO complete force endpoint still
spent about 2.67 seconds evaluating its triples response on CPU. CUDA forces
already executed the energy, Lambda and Hamiltonian/orbital graphs on device.

## Decision

Reuse the runtime-indexed triples TensorIR VJP and its generated CPU arena plan.
The CUDA generator verifies that backend preparation has identical logical
identity and symbolic liveness slots before sharing the admission function.
Emit a separate CUDA translation unit so this new evaluator does not alter the
existing large CCSD CPU/CUDA generated files. Both remain byte-identical to the
parent. Reuse the compiler's range-safe scaled-bilinear arithmetic helper.

The native CUDA owner uploads the eight physical inputs once, retains an arena
and eight accumulators, uploads only maps/active masks/degeneracies per triangular
virtual-triple page, and downloads completed projected cotangents once. A page
has at most 16 lanes by default. Preserve last-two ovvv/ovoo and composite-pair
ovov symmetry projection. Inactive lanes keep valid maps and denominators.

Use output-driven indexed scatter with one writer per destination and ascending
lane summation. This preserves deterministic source-order accumulation without
atomic FP adds. Each output examines at most q lanes; this is an explicit work
tradeoff, not a claim that memory bounds alone bound computational cost. Other
reductions also retain source-major order. Finite/error gates remain active.

## Capacity and lifetime

Charge host outputs and page controls plus device inputs, accumulators, arena,
page controls, seed, error flag and final allocation padding. Borrowed reference,
problem, amplitudes and orbital energies belong to the enclosing force phase.
Compose both sides into the complete force budget, and include triples device
capacity in the public device high-water mark. Native owner admission reduces
q when necessary; reject explicitly if one lane cannot fit. The complete force
planner retains its existing conservative default-page admission.

Each page fences its error flag before host controls are reused. The storage
owner also drains pending transfers on unwinding and is destroyed before any
borrowed host transfer buffer. Release device storage before diagnostic string
publication. Numeric payload accounting follows existing post-HF conventions:
allocator metadata, CUDA context state and implicit kernel stacks are excluded.

## Qualification

Native fixtures compare all eight blocks to the independent full-six-index
triples VJP for (o,v)=(1,2),(2,3),(3,2), with page capacities 1,2,3,16 and exact
budgets. A separate runtime TensorIR reference exercises arbitrary repeated
maps, a non-unit energy seed and an inactive final lane. Malformed inactive maps
must report an error without an out-of-range access. Tests also cover reduced
page admission, one-byte-short rejection, invalid devices, nonfinite amplitudes
and noncanonical denominators.

`benchmarks/results/cc-triples-response-cuda-20261003/` retains all 112 complete
calls: 96 paired, eight qualification, four memchecked and four profiled. All
pass the independent PySCF energy/triples/force and response-residual gates.
Maximum absolute errors are 1.4211e-12, 1.0514e-14 and 6.6629e-8 respectively.
Forty public CPU/CUDA CC tests pass. Native and all four complete 14-AO calls pass
memcheck with zero errors. CPU response, exact complete-force budgets, publication
and diagnostic checks pass; both changed CC units compile with CUDA disabled.

n1 job 5474 uses one Slurm-assigned RTX 5090 for six adjacent AB/BA process pairs
per size, CUDA 12.9 sm_120 Release, one CPU thread per numerical library, and an
8-GiB endpoint budget. Each process executes cold, warm, warm-repeat and changed
geometry (atom 1 z += 0.01 Bohr) calls. Bootstrap resamples six whole pairs,
averaging the two warm calls within each process. Shared-node intervals do not
establish performance for unmeasured hardware or basis sets.

| Size / workload | Parent mean | Candidate mean | Speedup | Paired bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| 14 AO cold | 1.525 s | 1.445 s | 1.055x | 1.044–1.067x |
| 14 AO warm | 0.895 s | 0.815 s | 1.098x | 1.068–1.141x |
| 14 AO changed | 0.951 s | 0.861 s | 1.103x | 1.084–1.128x |
| 28 AO cold | 9.961 s | 7.317 s | 1.362x | 1.331–1.380x |
| 28 AO warm | 9.475 s | 6.635 s | 1.428x | 1.409–1.449x |
| 28 AO changed | 9.369 s | 6.817 s | 1.376x | 1.318–1.417x |

Paired 28-AO warm triples time falls from 2.6593 to 0.02965 seconds. Other phases
also fluctuate: orbital response 0.6061 to 0.4809 seconds and derivatives 1.6138
to 1.5278 seconds. The retained ledger preserves all phase observations; do not
attribute the entire endpoint difference to the triples phase alone.

n2 jobs 2127/2129 separately qualify memcheck and Nsight. Per 28-AO endpoint,
Nsight confirms 4,776 generated VJP launches plus 64 projection launches over
eight pages, 1,011,432 H2D bytes and 1,006,336 D2H bytes. All eight scientific
inputs are uploaded once; all eight cotangents are downloaded once. Phase
numeric capacity is 94,317,664 bytes, of which 93,310,720 are device-owned. The
public device peak rises from 89,604,096 to 93,310,720 bytes; the complete numeric
peak is unchanged because a different phase dominates. Every other public
integer/string/bool work/capacity field matches in all paired calls. The 22,155
shell quartets, 7,137,963 primitive records and 436 derivative consumer calls are
unchanged. No faster integral traversal or altered scientific work is claimed.

Retained JSON factors repeated field names/defaults and verifies exact value
reconstruction; no floating-point rounding or samples are discarded. Its schema
states how to recover full records. The retained source patch reconstructs the
measured native source at its parent revision. A subsequent generator dependency
repair delays NumPy until interpreter evaluation: all three emitted CC files
remain byte-identical, including the CUDA triples evaluator. The stdlib-only
regression and scaled-arithmetic tests report 12 passes (12 GPU tests skipped).
This fixes minimal CMake builds without changing measured runtime mathematics.

Numerical retention is accepted separately from the paired speed observations.
The stricter automated compiler-promotion envelope additionally requires full
per-iteration histories and measured complete-memory/compilation costs; this
change does not claim that envelope.

## Alternatives and limits

Calling the development Python/CuPy triples response from production would add
an undeclared runtime dependency. Reimplementing triples algebra in CUDA would
create a second scientific owner. Atomic scatter would change accumulation
order. None is needed for this native TensorIR lowering.

This change does not enlarge the 28-AO force qualification boundary or resolve
the 56-AO same-space degeneracy/gauge problem. Revisit page scheduling or kernel
fusion only if complete endpoint profiles show that this phase is again material.

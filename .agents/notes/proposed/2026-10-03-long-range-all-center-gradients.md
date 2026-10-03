# Proposal: share high-order LR moments across center derivatives

Status: experimental; host and independent GPU numerical gates pass;
controlled full24 and moved12 pass; full96 comparison pending
Date: 2026-10-03

## Problem and decision

The existing compiler-emitted all-center full-range gradient covers total
angular orders 4, 5 and 6. The bounded LR consumer still repeats its separate
Dual3 contraction for every unique atom. Extend the same all-center primitive
and contraction with a default-false LR template parameter. Select it only for
LR orders 4--6 in the bounded range force consumer. One primitive/AO quartet
produces one LR moment ladder and all center derivatives before atom scatter.

At fixed primitive exponents and omega, LR moments satisfy
`dM_n/dT = -M_(n+1)`. The established shared range-moment owner therefore replaces
only the Boys ladder. The compiler keeps ownership of Wick pair coefficients,
Coulomb recurrence, canonicalization, primitive contraction and center mapping.
No independent recurrence or production CPU oracle is introduced.

## Invariants and fallback

Full-range is the default specialization and retains its existing arithmetic.
Low-order LR roots from #1756, short range, fused RSH, orders above six,
AO normalization, screening and source coefficients keep their existing paths.
This adds no retained allocation or data transfer; per-thread local gradients
remain bounded. The invalid-input branch publishes a compile-time NaN marker
for every coordinate, preserving the downstream numerical failure contract and
NVCC/CuMetal portability. Repeated shell centers accumulate into their shared
atom before the final unique atom is recovered by translation invariance.

This changes LR accumulation order. Scientific acceptance requires every
complete energy/force observation at 1e-8 Eh / 1e-7 Eh/Bohr and independent
CPU displaced-ERI tests. An all-center primitive may have more register/local
storage than a single-center Dual3 evaluator; fewer moment ladders alone do
not prove a faster complete endpoint.

## Evidence and retained identities

The isolated experiment starts from #1756 production commit
`ac8e14728e389d4a8791da9a603260455feccb81`, ultimately based on master
`d35ae539f645cb5e8b9c0c2fb7a426c5e2ad08e8`. That qualified baseline library is
`e51239008c378cf61e4703abd5e287825beb131bb192f1aea7a444a7a449f5bd`.
It does not include the source-screening or force-storage proposals. Existing
live benchmarks continue using frozen source/binary copies in their own roots.
Later master integration must retain its own source and binary provenance.

A host harness checks 19,440 center/axis derivatives across all ten pair-order
partitions of total orders 4--6, nine Cartesian orientations, three exponent
scales and omega 0, 1e-8, 0.3, 2 and 1e4. Its five-point displaced-value Hermite
contraction differs from the Wick gradient recurrence; maximum observed LR
absolute error is 2.01e-10. It also checks full-range derivatives, translation
balance and nonfinite propagation for invalid omega. This shared-moment
control is not a replacement for the independent native CPU oracle.

The native four-center gate is extended with d/p/p/s, d/d/p/s and d/d/p/p
fixtures, including repeated-center bindings and both spin contractions.
Those independent CPU displaced-ERI gates and complete molecular tests are
pending the candidate CUDA build. Artifact paths are ignored under
`.artifacts/range-all-center/`; no GPU speedup has yet been measured.

## Promotion

Require finite Slurm numerical/sanitizer qualification of the exact binary,
controlled complete cold/priming/three-warm endpoints, moved geometry and a
larger system. Retain and reject this direction if local storage or schedule
cost eliminates the anticipated reuse benefit. Do not infer actual executed
quartet counts from logical dense capacity.

## Frozen candidate real-device qualification

Candidate `d5ee2a85709e60d82f64a791f76b296899a56aa4` builds cleanly with all 442
compiler commands using verified ccache launchers. Independently recomputed
source identity `dcdbc29f79a2acee4d9f786077bd966898ea675716edc73641077d0550d8b5a0`
matches the native library. The library SHA-256 is
`795aaacff08f43b183d04646dddef12444fa71b8f825e4ed76ee36e689949449`, and the native
qualification executable is
`e8f51f3b81d7fd10c4c8a6ddcb6f94797490994c093aaa34d0f2dde31c36e056`.

n1 Slurm 5513 (finite 45-minute RTX 5090 allocation, exit 0) passes the native
independent CPU finite-difference gates, including new four-center order-4/5/6
and repeated-center fixtures. Memcheck reports zero errors. All seven complete
independent WB97M-V RKS/UKS force/rebuild/stale-state tests pass in 201.27 seconds.
The exact library and test hashes, source archive and log hashes are retained.

n1 Slurm 5519 now compares baseline and candidate at full-grid water24 in one
allocation, and 5520 checks the displaced water12 CPU oracle. n2 Slurm 2150 runs
both builds at full-grid water96 on RTX PRO 6000 with a finite seven-hour limit.
The latter starts with the baseline; its candidate phase requires the verified
5513 receipt matching the installed binary. Fresh full-Fock GPU4PySCF controls,
three engine-local warm repeats and all observations are retained. No timing
from these still-live jobs is promoted to an endpoint speedup yet.

## Completed isolated full24 and moved12 qualification

n1 Slurm 5519 completed (exit 0), comparing both frozen binaries sequentially
on one RTX 5090, full 48 x 16 x 32 grid, water24/192 spherical def2-SVP AOs.
Every one of five native/reference pairs per build and every fresh full-Fock
reference consistency gate passes. Maximum errors across both builds are
2.51e-12 Eh and 4.96e-10 Eh/Bohr.

| Complete endpoint | Base / s | Candidate / s |
| --- | --- | --- |
| Native cold | 255.054784309 | 252.644154511 |
| Native priming | 30.636664566 | 27.510406151 |
| Native warm median | 30.587367237 | 27.547977414 |
| Fresh reference warm median | 27.654653151 | 27.655192662 |

Native cold uses 18 iterations for both builds; all native warm/priming calls
use one. Reference cold uses 14 iterations and all warm calls one/two J-K
builds. The isolated native warm improvement is 9.937%. Candidate warm samples
are 27.529672, 27.547977 and 27.850305 seconds. Its median is only 0.107215 s
below the reference, and one candidate repeat is slower; this small margin
is treated as parity rather than a robust reference advantage. It does not
satisfy the large-system objective.

The first warm integral-derivative component falls from 6.300818 to 3.379603
seconds, while grid/pair drain stays 10.966036 versus 10.961820 seconds.
Grid, geometry, collocation and allocation work fields remain unchanged;
actual executed bounded shell-quartet counts are still not exported.

n1 Slurm 5520 also completes the reduced-grid water12/def2-TZVP original/moved
independent CPU gate. All five observations per build pass; maximum errors are
1.48e-12 Eh and 8.97e-10 Eh/Bohr. Single warm endpoints are 21.113944/20.208059
seconds, moved 91.407646/89.841197 seconds, with unchanged 22/1/1/11 iterations.
The full-grid 96-atom same-allocation comparison was restarted as n2 Slurm 2158
after the environment failure described below.

## Node2 JIT environment recovery

Original job 2150 failed after 1:17:01 during the first cold force generation:
the node's default GCC cannot execute `cc1plus`. No complete native cold,
warm or reference observation was produced. The original session exits 1 and
Slurm reports FAILED / NonZeroExitCode; raw JSON, log, runtime record and terminal
hashes are retained under `attempts/job2150-jit-host-compiler/`.

Use the existing complete `/usr/bin/g++-11` through `NVCC_CCBIN` for this task.
The explicit NVCC wrapper invokes the existing ccache, and its directory must
contain both the canonical `nvcc` entry and an adjacent matching `ptxas`:
compiler provenance queries these sibling tool names. Initial wrapper-layout
preflight 2155 failed before force qualification; its evidence is retained
separately and was not counted as a successful test.

Corrected job 2157 passes all seven complete independent force/rebuild tests
for each frozen build: 213.87 seconds for parent `e5123900...`, 195.44 seconds
for candidate `795aaacf...`. There are no skips. After copying the evidence,
both native-library hashes and both log hashes were independently checked
against the successful toolchain receipt. Compiler versions, wrapper/real-tool
hashes and before/after ccache statistics are retained. Slurm device visibility,
native binaries and scientific settings are preserved.

Only after these gates passed was the controlled full96 comparison restarted
as job 2158 with a finite eight-hour limit. Both variants use the corrected
environment; candidate eligibility still requires its original numerical
qualification receipt as well as the new environment gate. The failed job is
not a timing sample, and the retry remains pending performance evidence.

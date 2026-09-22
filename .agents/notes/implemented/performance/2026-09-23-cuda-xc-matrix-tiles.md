# Decision: compiler-owned bounded XC matrix tiles

Status: implemented; endpoint qualification in progress
Date: 2026-09-23

## Problem

Issue #1102: a bounded two-iteration PBE48 profile (Slurm 11350) attributes
48% of captured GPU kernel time to scalar density-product and potential
assembly contractions. The original density kernel independently reloads an
entire AO row for each output. Potential assembly repeats point coefficient
and gradient products for every AO pair. This is separate from preparation
(#1101) and source-integral work (#1078).

## Decision

Keep allocation-free resident execution. Compiler-owned 16-by-16 shared-memory
matrix tiles reuse density/AO operands; compact coefficient panels replace
per-pair gradient coefficient multiplication. The canonical AO-pair bilinear
moves from the XC jet-pullback builder into `dft.xc_bilinear`, which both
geometric pullbacks and matrix-panel lowering consume. The scalar Graph derives
panel coefficients. Only the shared neutral Graph/CUDA emitters are admitted
as IntegralIR dependencies for these two DFT owners; no integral recurrence or
method policy dependency is allowed.

GGA combines its three spatial coefficients into one panel. Symmetric cross
assembly uses half the scalar contribution in that panel. Meta-GGA retains
three additional half-kinetic panels; tau's scientific half is already in the
input coefficient and is unchanged. Signed weights and coefficients are legal.
The compiler preserves all quadrature points and AO pairs, applying weights
exactly once. FP64 numerical gates remain unchanged.

The work panels are reused only after reference/directional density features
and point coefficients are complete on the same stream. Their existing
spins*work_jets*tile*nao allocation is sufficient. Shared scratch is 4,352 bytes
for density and 8,704 bytes for potential per block (padded double arrays).
No extra global storage or device/host staging is introduced. Small shapes and
shapes exceeding the two-dimensional launch domain retain the original bounded
scalar kernels. Partial tiles load zeros without leaving barriers early; one
triangle is authoritative and mirrored exactly.

## Work accounting

The AO/point/spin census and density reduction count do not change. Each matrix
tile reuses each loaded density/AO value across up to 16 outputs. Potential
prepacking costs O(spins*points*nao*jets), instead of multiplying coefficient
and gradient factors independently at O(spins*points*nao^2). Symmetric matrix
assembly visits two legs per work jet (one jet for LDA/GGA, four for meta-GGA),
with no source/grid rebuilds or screening. Zero padding can add up to 15 lanes
per partial reduction block. Complete cold/warm/geometry and tile-tail evidence
must be checked before claiming an endpoint benefit.

## Rejected alternatives

A fresh cuBLAS handle retains library-internal allocations even with caller
workspace (the shared tensor runtime conservatively allows 96 MiB). Adding
that owner solely for these contractions would change global resource planning.
A future borrowed-handle schedule may be worthwhile but must qualify lifetime,
workspace, full endpoints and a bounded fallback. Increasing point tiles alone
changes resource pressure without removing repeated coefficient work. Private
native mathematical loops would split compiler scientific ownership.

## Evidence

49 host compact-factor/contraction/response/geometric derivative tests pass.
The new factor test uses independent dense matrix algebra, arbitrary AO jets,
signed weights and signed coefficients for LDA/GGA/meta-GGA. Native tests add
Cartesian and spherical f-shell shapes above the 16-AO threshold, partial AO
and point blocks, all three semilocal families and both spin modes, exact arena
canaries and CUDA Graph execution. Slurm 11380 passes the allocated dedicated native matrix schedule suite,
including variational finite differences and failure isolation. Slurm 11373
passes 26 composed direct/DF KS energy/resource cases (two expected skips).

Slurm 11376, bounded two unconverged PBE48 iterations, retains 5,184 grid tiles
per iteration / 10,368 contraction calls. Compared with Slurm 11350, density
contraction aggregate is 1.701→0.646 s, potential contraction 4.132→1.636 s;
new panel formation remains additional work. Two SCF iterations take
11.754→8.101 s. Both profiles include instrumentation overhead and preparation
kernels in aggregate percentages; no converged endpoint claim follows from
these diagnostics. Preparation is 0.974 s with the separate #1104 overlay.

An older composed integration fails the pre-existing zero-beta r2SCAN tail
case at n=2 (Slurm 11372), below this schedule's admission threshold. That base
predates merged #1054. The independent candidate is based on master 959b5612
and includes #1054; its full original native suite reaches a finite but unequal
zero-beta r2SCAN V element, 2.410941123885864 versus 2.4109061845583866
(Slurm 11374/11377). The new matrix cases pass before that unchanged n=2
fallback case. Keep the original full suite and gate intact; the dedicated
matrix CTest also exposes new schedule coverage independently. The baseline with no tiling changes reproduces the identical discrepancy
(Slurm 11382, head 086acada), so this numerical boundary is separately tracked.
Do not misattribute it to tiling.

## Revisit when

A reusable shared BLAS owner can preserve exact budgets, or endpoint profiling
shows that register blocking, tile shape or another contraction family wins
without changing the admitted mathematical work.

## Composed complete endpoint qualification

Slurm 11381, water24 / 192 spherical def2-SVP AOs / identical grid and FP64
settings, two warm repeats, all four native/reference energy pairs admitted:

| Path | Prior cold execute s | Candidate cold execute s | Candidate preparation s | Candidate warm s | Max error Eh |
| --- | ---: | ---: | ---: | --- | ---: |
| Direct PBE | 27.1023 | 21.9398 | 0.3845 | 2.7771 / 2.7904 | 1.103e-11 |
| DF PBE | 17.6167 | 12.3651 | 0.5953 | 1.5922 / 1.6056 | 3.752e-11 |

Prior baseline is the preserved v9 integration (Slurm 11363/11366); candidate
v10 adds only the XC contraction overlay to that integration. Both retain
preceding setup/grid/source repairs and are not standalone PR-head results.
Neither grid, convergence tolerances nor density-fitting settings were changed.
Both cold solves retain 16 iterations. These improvements qualify small full
energy endpoints, not full 48/96 endpoints or force performance.

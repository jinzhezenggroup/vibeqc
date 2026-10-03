# Decision: reuse the bounded shared eigensolver for CC force stability

Status: implemented
Date: 2026-10-03

## Problem

The complete 56-AO CCSD(T) force spends about 59–94 seconds in its orbital
response phase. Its private Jacobi minimum-eigenvalue check scans all matrix
entries to select every pivot, giving approximately fourth-power work in the
occupied/virtual response dimension. At 56 AOs that dimension is 640. The
physical matrix and stability certificate are necessary, but this pivot-search
schedule is not part of the scientific contract.

## Decision

Use `tensor::cpu_symmetric_eigen` with an explicit scalar plan. Its existing
cyclic Jacobi implementation avoids a global matrix search for each rotation
and explicitly refuses non-convergence. Preserve the full generated response
matrix, symmetry gate, minimum-curvature threshold of 1e-8, generated GMRES
actions and independent dense Z-residual check.

The scalar owner temporarily retains its matrix copy, working eigenvectors,
sorted output eigenvectors, eigenvalues and a size_t sorting permutation. Charge
all three additional matrices and both vectors in the complete-force phase
plan. The explicit scalar plan prevents optional external LAPACK allocations
from creating an unaccounted workspace dependency. This decision preserves the
CPU stability consumer already present in the production contract.

Completed-force traces distinguish response-matrix formation from curvature
checking, retain the full matrix dimension/column count, and report curvature
as a 17-digit decimal label. The progress counter API is integer-only: using it
for a small positive curvature silently truncates the diagnostic to zero.

## Alternatives

Do not replace the full stability certificate with a cheap diagonal estimate,
remove matrix columns, loosen the curvature threshold, or skip the independent
physical response audit. An iterative approximate minimum eigenvalue requires
separate rigorous acceptance, including negative and near-threshold cases.

A values-only shared eigensolver could avoid retaining eigenvectors, but adding
a second numerical implementation is unnecessary for this change. Revisit a
shared values-only API when the admitted storage or remaining eigensolver time
is an actual complete-endpoint limit. Optional LAPACK selection first requires
an explicit workspace contract.

## Validation

Known spectra transformed by independent NumPy QR rotations cover negative,
positive, repeated and near-threshold eigenvalues on both sides of the physical
stability gate. The ordinary-spectrum absolute eigenvalue gate is 2e-12.
A second scale reaches eigenvalues of 4e4 while retaining minima near 1e-8;
independent LAPACK of the actual stored FP64 matrix supplies its reference.
Its error allowance is the larger of 2e-12 and 64 epsilon times the largest
spectral magnitude; the physical stability decision must still agree exactly
with both the independent result and the intended side of the 1e-8 boundary.
Physical H2O, NH3, water dimer and internally degenerate methane tests intercept
real allocations and verify exact complete budgets, bounded provider fallback
and one-byte-short refusal. They also parse the curvature trace to prevent the
integer-truncation regression.

All 100 retained complete force calls pass pinned corrected-Lambda PySCF energy
(3e-9), triples (2e-9), force (1e-6) and response-residual (1e-9) gates. Four
AB/BA process pairs per size preserve every original integer/string/bool public
field and every non-timing completed force/derivative counter, including complete
numeric capacity and generated response work. Original, warm, repeat and changed
geometries are all retained. Slurm n2 PRO6000 measurements give:

| Endpoint | Baseline seconds | Candidate seconds | Paired speedup (95% descriptive bootstrap CI) |
| --- | ---: | ---: | ---: |
| 14 AO warm | 0.391712 | 0.390524 | 1.003 (1.002–1.004) |
| 28 AO warm | 4.685527 | 4.439716 | 1.055 (1.053–1.058) |
| 56 AO cold | 124.927231 | 77.143111 | 1.619 (1.617–1.622) |
| 56 AO warm | 123.801809 | 76.167804 | 1.625 (1.624–1.627) |
| 56 AO changed | 156.072728 | 73.744379 | 2.116 (2.112–2.123) |

The 56-AO pairs run across two GPUs, with both variants of each pair in one
allocation. Bootstrap resampling uses whole pairs; these intervals describe
this campaign and do not establish a hardware-population guarantee. The small
14-AO cold endpoint changes from 1.031936 to 1.033165 seconds (+0.12%); this
cost is retained rather than excluded. All 44 public CPU/CUDA CC tests pass.

The [publication](../../../../benchmarks/results/cc-curvature-eigensolver-20261003/publication.json)
retains exact snapshots, binary hashes, all process outputs, independent gates,
work counters and phase observations. Its test-only qualification patch restores
the expanded final tests on top of the measured production delta. The earlier
integer-only curvature diagnostic and interrupted comparisons are excluded.

## Retained boundary

### Review correction: absolute off-diagonal accuracy

The default shared solver stops at a relative off-diagonal tolerance. Applying
that rule without the former CC absolute cap can accept a matrix whose minimum
is below the unchanged 1e-8 stability gate. For example, a 2-by-2 block with
diagonal 1.01e-8 and coupling 3e-10, alongside an uncoupled 4e4 eigenvalue,
has minimum 9.8e-9. The global relative tolerance skips its coupling and reports
1.01e-8. Rotations of this clustered small-eigenvalue example reproduce the
problem; the original tests isolated their small eigenvalue.

CC now requests the stricter of the existing relative tolerance and an absolute
1e-13 off-diagonal cap through an explicit scalar-only overload. The original
shared API and all default consumers retain their prior behavior. This restores
the prior CC stopping accuracy; it is not a rigorous eigenvalue error bound.
There is no additional numeric storage or provider-dependent workspace.

The 100 complete-call observations and performance numbers above retain their
original measured source and binary identities. They predate this accuracy
repair and must not be described as measurements of the repaired source.

The native conventional CCSD(T) force limit remains 56 AOs. No new GPU
mathematics, derivative equations, precision policy, response tolerance, or
reference-oracle dependency is introduced.

# Decision: do not promote strided VV10 partner-lane reduction

Status: rejected as a production candidate on current evidence
Date: 2026-10-03

## Motivation and experiment

The qualified WB97M-V composition reaches small-case parity but remains slower
than GPU4PySCF at full48. Integral derivatives take about 10.37 seconds there;
the grid/pair drain takes about 44.97 seconds. An isolated scheduling experiment
therefore extracted the unchanged admitted VV10 scalar roots and row kernel
from `cac0727f3` without editing production code.

The prototype assigns 2, 4, 8 or 16 adjacent CUDA lanes to a row. Lanes sum
strided partner subsequences and then reduce through a fixed subgroup tree.
Every pair is still evaluated once for each ordered row/partner combination.
The scalar pair expressions and FP64 precision are unchanged, but summation
rounding changes. This is not a work-reduction algorithm. Subgroup masks exclude
screened/padded rows, and only the first lane publishes each row.

The probe uses NVCC 12.9, sm_120, `-O3` and default FMA behavior, with an explicit
ccache invocation. It tests the VV10 specialization, not rVV10. The retained
probe-library SHA-256 is
`ea157ac3a50cd9bab4a46f7b11cd363ceb885190f53f62b1c6aa42320559b361`.
Source, generated-header and original/transformed-kernel hashes are recorded
separately in `partner-lanes-probe/provenance.json`.

## Numerical evidence and its limits

n1 Slurm 5535 passes 160 lane/domain/shape checks, including partial subgroups,
signed and active-zero weights, screened rows, empty partner sets and output
canaries. Thirty small-input comparisons use independent 90-digit pair-energy
finite differences for the feature and geometry partials. Memcheck and
synccheck report zero errors.

The first larger run nevertheless fails its strict serial-GPU agreement gate:
six of 560000 fields exceed `atol=2e-11, rtol=2e-12` at 80000 points, with
maximum violating absolute difference 6.27e-11. Preserve that failed attempt;
it is not a successful qualification. A separate diagnostic job 5537 records
all timings and disagreements without claiming acceptance or changing that gate.

A selected-row independent extended-precision investigation (63-bit mantissa,
direct rational derivatives rather than generated energy-denominator roots)
finds six gate failures for the serial schedule and none for the tested parallel
schedules against that oracle. Maximum selected-row density-partial error is
9.69e-11 for serial and 4.15e-12 for 16 lanes. This suggests serial accumulation
rounding explains these discrepancies; it does not qualify every output or
replace complete independent molecular force tests. No production build or
full endpoint qualification is claimed for this prototype.

## Measured performance

The constructed dense domains have exactly N squared executed pairs per launch:
6.4 billion at N=80000 and 102.4 billion at N=320000. Timing is device kernel
time only, with allocation, host transfers and oracle work outside the timer.
The 80000-point values are medians of three interleaved observations, each
averaging three launches; the 320000-point values each average three launches
in one observation. Every timing and numerical disagreement is retained.

| Synthetic points / output demand | Serial seconds | 16-lane seconds | Reduction |
| --- | ---: | ---: | ---: |
| 80000 / SCF features | 0.209190 | 0.191572 | 8.42% |
| 80000 / features plus geometry | 0.289530 | 0.265077 | 8.45% |
| 320000 / SCF features | 3.123475 | 3.058738 | 2.07% |
| 320000 / features plus geometry | 4.316154 | 4.247833 | 1.58% |

The apparent smaller-input gain largely disappears as the workload grows.
Four lanes reach 4.239760 seconds for the larger geometry case, also less than
a 2% improvement; choosing another tested width does not establish a substantial
larger-kernel gain. These dense synthetic counts must not be substituted for
the unexported active-pair counts of molecular endpoints.

## Decision and revisit condition

Keep the existing ordered production reduction. The observed larger-kernel
benefit does not justify promoting a new numerical reduction policy on these
component results, and it has no demonstrated complete endpoint advantage.
Revisit only with a materially different schedule or reduced pair work, followed
by independent full-array/molecular accuracy gates, explicit bounded resource
accounting and complete cold/warm/changed-geometry endpoints.

Raw failed and diagnostic attempts, all observations, oracle code/results and
compiler/cache receipts remain under ignored
`.artifacts/wb97m-large-stack/partner-lanes-probe/`. No existing live benchmark
was stopped, and no production source, numerical threshold or public resource
budget was changed for this experiment.

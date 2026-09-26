# Decision: accept the measured combined low-angular Rys mapping

Status: implemented
Date: 2026-09-17

## Problem and evidence

#394 supplied all 42 low-angular polynomial/Rys schedule candidates. The
workload-weighted campaign proposed adding Rys/compact for 001/002/100/200,
but isolated kernel results cannot establish complete endpoint performance.
#404 compared that combined map against the pinned post-#411 policy using
one library and one prepared state with a frozen checkpoint density.

Five interleaved complete force samples per arm gave baseline/candidate
medians of 0.709223/0.691502 s at 384 AO and 2.618872/2.549564 s at 768 AO:
2.50% and 2.65% savings. All ten pairs improved. Exact paired bootstrap
95% ratio intervals were [0.95643, 0.98063] and [0.96426, 0.98175]. Those
intervals describe the five retained pairs, not independent replication.
Energy-only medians were unchanged within 0.12% at both target sizes.

The originally declared 3% magnitude requirement **did not pass**. After
reviewing the measured benefit, the user explicitly accepted retaining a
2.6% improvement. `protocol.json` preserves the original rule and
`decision.json` records that separate acceptance; the result is not relabeled
as a pass of the original requirement. All numerical, work and stability
requirements remain in force.

## Decision and invariants

Promote Rys/compact for 000/001/002/100/200 and retain polynomial/compact for
101/110. 000 was already promoted and contributes no claimed new saving.
Remove the embedded baseline from the production manifest, while retaining
the optional compiler support for future controlled campaigns. The retired
SSS-only environment override remains retired.

Automatic selection remains limited to sm_120, 384/768 AO and equal auxiliary
dimension. 96/192 explicit-candidate runs qualify numerics only; they do not
expand this domain. FP64, screening-off, metric threshold, SCF tolerances,
device final validation and final-projection reuse remain fixed.

Both target cases execute three converged SCF updates, four J builds and four
K builds. The 384 graph route executes three eigensolves; the 768 occupied
route executes four including its imported-density factorization. Final
state checks accept without density corrections or candidate rejections.
The 768 force stage retains 13 derivative panels and its final projection;
raw-value reuse, transfers and shell work are unchanged between arms.

The largest energy/full-force errors across the target clean and diagnostic
samples are below 9e-12 Eh and 1.7e-10 Eh/Bohr, against independent matched
GPU4PySCF numerical references. Those archived references are not fresh
external timing evidence. #206 still owns the stock GPU4PySCF comparison.

## Rejected alternatives and consequences

Do not lower the recorded threshold retrospectively or describe this as a 3%
win. Do not widen automatic selection based on the smaller-domain diagnostic
gain. The historical `legacy` control is not the post-#411 baseline.

The campaign library grows by 324,304 bytes (0.20%) relative to the pinned
baseline. Per-arm live process GPU residency is unchanged in the clean-run
diagnostics. Separate sampled resource/control records define peaks and their
limits; they are excluded from clean timing. Actual derivative contraction
savings include all repeated panels, not only the first panel.

## References and revisit conditions

- [Retained endpoint evidence](../../../../benchmarks/results/issue404-combined-rys/README.md)
- [Campaign comparison design](2026-09-17-df-campaign-baseline.md)
- #394, #404, #206; independent occupied-Gram experiment #412.

Revisit for independently qualified architectures/domains, or when a combined
optimization changes the bottleneck. Measure combinations directly; do not add
this saving to a future Gram-kernel result.

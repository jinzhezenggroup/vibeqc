# Experiment: reuse VV10 partner indices for compact outer rows

Status: rejected for promotion after complete24 regression; default unchanged
Date: 2026-10-03

## Problem and decision

The warm water24 SCF trace of composition 0132d7584 has two admitted VV10
feature calls totaling 14.71135 s, 58% of instrumented GPU kernel time.
This intrusive trace diagnoses a remaining hotspot; it is not a clean endpoint
comparison or a force-component breakdown. Partner staging was already rejected
after complete endpoint measurements; do not repeat that experiment merely
because a reference implementation uses shared memory.

Under explicit `GENERATIVEQC_VV10_COMPACT_ROWS=1`, admitted molecular VV10
feature/geometry consumers reuse the existing sorted nonzero weighted-density
partner map as the outer-row map. Nonzero rows occupy adjacent lanes. A second
launch owns every zero row in original grid order: negative zero writes screened
outputs, while positive zero still evaluates observable potential/weight
responses. The two partitions are disjoint and cover every original row.

Compact launch offsets address partner-map slots, whereas complementary launch
offsets address grid rows. Both use the original finite launch bounds. The
map's count stays on device. No allocation, host readback, pair formula,
partner-addition order, energy reduction, density threshold, or actual pair
count changes. The original fallback executes alone when device admission fails;
unmasked, rVV10 and energy-only consumers retain their original schedule.

## Qualification and promotion boundary

Host execution of the production pair loop observes equal outputs and actual
partner visits with signed, positive-zero, screened, subnormal/underflow and
exceptional-domain weights. Device tests compare every output bit against the
original admitted schedule, including empty partners, positive-zero-only grids,
partial blocks, disjoint small chunks with nonzero offsets, output canaries and
sticky failure publication. Independent molecular energy/force tests and CUDA
sanitizers are required before measuring the complete endpoint.

The option remains experimental until same-source, same-allocation complete
energy/force comparisons, including a larger grid, show a repeatable benefit.
Do not infer speedup from row packing or unchanged semantic work. Retain the
original schedule if the extra launch or changed memory access loses performance.

## References

- [Rejected partner staging](../rejected/2026-10-03-vv10-shared-partner-staging.md)
- [Screened-row contract](../proposed/2026-10-03-scf-vv10-screened-rows.md)
- Ignored `.artifacts/vv10-compact-rows/` retains source/binary identities,
  compiler-cache receipts, validation and complete endpoint comparisons.

## Completed qualification and negative endpoint

The frozen prototype is c28d89a07 on the joint SCF/force-map composition,
source identity `37eaeda128af7638f90a602dd4b1607caa6886b840feed5aef226fda1acefb17`,
library `aa388e099eca0c2b331ef7d32b30a69ffeeeccab1761506da6e39850ec8b91ff`.
All 74 host kernel cases pass. n1 Slurm 5590 passes all 82 real-device scheduling
cases normally and under memcheck/synccheck, both with zero reported errors.
Seven independent complete WB97M-V molecular tests pass with joint maps, and
seven with force-cache allowance zero. Each group observes 66 successful
native calls and 467 XC submissions. No tolerance was changed.

n1 Slurm 5591 compares the option off/on in one allocation, with the same source,
library, GPU, 24 atoms / 192 spherical def2-SVP AOs and 589824 unpruned points.
Both engines use their own fixed post-cold densities. Original warm median is
26.626242 s; compact is 27.973673 s, a 5.06% regression. Paired GPU4PySCF medians
are 27.662087 and 27.677249 s. Complete native cold (synchronized preparation
plus first E/F execution) is 243.145527 s original and 254.182505 s compact;
paired reference cold is 117.654511 and 118.359315 s. Every one of the five E/F pairs per variant passes
1e-8 Eh / 1e-7 Eh/Bohr, and all reference XC backend flags are on-GPU.
`scripts/verify-matched.py` independently checks raw samples and identities.
There is no basis for a default or auto-selection PR from this result.

A separate frozen-layout diagnostic (Slurm 5594) reads retained force workspace
only after synchronized execution; its independent cold E/F errors are below
2.502e-12 Eh and 4.921e-10 Eh/Bohr. It sees 474505 nonzero partners, 475644
observable rows, 114180 screened negative-zero rows and 1139 positive-zero
rows. The retained domains imply exactly 225695456220 pair visits for the
inspected all-partner row loop; this is a source-derived semantic count, not an
atomic per-pair counter or an SCF count. Device admission is accepted.

Original rows occupy 15284 warps that execute pairs. The compact nonzero launch
occupies 14829, but the complementary positive-zero launch occupies another
685, totaling 15514. The original spatial grid already clusters screened rows
well. Packing one partition creates sparse lanes in the other, and neither
schedule reduces pair work. These counts explain why a large lane-utilization
benefit was not justified; they do not isolate the cause of the full 5% loss.

The diagnostic's first attempt (5593) failed in artifact serialization after
execution because BatchResult exposes forces per item. The corrected wrapper
uses the established comparator's item-forces API; both scripts and the failed
receipt are retained. This was not a scientific endpoint failure.

A previously submitted 96-atom A/B remains running as separate evidence. It
cannot retrospectively change the 24-atom regression. Revisit only with a
measured admitted domain or a different schedule that addresses both zero and
nonzero observable rows without losing signed-zero response or ordered sums.

## Completed larger control (2026-10-04)

Finite n1 Slurm 5595 completes the already-running 96-atom comparison on the
same frozen prototype. All ten independent cold/priming/warm E/F pairs pass
unchanged gates, with maximum errors below 1.210e-10 Eh / 4.864e-10 Eh/Bohr.
All priming/warm calls take one iteration and reference XC remains on GPU.
Original/compact native warm medians are 385.378282 / 395.113502 s, a 2.526%
regression; paired GPU4PySCF medians are 402.886167 / 403.349944 s. Native
complete cold is 3776.098416 / 3833.902642 s, versus paired reference
1827.706081 / 1828.588419 s. Both native cold solves take 23 iterations, both
references 16. Thus the completed larger control also rejects promotion.
The compact option's complete endpoint being slightly faster than its reference
does not make it an improvement over the same-binary original schedule.

The four completed variants at 24/96 atoms remain frozen to c28d89a07 and the
37eaeda1/aa388e09 source/library above; they are not latest-master measurements.
The experiment checkout `qc-wb97m-vv10-compact-rows-20261003` retains all raw
receipts under `.artifacts/vv10-compact-rows/`. The integration also retains the
completed raw reports, independent verifier, original input identity and
successful job outcome under ignored `.artifacts/compact-rows-completed-20261004/`.
Only this rejection rationale is brought into the main WB97M-V integration;
the compact-row implementation is not promoted.

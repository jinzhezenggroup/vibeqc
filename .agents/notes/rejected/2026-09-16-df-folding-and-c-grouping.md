# Decision: reject response-folding and C-grouping candidates

Status: rejected for isolated performance promotion
Date: 2026-09-16

## Problem

#392 identifies unnecessary shared FP64 response-folding atomics for s/p
shells. #393 identifies the nine global gradient atomics per active triple.
Eliminating these operations does not establish a complete-force speedup.

## Decision

Do not promote these measured candidates solely on their operation counts:

- Direct folding replaces collision-free s/p shared atomics with shared stores.
- Register folding retains those components in lane registers while preserving
  the general runtime tuple dimensions and coefficient loads.
- Identity folding retains one response component in its owning lane. Both
  public representations map s/p AOs injectively to Cartesian components with
  unit expansion and component-normalization coefficients. d/f retains the
  general expansion/atomic path.
- Serial grouping adds six persistent A/B sums to a subgroup's traversal of
  up to four homogeneous C shells. It scatters generated C for each active
  shell, then A/B once per owner. Signature, panel and geometry boundaries
  remain intact.
- Block grouping runs four C subgroups concurrently, publishes each shell's
  six contracted A/B coordinates into a disjoint 64-byte shared slot, waits at
  one block barrier, and combines the active slots before scattering A/B.
- Warp grouping uses 72-byte partial slots and a warp barrier, combining four
  C shells for compact SSS/one-p classes and two for two-p classes.

The direct, register and identity candidates did not demonstrate a clean
endpoint improvement. Serial grouping regressed both endpoints and
the derivative component. Its persistent sums increased registers. Block
grouping reduces register pressure and has a lower 384-AO endpoint median,
but its 768-AO endpoint and derivative component regress. Warp grouping has the
same qualitative outcome. None satisfies the promotion gates. Stop this round
of exploration and retain the existing production implementation.

The Rys work in #394 is **deferred**, rather than a measured performance
failure. Its limited numerical prototype has no completed force-library or
endpoint qualification. See the
[deferred Rys record](../../../benchmarks/results/issue394-deferred/README.md).
#392–#394 remain unresolved; this evidence does not close them.

## Evidence

The retained [records](../../../benchmarks/results/issue392-393-rejected/README.md)
pin the baseline, dirty-source reconstruction patches, native/library hashes,
raw clean samples, independent errors, work counts and separate profiles.
The baseline is the retained #395 diagnostic build of the #397 production path.
Reconstruction bases are recorded per build: the early direct/register patches
use `c5475c49b331c989fbe0aafb3ce19685d7cb254d`; the later candidates use
`1c3f2ab8d6df6f06bee526b89a807511ef726301`. Runs use default Release sm_120
flags, RTX 5090 through Slurm, one CPU math thread, spherical def2-SVP RHF,
the same frozen density and three actual SCF updates. No compilation overlaps
clean timing. Each row has five clean energy+force samples per binary.

| Experiment | AOs | Baseline median (s) | Candidate median (s) | Change |
| --- | ---: | ---: | ---: | ---: |
| Direct folding | 384 | 0.9230750641 | 0.9262858119 | +0.348% |
| Direct folding | 768 | 5.3288771540 | 5.3301386780 | +0.0237% |
| Register folding | 384 | 0.9230750641 | 0.9256038300 | +0.274% |
| Register folding | 768 | 5.3288771540 | 5.3300863979 | +0.0227% |
| Identity folding | 384 | 0.9219331860 | 0.9259967010 | +0.441% |
| Identity folding | 768 | 5.3291304151 | 5.3295289380 | +0.0075% |
| Serial grouping + identity folding | 384 | 0.9249890889 | 0.9339029749 | +0.964% |
| Serial grouping + identity folding | 768 | 5.3198781800 | 5.4178290700 | +1.841% |
| Block grouping + identity folding | 384 | 0.9249414711 | 0.9156757200 | -1.002% |
| Block grouping + identity folding | 768 | 5.3241894329 | 5.4372721051 | +2.124% |
| Warp grouping + identity folding | 384 | 0.9280275221 | 0.9211481821 | -0.741% |
| Warp grouping + identity folding | 768 | 5.3298044831 | 5.4292655380 | +1.866% |

Small differences near zero do not establish a statistically resolved
regression or improvement. The early direct/register clean samples had no
compilation overlap; their separate intrusive profile captures did overlap
some later compilation. `early-folding/conditions.json` preserves this limitation.

Identity folding eliminates 14,151,808 shared folding atomics at 384 AOs and
112,562,688 at 768 AOs. Its 768-AO derivative observations change from
1308.564/1305.741 ms to 1292.114/1291.999 ms, but this isolated component gain
does not produce an endpoint improvement.

Serial grouping reduces each 768-AO A/B channel from 85,155,840 to 39,864,528
global atomic updates; C remains at 85,155,840. Total gradient updates decrease
from 255,467,520 to 164,884,896. Nevertheless, derivative observations change
from 1305.182/1311.481 ms to 1387.652/1390.827 ms. Separate disabled-counter
Nsight captures show the 000/001/100/110 classes regressing by approximately
19%/46%/10%/18%. These are kernel timings, not hardware contention counters.

Serial grouping changes registers for 000/001/100/101/110/111 from identity-only
128/132/132/134/135/156 to 146/146/146/147/147/186. Shared storage is unchanged.
The evidence records actual static shared bytes and theoretical residency as
well as binary size and process device residency.

Block grouping has the same exact atomic counts as serial grouping. Registers
for the same six classes are 124/128/128/136/136/158, with 64 shared bytes per
subgroup added for partials. Yet its 768-AO derivative observations are
1417.289/1407.519 ms. The first component observation puts most regression in
000/001/100, around 28%/20%/28%; 101/110 regress around 10%/9%, and 111 is nearly
unchanged. A power-of-two shared-slot stride aliases FP64 banks among compact
subgroup leaders, and the block barrier waits across warps. These structural
costs motivate a warp-contained, padded-slot alternative; they are not measured
contention percentages or proof that the alternative will be faster.

The subsequently measured warp alternative also fails. At 768 AOs it reduces
each A/B channel to 46,195,344 atomics, with C unchanged at 85,155,840, for
177,546,528 total updates. Derivative observations increase from
1305.624/1298.855 ms to 1422.519/1410.378 ms. Separate Nsight observations put
000/001/100 at 171.385/72.115/162.233 ms versus
133.898/60.510/126.226 ms. Registers for 000/001/100/101/110/111 are
126/140/138/136/136/156. Padding and removal of the block barrier did not
produce an endpoint gain. These observations do not isolate a hardware
contention cause.

Scientific shell, primitive and component work stays identical. In particular,
768 AOs execute 28,385,280 shell triples and 175,132,672 primitive products.
Paired energy differences are zero. Serial-grouped forces differ from baseline
by at most 9.95e-13 Hartree/Bohr; independent force errors stay below 1.36e-10,
within the unchanged 1e-8 gate. Block-grouped paired force differences stay
below 1.02e-12 Hartree/Bohr with the same independent error bound. Warp-grouped
paired force differences stay below 8.25e-13 Hartree/Bohr. Identity folding and
all three grouping candidates passed the independent native
oracle, memcheck, and 120 CUDA force/response tests. Serial grouping also passed
223 CPU compiler/ledger tests. Block grouping passed those tests and synccheck.
Its initial Python trace-filter failure was fixed by matching complete primitive
signature names rather than the substring `_p`; all 120 CUDA tests then passed.
Validation hashes and exact scopes are retained.
Warp grouping passed 223 CPU tests, memcheck and synccheck. The early direct
candidate has native/memcheck evidence; its record does not claim a complete
Python CUDA suite. The early register candidate also passed all 120 CUDA tests.

## Invariants

Preserve FP64, response weights, normalization, public component ordering,
screening, metric rank and convergence gates. Generate the C translation
equation in the compiler. Never group across primitive signatures, panels or
geometries. Sparse/zero owners, clipped shells, all pair layouts and shared
physical atoms require independent coverage.

## Revisit when

New evidence identifies a different cost or a different mathematical lowering
changes the tradeoff. A smaller atomic count, a padded shared slot or a narrower
barrier alone is insufficient reason to repeat these experiments. Any resumed
work needs new native/sanitizer/force checks and clean endpoints. These rejected
timings cannot qualify a new grouping or Rys implementation.

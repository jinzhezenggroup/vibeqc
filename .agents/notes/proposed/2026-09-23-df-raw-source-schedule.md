# Candidate: distinguish raw and transformed DF source schedules

Status: proposed (implementation present; endpoint qualification pending)
Date: 2026-09-23

## Problem

The 96-atom water HF-DF fixture uses 768 public AOs and 3712 auxiliary
functions (def2-SVP spherical and cc-pVDZ-JKFIT). Its bounded source sends
raw tiles through the same launcher as metric-transformed tiles. The previous
automatic mapping assigned a whole primitive-reduction warp to each raw
output, even though raw outputs have just one source auxiliary and many short
primitive contractions. The bulk raw exporter has separate controls; changing
`VIBEQC_DF_VALUE_RAW_MAPPING` does not change these bounded source tiles.

## Candidate

The compiler emits a finite value-source mapping policy: raw outputs use
one lane per output with contiguous auxiliary writes, while transformed
outputs retain the existing cooperative primitive/source warp. Native source
creation freezes both resolved mappings. Launch dimensions and the kernel
mapping use the same frozen selection. Explicit auxiliary/component/primitive
overrides still select both consumers; `auto` resolves them separately.

This changes work placement only. Scalar equations, FP64 accumulation,
normalization, metric threshold, derivative scheduling, tile extents, budgets
and contraction consumers remain unchanged. No new handwritten recurrence or
GPU generation dependency is introduced.

## Evidence and limits

Bounded Slurm RTX 5090 diagnostics on the same original production library
(`78d01e579cff2ad2832e1b18effce007f27ecb1ac9bdc3e2806fea2268bd54e6`)
compared explicit mapping overrides at 96 atoms. Completed stream-execution
raw panels of 589824 AO pairs by 580 auxiliary functions took:

| Mapping | Seconds per full panel |
| --- | ---: |
| primitive (previous default) | 14.43–14.56 |
| component | 2.10–2.19 |
| auxiliary | 1.33–1.41 |

Partial-tail panels were measured separately. Graph-capture enqueue durations
are excluded: they are not executed GPU work. The probes stopped after short
finite deadlines; none of these numbers is a complete endpoint speedup.

The streamed fixture still generates two complete raw passes for J and seven
for K, or 19,704,840,192 source values per Fock. This candidate does not fix
that amplification or establish completion of the earlier 900-second case.
The dense B owner alone is 17,515,413,504 bytes; simply borrowing the response
budget or enabling the current double-buffered packed owner does not establish
safe capacity for the complete value/response lifetime.

## Rejected alternatives

- Global component selection regressed earlier transformed-source endpoints.
  The new policy distinguishes consumers rather than reversing that result.
- Promoting the specialized scalar-math candidate did not improve these raw
  panels and would ignore its retained failed endpoint gates.
- PR #1076 fixes later response algebra, not first-SCF source generation.
- PR #970 resident admission still streamed for this exact auxiliary basis;
  its other 768-AO evidence uses a different auxiliary basis.

## Acceptance and follow-up

Before promotion, run the emitted host schedule test and independent
libcint/NumPy raw, metric, RHF/UHF J/K checks through f, mixed representations,
different batch primitive offsets, long contractions, rank deficiency and
partial tiles (`tools/validate_df_source.py`, including automatic selection).
Compare complete cold, warm and changed-geometry endpoints with unchanged
energy/force and iteration gates, including a larger streamed case. Retain
short deadlines when the 96-atom endpoint remains abnormally slow.

Source-driven reuse remains follow-up work under #1078. Revisit this mapping
if long contractions or another target demonstrate a complete endpoint
regression; keep explicit controls for a reproducible comparison.

## Allocated source qualification

All 48 independent libcint/NumPy cases passed on RTX 5090: automatic plus
three explicit mappings, full and 7-pair/3-auxiliary tiles, Cartesian,
spherical, both mixed representations, long contractions and rank deficiency.
Each fixture has two geometries with different shell/primitive ordering;
raw/metric values and RHF/UHF J/K use the unchanged gates above. The integration
library hash was
`93b9d7ddf6366f618fd8037a9d85e626b2f441ab8bfdbc3c58be9a8ed7fa6ec5`.
Complete endpoint qualification remains pending.

## Bounded small endpoint A/B

After fixing the shared comparator's GPU4PySCF DF `direct_scf=False` policy
(PR #1085), HF-DF water12 (96 AOs, 464 cc-pVDZ-JKFIT auxiliaries, one system)
passed complete cold/priming/warm energy-plus-force comparison on RTX 5090.
Automatic mapping cold was 0.554 seconds versus explicit primitive mapping
0.785 seconds; warm medians were 0.06573 and 0.06565 seconds. Both use full
resident storage, so this does not qualify the large streamed work schedule.
Maximum warm-pair energy/force errors were 5.344e-12 Hartree / 1.424e-11
Hartree/Bohr (auto) and 5.230e-12 / 1.013e-11 (primitive), with the original
convergence settings. Native warm solves used two iterations; reference used
one. These are scoped endpoint comparisons, not iteration-matched speedups.

Evidence: `hfdf12-{auto,primitive}-v3.json` and `hfdf12-mapping-v3.log` under
the integration artifact directory, pinned to the same raw-v2 binary above.
Changed-geometry and larger streamed endpoint qualification remain pending.

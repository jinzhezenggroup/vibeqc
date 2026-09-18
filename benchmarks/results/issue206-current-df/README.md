# Current stock GPU4PySCF comparison: acceptance remains open

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

VibeQC passes the 192-AO batch-1/4 warm DF force timing gates in this campaign,
but it does **not** pass the complete #206 acceptance. Large DF endpoints are
slower than stock, strict numerical failures remain, and final direct-SCF
acceptance fails. No failed or missing cell is silently removed.

## Pinned protocol

All GPU runs use one Slurm-assigned RTX 5090. Stock versions are GPU4PySCF 1.8.1,
PySCF 2.14.0, CuPy 14.2.0, cuTENSOR 2.3.1 and NumPy 2.5.3. The imported cuTENSOR
backend is verified; no weaker contraction engine is substituted. Per-cell
qualification retains physical bases, metric extrema/rank, software versions
and backend identities. These equal-basis DF cells explicitly use the orbital
basis as the auxiliary basis; they do not substitute stock's automatic JKFIT.

The native DF source contains the merged #394/#404 work and the #409 candidate,
with **dense selected**. #412 split Gram was rejected and is absent. Frozen
native identity: `2d59ce1e7d001a6e9d0a28b90f44e6fcbcf8e752a6682b1d80a98738c72b3133`;
library SHA-256: `c1d4c033b73e3b42da58acab0b67c2891f51e3d8d02e7dd72e3165febaf88547`.
The source reconstruction and validation are retained in
[PR #417](https://github.com/jinzhezenggroup/vibeqc/pull/417). Its final formatted
library has distinct identity `ceafaa3df32bf05154e7532fb962c19b35c5d919603a9651b94274dcf17d0356`
and SHA-256 `6100ffe6cf4997935ff440b91e622e37dbb8b96207894291d8ff586071320d9d`;
only that final library supplies the direct acceptance below.

Each warm cell has seven interleaved, synchronized repeats per engine. Each
engine restores its own immutable post-cold density and receives untimed
priming. Cross-engine AO/density byte identity is not asserted. Legitimate
convergence branches remain distinct; these are ordinary converged latencies,
not fixed-work speedups. All seven paired numerical results are checked against
the original fixture gates, beyond the comparator's final-pair verdict.

## Warm results

Times are median seconds. F means energy plus complete analytic forces; E means
energy only. The failed row retains its raw timing without performance admission.

| Case / batch | Observable | VibeQC | Stock | All seven numerical pairs |
| --- | --- | ---: | ---: | --- |
| Water 96 / 1 | F | 0.121039 | 0.263416 | pass |
| Water 96 / 4 | F | 0.467436 | 1.057110 | **fail** |
| Water 192 / 1 | F | 0.166327 | 0.350230 | pass |
| Water 192 / 4 | F | 0.658095 | 1.400536 | pass |
| Water 384 / 1 | E | 0.297037 | 0.136630 | pass |
| Water 384 / 1 | F | 0.712707 | 0.616682 | pass |
| Water 768 / 1 | E | 1.015553 | 0.615638 | pass |
| Water 768 / 1 | F | 2.559992 | 2.138115 | pass |
| NH3 29 / 1 | F | 0.008801 | 0.236012 | pass |
| OH UHF 19 / 1 | F | 0.007782 | 0.243861 | pass |

The 192-AO stock/native ratios are 2.106 and 2.128. Their native relative median
absolute deviations are 0.361% and 0.094%; stock values are 0.145% and 0.117%.
These dispersion statistics describe this sample. `summary.json` retains every
case's branch distribution, tolerances, maximum error and metric diagnostics.
For example, 192/b1 uses three native updates versus one stock update; 768 force
stock branches vary across one, three and five updates. No favorable common
branch is selected or pooled into a fixed-work claim.

The 96/b4 maximum force difference is 3.131631759e-11 Eh/Bohr, above its unchanged
3e-11 gate. Independent tighter CPU DF diagnosis finds native and original stock
errors of at most 1.794522691e-11 and 2.467226423e-11, respectively. Tightening
stock's orbital-gradient convergence from 1e-9 to 1e-10 lowers its CPU-reference
error to about 3.06e-12. This identifies convergence sensitivity; it neither
replaces the original timings nor converts the failed admission into a pass.

## Cold and changed geometry

Job 9855 measures seven isolated-process pairs for 192/b1. Imports/driver startup
are untimed; molecular preparation, SCF and host-returned complete forces are
inside cold timing. Changed timing includes geometry reset/rebuild and starts
from each engine's immutable original-geometry density after equivalent priming.
The predeclared noise criterion is at most 3% relative MAD, with a 2% superiority
margin and a paired bootstrap interval. Both series meet the noise criterion.

| 192/b1 endpoint | VibeQC seconds | Stock seconds | Numerical result |
| --- | ---: | ---: | --- |
| Cold | 2.134694 | 1.395013 | pass |
| Changed geometry | 1.433309 | 0.473232 | **fail** |

The changed force discrepancy is 7.302685837e-10 versus the unchanged 5e-10 gate.
Its timing is unqualified. Warm gains therefore establish no geometry-optimization
or molecular-dynamics throughput claim. Other required cold/changed cells remain
unmeasured in this external campaign.

## Stock work and memory observations

The separate 192/b1 stock profile observes two logical J/K calls, one eigensolve,
one rebuilt CDERI and seven raw three-center evaluator blocks. The retained
factor is `[192, 17314]`, 26,594,304 bytes; normal gradient release is preserved.
It reproduces the clean stock result within 4.55e-13 Eh and 1.30e-13 Eh/Bohr.

CUPTI observes 243,036 H2D, 315,419 D2H and 72,864 D2D bytes. All 116 observed
device synchronizations are the component wrappers' two fences per call; they
are not production synchronization counts. Thread-local exclusive scopes,
runtime calls, exact transfers and source hashes are in `stock-diagnostics/`.
Kernel/API/host durations are never summed into endpoint time. Sampled process
peaks include profiler/wrapper overhead and do not satisfy unprofiled memory
acceptance. Complete per-cell stock/native work and memory coverage remains open.

## Final direct-SCF acceptance

Job 9860 uses the final qualified library and unchanged four direct gate points,
seven repeats and original reference settings. Both 96-AO batches fail the
3e-11 force gate; the largest retained paired error is about 5.73e-11. Both
192-AO batches fail native cold execution with `CUDA runtime error`, before a
usable timed series. Full tracebacks and missing-result verdicts are retained.
The 192/b1 error also reproduces in both frozen and final libraries under a
separate synchronous-launch diagnostic; that probe supplies no timing claim.
The underlying direct failure is unresolved. These gates are not waived.

## Acceptance mapping

| #5 criterion / #206 requirement | Current evidence and remaining work |
| --- | --- |
| Explicit reference energy/force tolerances | Most warm cells pass; 96/b4 warm and 192/b1 changed fail. Reconcile convergence/numerics without relaxing errors. |
| 192/b1 and 192/b4 DF force median no slower | Passed for ordinary warm latency on this pinned stack. |
| Bounded memory and no required full tensor beyond budget | #409 retains separate native constrained-memory evidence; complete external memory/transfer comparison remains open. No global Python inventory expansion is inferred. |
| Resident warm execution and per-item failure isolation | Native integration/neighbor tests are retained in #417; fresh batch convergence is retained here. This does not supply every missing per-cell resource diagnostic. |
| Direct gates remain unchanged | Unchanged and rerun; numerical and runtime failures remain open. |
| Reusable interfaces for future DFT/MP2 consumers | Existing Fock/source interfaces and #417 compatibility tests are retained. This campaign does not certify new future-consumer coverage. |
| Larger DF, non-water and UHF coverage | 384/768, NH3 and OH UHF are retained; larger cases are slower than stock. Smaller/UHF energy-only coverage remains open. |
| Practical unequal and low-memory external comparison | #409's 24/116 and 96/464 internal comparisons pass; 384/1856 fails. Matched external endpoints remain open. |
| Complete cold/changed/warm and operator ledger | Only the stated subset is complete; missing cells are listed in `manifest.json`. |

#5's existing closed GitHub state is not evidence that these current gates pass.
#206 remains open. #394/#404 gains are already included, and no rejected Gram
candidate is counted again or combined arithmetically.

## Evidence and reproduction

`manifest.json` binds retained records to original hashes and lists open work.
Raw arrays, failed records and stock rank qualifications remain in their original
campaign directories. `reproduction/` preserves exact measured scripts as text;
their workstation paths are provenance. For fresh runs, use the repository's
`compare_gpu4pyscf_batch.py` with the recorded fixture, convergence, gate and
control arguments, within finite Slurm allocations. Do not build or profile
concurrently with clean timing. Final direct reproduction uses the existing
`real_molecule_gate.py --density-fitting none --repeats 7` matrix.

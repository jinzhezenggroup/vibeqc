# Exact packed DF values: scoped Phase-A qualification

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

This retains the implementation and completed measurements for #409.
**Keep explicit packed selection for measured domains; unset/auto stays dense.**
The useful domains are 192-AO warm forces and the measured 768-AO/12-GiB
endpoint. Large warm and changed-geometry regressions remain visible, and the
failed 384/1856 unequal cell is excluded. This establishes no general
superiority over stock GPU4PySCF (#206).
All timings below are seconds on the RTX 5090, in finite Slurm jobs.

## Completed clean endpoints

Each row contains seven interleaved dense/packed pairs. Clean calls have no
tracing or memory sampling; intrusive diagnostics are separate. Both policies
start from the same frozen density for warm/changed cells and use normal
convergence. Every sample in the completed rows below passes `1e-9 Eh / 1e-8 Eh/Bohr` against
independent references. These large-case gates do not relax stricter fixtures.

| Endpoint | Dense median | Packed median | Dense/packed SCF updates |
| --- | ---: | ---: | --- |
| 96 warm forces | 0.108850834 | 0.105526933 | 2 / 2 |
| 192 warm forces | 0.165658090 | 0.130553789 | 3 / 3 |
| 384 warm forces | 0.693052435 | 1.820831211 | 3 / 3 |
| 384 warm energy | 0.284203221 | 0.269924065 | 3 / 3 |
| 768 warm forces | 2.548116406 | 3.216367704 | 3 / 6 |
| 768 warm energy | 1.013974837 | 1.676097540 | 3 / 6 |
| 384 cold forces | 10.135832240 | 11.948631649 | 20 / 18 |
| 384 changed-geometry forces | 9.606932915 | 11.256079176 | 12 / 12 |
| 768 cold forces | 85.685922099 | 82.768054998 | 23 / 24 |
| 768 changed-geometry forces | 86.357965270 | 191.511216193 | 9 / 9 |
| 24/116 unequal warm forces | 0.019747868 | 0.018764724 | 2 / 2 |
| 96/464 unequal warm forces | 0.568663857 | 0.552863172 | 2 / 2 |
| 768 / 12-GiB warm forces | 162.023276958 | 65.028559662 | 5 / 6 |

Changed SCF branches are ordinary-latency results, not iteration-matched
comparisons. The 192-AO force gain is about 21.2%; packed force response removes
56,623,104 raw host-upload bytes and associated gathering/synchronization while
retaining the same AO/density/metric work. At 384 AO, exact bounded packed raw
response loses the dense path's borrowed full-buffer optimization and dominates
the force regression. At 768 AO, J/K/eigensystem work increases from 4 to 7
logical calls with the changed SCF branch. The single final physical Fock remains
present; final eigensolves/density corrections/rejections remain zero.

`warm/` and `rebuild/` retain original raw numerical samples. The
`phasea-summary.json` contains paired bootstrap intervals, work distributions
and independent-reference errors. `clean-warm/work-reconciliation.json` and
the per-cell diagnostics reconcile eager operations with **observed executed
graph replays**; graph construction is never counted as work. These counts are
logical operations, not kernel counts or estimates of GPU time.

`storage-dataflow.json` derives simultaneous packed A/B and three shared scratch
capacities from the observed shapes. At 768 AO these sum to 5,591,531,520 bytes,
before other source/metric/SCF/library allocations. It distinguishes directly
observed eager counters from logical J FLOPs derived using the observed replay
count: the warm graph was cached before tracing, so there are no in-range graph
construction counters to multiply. Transfer counters remain scoped to individual
operations because nested semantic counters overlap. Hardware DRAM traffic and
raw-integral recurrence FLOPs remain explicitly unmeasured. The separately
retained CUPTI observations now supply exact CUDA transfer bytes, as described
below.

The 768-AO changed-geometry clean series also completed all seven pairs:
86.357965270 -> 191.511216193 seconds, with nine updates in each arm. Its
maximum energy/force differences are 5.87e-11 Eh / 1.78e-10 Eh/Bohr. Job 9845
then timed out before completing the separate diagnostic pass. The original
clean file and terminal scheduler record are retained under `partial/` and
`campaigns/`. Job 9851 completed only the separate diagnostics. The composed
`rebuild/768-changed.json` verifies the original binary, geometry, reference,
checkpoint and frozen-density identities; it repeats or pools no clean samples.

Both diagnostic arms accept the occupied seed and execute 16 J/K calls: ten
occupied K and six dense K, with six final density corrections. Response takes
3.495 seconds for dense and 112.362 seconds for packed. Dense borrows full all-Q
scratch, while packed uses the bounded raw fallback: 77 auxiliary blocks,
118,272 AO response products versus 1,536, and 35,326,918,656 unpacked logical
elements. The packed fallback also lies outside the automatic 768-AO borrowed
shell/BLAS domain and executes scalar products and generic derivative consumption.
`rebuild/768-changed-response-attribution.json` binds these observations to the
separate diagnostic and relevant source. This identifies executed work and
expensive scopes; it does not measure a counterfactual optimization saving.

The unequal cases use unmodified cc-pVDZ/cc-pVDZ-JKFIT bases. At 384/1856 and
an 8-GiB total DF allowance, the dense cold preflight fails the unchanged force
gate: 1.300395833e-8 exceeds 1e-8 Eh/Bohr (energy error 3.98e-10 Eh). Both
metric ranks are 1856. The campaign stops before packed or warm timing, so this
larger case is unqualified and establishes no representation speed comparison.
`failed/unequal/` retains its full numerical record, including the failure.
Independent CPU PySCF/libcint diagnosis in job 9854 confirms this is a native
accuracy discrepancy: the dense force differs from the tighter CPU oracle by
1.297054950e-8 Eh/Bohr, while the original stock reference differs by only
1.219315759e-10. The original failure remains unchanged. The qualified unequal
domain is limited to the passing 24/116 and 96/464 cells; 384/1856 is explicitly
excluded pending a separate numerical fix. No packed timing exists for it.

## Constrained memory: clean endpoints and separate diagnostics

Job 9848 completes seven interleaved clean pairs at 768 AO with the same
12-GiB total DF allowance. Dense and packed complete-force medians are
162.023276958 and 65.028559662 seconds: packed uses 59.86% less time. The
paired-bootstrap packed/dense ratio interval is [0.401023237, 0.401756930].
Every sample passes the unchanged numerical gates; maximum energy/force
errors are 9.05e-11 Eh / 1.99e-10 Eh/Bohr. SCF updates are 5 versus 6, so this
is ordinary converged latency, with different work. Raw samples and the
terminal successful scheduler record are retained.

Separate job 9841 measures the intrusive components and unprofiled process
memory under the same allowance. Both arms retain resident B; no
streamed-to-resident B crossover is established.

| Observation | Dense | Packed |
| --- | ---: | ---: |
| Intrusive complete forces, s | 162.144786 | 65.127802 |
| SCF updates | 5 | 6 |
| Native observed peak, bytes | 12,133,970,329 | 5,892,268,465 |
| Sampled warm process device peak, bytes | 18,647,875,584 | 12,431,917,056 |
| Sampled warm process host peak, bytes | 6,448,332,800 | 7,558,451,200 |

The native/device peaks decrease 51.4%/33.3%, but sampled host peak increases
17.2%. Both native ledgers close to zero. The 32-GiB observation ceiling covers
owned native buffers, not driver/library allocations, and is not large-domain
Python inventory admission or a whole-process memory guarantee. Dense response
generates 1,626 raw blocks and takes about 94.72 s; packed response reuses raw
and final U and takes about 1.447 s. Both spend about 58.56 s exporting bounded
one-electron derivatives. The clean timing above is separate from these intrusive observations.
The useful benefit is limited to this constrained endpoint; broader automatic
selection remains unqualified.

## CUDA dataflow

`nsys-dataflow-summary.json` and `dataflow/` retain six completed warm profiles
from job 9850, with exact SQLite/CUPTI transfer bytes, kernel launch shapes,
register/shared/local declarations, API counts and source hashes. Explicit
CUDA Graph **node** tracing includes work inside replayed graphs. Device kernel
busy intervals are unioned; kernel/API/host durations are never added together.
These profiles use the frozen measured library and pass the numerical gates.

At 192 AO, observed H2D bytes fall from 59,302,772 to 2,679,668. The difference
is exactly the independently recorded 56,623,104 raw-upload bytes. Kernel launches
increase from 1,659 to 1,859 while both retain three updates. At 384 AO, H2D
bytes remain 10,669,236 while launches increase from 1,189 to 20,106, consistent
with the bounded packed raw-response cost. At 768 AO, both transfer 42,572,084
H2D bytes and 4,718,592 D2D bytes; their three versus six updates remain distinct.
Complete D2H counts are retained alongside the other directions.

Native progress tracing inserts one event fence per eager region and one more
per eager operation. The account separates these from observed API counts:
all 9,796 event synchronizations in the profiled 384-AO packed call come from
those tracing fences. Remaining synchronization counts are still intrusive
observations, not certified clean-production counts. Sampled process peaks under
Nsight include profiler overhead and do not replace the unprofiled capacity
acceptance observations above.

Jobs 9856/9857 complete the two unequal warm and eight cold profiles, bringing
the total to sixteen passing profiles. For 96/464 warm forces, H2D falls from
34,902,276 to 692,484 bytes, while launches rise from 3,328 to 3,819. Cold
768-AO H2D falls from 3,863,338,950 to 93,193,158 bytes and D2H from
7,107,709,859 to 2,987,702,524 bytes; updates remain 23 versus 24. All exact
per-case directions and shapes are retained. The later profiles explicitly
disable optional CUDA event-completion tracing, while retaining CUDA API and
graph-node activity. Job 9850 used its original default event tracing. These
control differences are preserved; profiled timings are not pooled.

## Final qualification and limits

Native density-fitting/occupied-response tests and memcheck/initcheck/synccheck
passed in jobs 9833–9835. Stage-2 job 9836 passed both native suites and 15
molecular/Fock GPU cases; 15 CPU resource cases also passed. Molecular memcheck
job 9838 passed 15 cases with zero errors/leaks; two failed-neighbor cases were
initially skipped by a missing tier flag, then passed separately in job 9839.
`qualification/` retains observed summaries and original log hashes without
publishing routine logs. Stage-1 and stage-2 library identities are distinct.

Job 9859 rebuilt the publishing source after all clean campaigns finished. Both
native suites, 17 molecular/failed-neighbor cases, 15 resource cases, density and
response memcheck, response initcheck/synccheck and the portable warm wrapper
pass. The final library's independent seven-pair 192-AO check gives
0.165593741 -> 0.130891700 seconds, three updates in both arms, with all energy
and force gates passing. It is a separate fresh-seed series, never pooled with
the frozen measurements above.

The cached incremental rebuild took 8.68 seconds and peaked at 367,900 KiB host
RSS. This is not a clean-build or baseline compilation-cost comparison. Static
sm_120 declarations report 40 registers and 21,632 shared bytes for either packed
projection layout, with zero stack/local bytes. Actual launches and declarations
are also retained in the CUPTI accounts. Hardware DRAM traffic and raw-integral
recurrence FLOPs remain unmeasured and are not inferred from tensor sizes.

The retention decision is explicit opt-in, supported by the measured warm and
constrained endpoints. It makes no automatic-selection or general capacity
crossover claim. The two incomplete manifest entries preserve the failed larger
unequal campaign and its preflight; that domain requires a separate numerical
fix. Optional screening and external #206 acceptance remain separate work.

Practical unequal auxiliary coverage uses unmodified basis definitions. The
def2 universal fitting basis has unsupported g shells for O/N/C on this backend;
silently truncating those shells is not an acceptable comparison. Initial stock
metadata harness failures were corrected without changing numerical gates:
the active GPU CDERI is a list, and stock gradients release it. Rank is observed
before the gradient, without artificially retaining its buffers.

The Python global resource inventory remains limited to its supported
admission domain (at most 16 orbital/128 auxiliary AOs); the shape query and
large-case observation ledger do not extend that domain.

## Identities and reproduction

The frozen measured source starts at
`25e8efebb069e6067ba3649f220c37e844ecc8f6` plus
`reproduction/source.patch`. Its native identity is
`2d59ce1e7d001a6e9d0a28b90f44e6fcbcf8e752a6682b1d80a98738c72b3133`;
the measured library SHA-256 is
`c1d4c033b73e3b42da58acab0b67c2891f51e3d8d02e7dd72e3165febaf88547`.
Build flags and the generated identity header are retained beside that patch.
The publishing source was subsequently clang-formatted; frozen measurements
must not be relabeled with a new binary identity.
`qualification/format-verification.json` verifies that all 31 files in the
pre-format snapshot produce the current bytes through clang-format 23.1.1.
The final rebuilt library is separately validated in
`qualification/final-manifest.json`, with native identity
`ceafaa3df32bf05154e7532fb962c19b35c5d919603a9651b94274dcf17d0356`
and library SHA-256
`6100ffe6cf4997935ff440b91e622e37dbb8b96207894291d8ff586071320d9d`.

`manifest.json` records retained/original hashes and explicitly lists incomplete
cells. The collector's `--partial` permits a draft only; it is not a passing
qualification flag. Intrusive original trace hashes remain in numerical
records. Full logs, traces, libraries and temporary checkpoints are transient.
Archived workstation paths are provenance, not portable launch commands.

Use the [experiment reproduction guide](../../experiments/issue409-packed-values/README.md)
and its `run_endpoints.py` for a fresh warm matrix. Without supplied checkpoints,
the runner freezes a fresh post-cold density and records its own branch identity.
Exact historical scripts are retained as `reproduction/measured-*.txt`, preserving
their bytes through format hooks. The `.hpp.txt` identity snapshot likewise
preserves its original header bytes. The source patch is applied in a separate
checkout at the pinned base before rebuilding, never over an existing dirty tree.

`projection/` retains fixed-input plans, raw samples and candidate summaries;
`producer/` retains independent raw/J/truncated-metric/unequal-size checks.
Neither isolated projection wins nor smaller factor arrays qualify complete
SCF/force latency. The [Agent Note](../../../.agents/notes/implemented/performance/2026-09-17-packed-df-retention.md)
records the exact representation, fallback boundaries and retirement conditions.

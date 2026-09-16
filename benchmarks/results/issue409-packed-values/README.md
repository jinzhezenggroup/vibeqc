# Exact packed DF values: partial Phase-A evidence

This draft retains the experimental implementation and completed measurements
for #409. **Qualification is incomplete; unset/auto still selects dense.**
It does not close #409 or establish superiority over stock GPU4PySCF (#206).
All timings below are seconds on the RTX 5090, in finite Slurm jobs.

## Completed clean endpoints

Each row contains seven interleaved dense/packed pairs. Clean calls have no
tracing or memory sampling; intrusive diagnostics are separate. Both policies
start from the same frozen density for warm/changed cells and use normal
convergence. Every retained sample passes `1e-9 Eh / 1e-8 Eh/Bohr` against
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

## Constrained memory: diagnostic evidence only

Job 9841 measures 768 AO with the same 12-GiB total DF allowance. Both arms
retain resident B, so this is not a streamed-to-resident crossover.

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
one-electron derivatives. **Seven clean constrained pairs remain required.**

## Qualification and remaining work

Native density-fitting/occupied-response tests and memcheck/initcheck/synccheck
passed in jobs 9833–9835. Stage-2 job 9836 passed both native suites and 15
molecular/Fock GPU cases; 15 CPU resource cases also passed. Molecular memcheck
job 9838 passed 15 cases with zero errors/leaks; two failed-neighbor cases were
initially skipped by a missing tier flag, then passed separately in job 9839.
`qualification/` retains observed summaries and original log hashes without
publishing routine logs. Stage-1 and stage-2 library identities are distinct.

The draft still requires:

- 768-AO cold/changed endpoints and their independent references/work counts;
- complete-shell cc-pVDZ/cc-pVDZ-JKFIT unequal-auxiliary endpoints;
- seven clean constrained pairs and a final domain/negative-result decision;
- the remaining complete traffic/component/resource and compilation audit;
- a rebuild and required validation of the subsequently formatted native source;
- GPU validation of the new portable warm wrapper.

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
The publishing source was subsequently clang-formatted and has not yet been
rebuilt; frozen measurements must not be relabeled with a new binary identity.

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
SCF/force latency. The [Agent Note](../../../.agents/notes/proposed/2026-09-17-packed-df-values.md)
records the exact representation, fallback boundaries and retirement conditions.

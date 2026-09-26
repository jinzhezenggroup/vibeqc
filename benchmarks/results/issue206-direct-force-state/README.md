# Direct CUDA HF force-state correction

**Final external acceptance remains FAILED.** The final-source campaign retains
all seven pairs per point. Both 96-AO points exceed the original 3e-11
Eh/bohr force gate. The scientific correction is a local/draft candidate;
this evidence keeps #206 open and establishes no general performance lead.

The native change checks physical stationarity within the existing direct-force
SCF iteration budget, constructs a common final determinant for energy/forces,
rebuilds its physical Fock and revalidates its residual. The Pulay weight uses
P F(P) P / 2 for RHF and P_sigma F_sigma P_sigma for UHF. Coarse mixed items
retain their earlier stop; exact neighbors and mandatory FP64 refinement apply
the physical criterion. Production uses no CPU reference products.

## Historical record storage

The original 66 records (3,692,876 bytes) remain byte-exact in existing Git
history at `5ff708e3d51212f6baaa12efef346ebc172623ef`. 55 reports, source snapshots
and work journals have left the current checkout; no Git history was rewritten,
and no Release, tag or external copy was created. The
[history snapshot](history-snapshot.manifest.json) records every path, size and
SHA-256. All 66 entries were restored and verified before checkout removal.

Integration must use a **merge commit**, not squash or rebase: the pinned
snapshot revision must remain an ancestor of the default branch after PR-branch
cleanup. Repository settings already allow merge commits; no setting change is
needed. The checkout compaction does not imply a reduction in Git-history size.

Each historical matrix retains `retained-samples.json`: all seven paired scalar
errors, raw timings, convergence states, settings, input hashes, build identities and
original failed verdicts. Full force arrays and journals are recovered with:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/issue206-direct-force-state/history-snapshot.manifest.json \
  --output .artifacts/pr427-historical-snapshot
```

Restore the complete snapshot before inspecting the original `manifest.json` or
replaying its historical scripts. Later qualification runs are separate records;
this storage change neither changes historical verdicts nor qualifies the
post-merge build.

## Final build and unchanged external gates

- Source identity: `7951fbce9d6bb0d39332e4a40c9a3919cfad0869e7f6fb360951c609dc33daa7`.
- Library SHA-256: `e2e92b07f1fcfc681a88277710cbac68305b89626bc78464873853ebcadb4d45`.
- Build base, exact source patch, native test identity and work journal are in
  [candidate/](candidate/); binaries and routine logs remain transient.
- Native energy/density/screening tolerances: 1e-12/1e-10/1e-14; max iterations 100.
- Stock gradient tolerances remain 1e-9 at 96 AO and 1e-8 at 192 AO.
- Energy/force error gates remain 3e-11/3e-11 at 96 AO and 1e-10/5e-10 at 192 AO.
- Seven interleaved samples per engine; Slurm job 9890 on RTX 5090, Release AOT
  FP64. No build or profiler ran during these timings. All added final work is
  included. Values below are complete ordinary warm endpoint medians in seconds.

| AO/batch | Native | Stock | Max paired force error | Verdict |
| --- | ---: | ---: | ---: | --- |
| 96/1 | 0.204561 | 1.694131 | 4.242826e-11 | FAIL |
| 96/4 | 0.457159 | 6.466334 | 3.705813e-11 | FAIL |
| 192/1 | 0.491018 | 2.222034 | 2.308518e-10 | PASS |
| 192/4 | 1.633100 | 7.800139 | 2.352990e-10 | PASS |

Iteration branches vary. The benchmark retains matched subsets separately,
but numerical acceptance inspects every pair, including the failing one. Stock
uses sequential single-system objects; these timings do not establish equal
operator work, throughput for a native stock batch interface, or general superiority.

## Independent diagnosis and validation

The failing 96/b1 pair has native/stock force errors of 3.28845e-12/4.08849e-11
against the exact-geometry tight CPU oracle. The stock orbital-gradient norm
3.82957e-10 satisfies its original 1e-9 control, yet its force error exceeds the
paired force gate. Every native repeat stays within 3.30e-12 of that oracle.
The 96/b4 point also fails, with paired maximum 3.70581e-11; its native/stock
maxima against CPU are 4.24476e-12/3.56404e-11. Both 192-AO points pass.
[The decomposition](https://github.com/njzjz-bot/vibeqc/blob/5ff708e3d51212f6baaa12efef346ebc172623ef/benchmarks/results/issue206-direct-force-state/diagnosis/final-direct-pairs.json) preserves all fourteen samples
and the failed verdict; it is not a replacement acceptance run.

The final binary passes 49 native tests, a retained independent-CPU OH public
regression, and cold/frozen-warm/changed-geometry probes for exact two-AO
RHF/UHF, 96-AO RHF and 19-AO OH/UHF. Final OH force errors are below 6.12e-11
against its unchanged 1e-9 gate. Independent CPU F(P) at the actual force
states gives max commutator <=6.24e-11 against 1e-10, and reproduces native
forces within 4.10e-14. The 106 host structure/ownership checks and repository
hooks also pass. These checks are separate from clean timings (job 9889).

Public iteration counts describe iterative updates. Each tested force item
additionally executes one canonical density projection, one physical Fock build,
four residual products and two PFP products per spin, plus a four-byte work
counter download at the existing fence. The separate intrusive qualification
journal counts 97 added projections/Fock builds, 97 final residual checks,
zero final rejections and 718 iterative updates across its calls. Energy-only
calls assert zero additional force-operation records. No component speedup
or operator elimination is claimed.

The four compressed `integration-e4f2f67/*.progress.jsonl.gz` streams contain
only per-call execution progress from the final integration matrix. Their exact
bytes now live in the checksum-bound [`retention-488`](../retention-488/README.md)
Git-history snapshot. The retained integration logs/XML, numerical records and
qualification summaries preserve the source/build identity and pass/failure
conclusions without treating progress streams as accepted benchmark evidence.

## Earlier attempts and reproduction

[history/](history/) retains the PFP-only paired failure and the fixed one-update
OH failure. Two incomplete one-update runner attempts are preserved and explained
in [manifest.json](manifest.json): one mistook generic trace-file growth for
force work; another omitted changed coordinates from an energy-only call.
[preliminary/](preliminary/) retains job 9888, which passed its 28 pairs on an
earlier source. Those samples are never substituted for final-source job 9890.

The manifest indexes byte-exact input/output records and historical scripts.
The scripts under [reproduction/](https://github.com/njzjz-bot/vibeqc/tree/5ff708e3d51212f6baaa12efef346ebc172623ef/benchmarks/results/issue206-direct-force-state/reproduction/) are saved as `.py.txt` to preserve
measured bytes. Restore their `.artifacts/issue206/` paths and original relative
inputs, or adapt only those paths to this archive. Build with the retained
configuration, use the recorded source patch, run native/independent correctness
checks first, then run the clean matrix in a fresh finite Slurm allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:15:00 python .artifacts/issue206/run-direct-force-final.py
```

Preserve Slurm device visibility. Do not combine clean timings with builds,
profiling or correctness instrumentation. The runners preserve nonzero gate
exits. DF accuracy, stock reference convergence, cold/changed/energy-only and
small/UHF external coverage, unprofiled memory/transfer accounting, unequal
auxiliary bases and constrained-memory comparisons remain #206 work.

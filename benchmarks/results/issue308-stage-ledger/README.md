# Issue #308 follow-up stage ledger

This increment follows the September 14 analysis in
[issue comment 5661246967](https://github.com/jinzhezenggroup/vibeqc/issues/308#issuecomment-5661246967).
It adds crash-visible phase observations and bounded fixed-density experiments.
It does not establish the full 768-AO VibeQC energy/force endpoint or close
#308/#310/#311/#206.

`historical-followup.zip` retains selected original evidence from Slurm jobs
9531/9534/9535/9536/9537: exact JSON measurements, reproduction sources and
geometry, original hashes, and the first 24-pair sample of the complete raw
three-center generation. Its manifest preserves all original file identities,
source qualifications, and the selected-member restoration audit. Every member
was reread and compared byte-for-byte. Bulk exploratory arrays, binaries and
routine logs remain in the original debug archive, copied from `/tmp` into the
workspace's ignored `.artifacts/issue308-historical/` directory; its path and
SHA-256 are recorded, and reproduction does not depend on that debugging copy.

The complete raw generation in job 9536 took 103.113819448 seconds and generated
452,984,832 values. Only its first 18,432 values were retained and compared to
the earlier row probe. This is neither a complete independent tensor-validation
result nor timing of metric transformation, SCF or forces. The temporary
register interposition in 9531 remains an unpromoted diagnostic. Its four
complete 589,824-value row outputs and the 9535 row-zero output were rechecked
for exact equality during retention.

## Unified build and fixed-density diagnosis

The measurements in `stage-summary.json` and `stage-evidence.zip` used commit
`4d515cf800bf75606098c968ba59ac8c3726be5c`, native source identity
`0794d1cd4ea24b0a28130ae03c7f058eda8114497d2c5704035fe20fed1d36c8`, and library
SHA-256 `30010c925eb3621fea5d93861838739b1d9d25d365774cf8aad786fe8d71600b`.
Release CUDA 12.9.1, sm_120, stable-shards, fast compile OFF was built from
the unified source containing #323–#328. The exact path, driver, environment,
input identities and per-run outcomes are retained. The measured library was
frozen before subsequent instrumentation changes; it is not the later PR-head
build. Jobs 9538/9539 used the Slurm-assigned RTX 5090.

All following seconds are **diagnostic**, with event/progress tracing enabled.
The repeated J/K sample is the second call on the same prepared plan. Setup
includes source and metric preparation; its raw/GEMM subphases must not be added
to setup again. These are not clean performance or GPU4PySCF parity results.

| AOs | Resident setup | Raw generation | Metric GEMM | Repeated dense K | Repeated occupied K |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.1907 | 0.0825 | 0.00030 | 0.00060 | 0.00039 |
| 192 | 0.8975 | 0.7744 | 0.00175 | 0.00365 | 0.00149 |
| 384 | 8.1507 | 7.9866 | 0.02553 | 0.05564 | 0.01824 |
| 768 | 104.2271 | 103.5875 | 0.40778 | 0.82903 | 0.20976 |

At 768 AOs, source setup took 0.1131 s, metric factorization 0.1004 s, and
repeated J 0.00566 s. Sampled process GPU high water was 14,514 MiB. All full
resident J/K outputs passed independent PySCF comparisons at the existing
1e-9 gate: maximum J error 4.17e-12 and K errors below 9.9e-13. Orbital order,
spherical convention, overlap, imported occupations and full metric rank were
qualified before fixed-D measurements. The PySCF DF metric threshold is 1e-10.

Streamed 192-AO K actually uses all 192 AO rows and output width 28, regenerating
seven equivalent raw tensors per K. J takes 1.44–1.50 s and dense/occupied K
about 5.0 s, with full independent J/K errors below 1.4e-12. The 384-AO probe
uses width 7 and timed out after 180 s during repeated generation.

For 768 AOs, nominal 8192 AO pairs × 128 auxiliaries becomes **768 rows × Q=1**.
Each fused panel declares 452,984,832 logical `(pair,Q,P)` source evaluations.
Three panels completed in 33.6600, 33.5138 and 33.5193 s; the fourth began
before the 120-second timeout. Full K did not finish and has no correctness
claim. Multiplying the raw-tensor time by 768 is not a measured runtime.
The original invalid 96-AO request (128 auxiliaries exceeds naux) is retained;
the corrected 96-auxiliary request passed independently in job 9539.

## Completed 768-AO native energy diagnosis

`scf-summary.json` and `scf-evidence.zip` retain job 9540: the original core
guess and an independently converged PySCF seed under the same strict FP64
RHF settings, max_iterations=100, physical export, and a 28-GiB **DF subbudget**.
These are native diagnostic solves, not public global-resource acceptance.
Preparation is included in the SCF interval; the outer probe also performs
independent seed/overlap qualification. Seeded timing is not cold-start timing.

| Initial density | Native SCF seconds | Compact iterations | DIIS retry iterations | Final physical Fock builds |
| --- | ---: | ---: | ---: | ---: |
| Independent seed | 111.1741 | 3 | 0 | 1 |
| Original core guess | 290.5779 | 100 | 48 | 3 |

The cold solve's reported `iterations=48` omits the preceding **100 compact
iterations**. Compact SCF took 86.5577 s, retry 89.9437 s, and finalization with
export 8.1857 s. Seeded compact SCF took 2.6804 s and finalization 3.3040 s.
Each solve attempted graph construction once, executed ordinary iterations,
and performed no host graph replay. Both recorded zero CPU-reference
eigensolves. The remaining resident tail therefore warrants work on iteration
orchestration and its original-seed restart; the measured metric GEMM is small.
This does not retroactively identify the unobserved tail of historical job 9521.

Both energies agree with retained PySCF `-2438.354141054296 Ha` within
3.8e-11 Ha. Full exported D/S/H/F/C/epsilon checks give maximum eigen residual
5.7e-13, S-orthogonality 3.1e-14, electron trace error 4.6e-13 (320 electrons),
metric idempotency 2.8e-14 and physical commutator 2.2e-13. D differs from
PySCF by at most 2.3e-12 and F from independent J/K by at most 4.8e-12.
Energy reconstruction and canonical-density consistency also passed.
**W is derived from exported C/epsilon; native force W and full forces are
not yet validated.** Full transient exports remain in `.artifacts/`; the
reviewed archive contains all comparison errors, exact journals, first-24-row
state samples, every orbital energy, and reproduction/validation code.

The SCF launcher recorded library path and native source identity, matching
the frozen fixed-D library. It did not take a separate launch-time library
hash or sample GPU memory. These limitations are explicit in the summary;
the fixed-D GPU peak must not be attributed to SCF. All archived members were
restored byte-for-byte and their hashes are listed in the summaries.

The stable fixed-D interface is documented in
[the component-trace contract](../../../docs/df_component_trace.md).
For SCF reproduction, extract `scf-evidence.zip`, compile its
`reproduction/scf-seed-probe.cpp` with the same include/link flags as
`benchmarks/df_stage_probe.cpp`, prepare the 768-AO fixture using
`python -m benchmarks.issue308_stage_probe --prepare --case water-32mer-4s4-def2-svp-spherical --output INPUT --checkpoint CHECKPOINT`, and adapt only the absolute paths
in the retained `run-scf-seeds.sh`. Run it inside a finite Slurm allocation;
restore the independent checkpoint from `../issue308-large-diagnostics/reference.zip`.
The validation script expects the same run/input directory layout as the
retained launcher. The archive includes the measured CMake configuration.

Property/provider-aware budgets, streamed-K raw reuse, bounded retained-B
scratch, SCF orchestration, complete analytic forces and the #206 acceptance
matrix remain outstanding. This evidence does not close #308/#310/#311/#206.

## Archive storage correction

`scf-evidence.zip` was removed from the current tree when restoring the hard
1 MiB file limit. Its exact bytes remain in commit `daa2da0867877c94c40f379ffe1f3db6e3036ef8`; the
[storage migration](../retention-size-limit/migration.json) pins its SHA-256
and size. Existing numerical conclusions and measured identities are unchanged.
Restore the historical archive to an ignored working directory with:

```bash
python tools/restore_retained_evidence.py benchmarks/results/issue308-stage-ledger/scf-evidence.zip
```

Archive restoration is only needed for historical raw-run inspection. New runs
keep full logs, profiles and retries outside Git.

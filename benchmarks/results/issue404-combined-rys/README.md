> **Checkout retention note (2026-09-25):** bulky raw campaign members were moved out of the normal checkout and remain byte-for-byte recoverable from existing Git history. See [the bulk retention manifest](../retention-2026-09-25/bulk.manifest.json) and use `tools/restore_retained_evidence.py` before following links or reproduction steps that require archived members.

# Combined low-angular Rys endpoint qualification (#404)

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

The full 42-candidate campaign in [#394](../issue394-batch-rys/README.md)
selects Rys/compact for 001/002/100/200 in addition to the existing 000 choice;
101/110 retain polynomial/compact. This bundle qualifies that combined mapping
against the post-#411 baseline. Automatic selection remains limited to sm_120,
384/768 AO and equal auxiliary dimension. No screening or precision change is
part of this comparison.

## Decision

| AO | Baseline complete force median (s) | Candidate (s) | Saving | Paired bootstrap 95% ratio interval |
| --- | ---: | ---: | ---: | --- |
| 384 | 0.709223337 | 0.691501730 | 2.50% | [0.95643, 0.98063] |
| 768 | 2.618872429 | 2.549564063 | 2.65% | [0.96426, 0.98175] |

The original 3% minimum gain in [protocol.json](protocol.json) was not met.
The user explicitly accepted the observed benefit; [decision.json](decision.json)
preserves that separate decision. The original gate is still reported as failed.
All ten target-size pairs improve, the paired stability requirement passes,
and unchanged independent numerical and work gates pass. Five pairs give
limited statistical resolution; bootstrap intervals are not a fresh replication.

Energy-only baseline/candidate medians are 0.269720915/0.269717350 s at 384 AO
and 1.014230638/1.013041834 s at 768 AO. The 96/192 regression runs also satisfy
the 3% energy regression margin. Explicit candidate selection at small sizes
is numerical/diagnostic evidence and does not promote a broader automatic domain.

## Scientific and execution controls

Slurm job 9818 executes five interleaved clean repeats per arm, alternating
order, after all compilation finishes. Every transition receives an untimed
prime. Both arms use one prepared state and the same archived density with
warm-start updates disabled. Initialization and checkpoint validation are
reported separately and excluded from the warm endpoint. A separate diagnostic
pass follows all clean repetitions for each workload/observable.

All workloads use RHF, spherical def2-SVP, the same auxiliary basis, FP64,
metric threshold 1e-10, energy/density tolerances 1e-12/1e-10, batch one,
normal convergence/final-state checks, device final validation and existing
final-projection reuse. Screening is off. Matched archived GPU4PySCF results
provide independent energy/complete-force references; this is not a fresh
comparison of external execution times.

Across all retained clean and diagnostic records, maximum energy error is
1.501e-11 Eh and full-force error is 1.620e-10 Eh/Bohr, below 1e-9/1e-8 gates.
Both target arms execute three converged SCF updates, four J and four K builds.
The 384 graph path executes three eigensolves. The 768 occupied path executes
four including imported-density factorization. Both accept the final state
without correction, candidate rejection or warm fallback. Final commutators
are 8.093e-13 and 1.402e-12, with identical residuals across arms.

`work.json` retains observed scope counts, selected progress observations,
inclusive component times and hashes of the complete local diagnostic traces.
Captured iterations are counted from completed device iterations and the
captured iteration body in `src/scf/cuda/df_rhf_scf.cpp`; host graph launches
alone are not an operator count. Non-graph operations come from observed
runtime scopes, including the seed factorization and final physical build.

## Components and resources

Diagnostic derivative-contraction totals include every panel:

| AO | Panels | Baseline (ms) | Candidate (ms) | Complete host force stage, baseline/candidate (s) |
| --- | ---: | ---: | ---: | --- |
| 384 | 1 | 191.745 | 174.863 | 0.462361 / 0.446861 |
| 768 | 13 | 1337.204 | 1272.834 | 1.712672 / 1.652259 |

Per-class packet totals, register/shared-memory counters, transfers, shell
work conservation and scratch sizes are retained in each endpoint JSON and
`summary.json`. The existing 000 Rys choice is held fixed and contributes no
claimed new gain. The 768 final projection is reused; derivative panel count
and scientific work are unchanged. Intrusive component times are explanatory
and must not be substituted for clean endpoint times.

The campaign library grows from 164,009,448 to 164,333,752 bytes: 324,304 bytes
(0.20%). Both arms report identical live process device residency after the
diagnostic endpoint: 8,438,939,648 bytes at 384 and 21,296,578,560 bytes at 768.
These NVIDIA process readings include context/modules and retained allocations;
they are not allocator peaks. `control-resources/` separately records sampled
process peaks and pinned-baseline equivalence controls. Its execution is
intrusive, and **all** its timings are excluded from the clean campaign.
At 768 AO both standalone processes reach the same sampled device peak of
24,572,329,984 bytes including initialization. The 384 baseline/campaign peaks
are 8,659,140,608/8,573,157,376 bytes; these sampled lower bounds do not establish
an allocation reduction. Standalone baseline and campaign `auto` energies
match exactly, complete-force differences are below 8e-14 Eh/Bohr, and frozen
density, selected mathematics and executed operation counts match.

## Identity and reproduction

The measured campaign source is `490d4ba0266d4161b495ad5808fba10a5c843625`;
the pinned standalone baseline is `796ce85a607dfde85bf6bafd20575d7c538fdad1`.
`campaign-build.json` and `baseline-build.json` bind source/library identities,
generated-header hashes, manifest hashes, flags and CUDA 12.9.1 toolchain.
`campaign-source.patch` reconstructs the comparison changes on merged
`61168b982c33c43b47c2d9f2aa438665b106e271`. The campaign manifest preserves
both measured arms; the final production manifest removes the embedded baseline
and qualifies the same candidate math. Final-library validation is recorded
separately, never relabeled as the measured campaign source.
`promoted-build.json` binds the final library to the campaign source plus
`promoted-source.patch`, whose native/compiler contents were committed unchanged
at `8e3a19a73f83a723f87fbe6d4dd6240cb7ceb2dd`. The final library is 164,232,888
bytes, a 223,440-byte (0.14%) increase over the standalone baseline.

Build Release sm_120 with fast compile and AOT shells off before timing. On the
measured campaign source, use the existing runner inside a finite allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:20:00 \
  env PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VIBEQC_LIBRARY=/path/to/frozen/campaign/libvibeqc.so \
  VIBEQC_DF_REFERENCE_FINAL_VALIDATION=0 VIBEQC_DF_FINAL_PROJECTION=auto \
  VIBEQC_DF_FORCE_SCREEN_ABS=off VIBEQC_DF_EXCHANGE=auto \
  VIBEQC_DF_SEED_EXCHANGE=auto VIBEQC_DF_FINAL_EXCHANGE=auto \
  VIBEQC_DF_RESIDENT_EXCHANGE=auto \
  python -m benchmarks.df_policy_endpoint --aos 384 --output /new/run/384-forces.json \
    --repeats 5 --control VIBEQC_DF_SHELL_POLICY --policies auto candidate \
    --components-after --source-patch /path/to/frozen/campaign/source.patch \
    --skip-cold --warm-checkpoint-in /path/to/384.checkpoint --expected-iterations 3 \
    --reference benchmarks/results/issue377-379-df/gpu4pyscf/water-hexadecamer-2s4-def2-svp-spherical.json
```

Repeat for 768 and with `--energy-only`; 96/192 omit the three-update gate.
Checkpoint and density hashes are retained in every endpoint record. Exact
historical checkpoint bytes remain local; a newly generated checkpoint is a
new campaign and must pass the same convergence, work and numerical gates.
The historical raw source/diagnostic file hashes are in `raw-files.json`.
Retained endpoint records preserve every value and sample order; scalar arrays
are formatted inline for review. Original run-byte hashes remain in that index.
Routine logs, binaries, checkpoints and complete profiler traces remain local.

Recompute statistics and gates from the retained records without a GPU:

```bash
python benchmarks/results/issue404-combined-rys/analyze.py > /tmp/issue404-summary.json
```

The full candidate holdouts, independent oracles and sanitizer coverage remain
in #394's [qualification record](../issue394-batch-rys/qualification.json).
The [final-library validation](validation.json) additionally records one native
shell-pair suite, 22 complete-force oracle cases with forced candidate selection,
two failed-neighbor cases, five checkpoint cases, and automatic 96/192/384/768
endpoints. Memcheck with leak checking and initcheck report zero errors for the
native automatic-fallback suite; the 42 mathematical candidates retain their
separate #394 sanitizer qualification. All final automatic endpoint energies
match the measured comparison arm exactly, force differences are below 8e-14
Eh/Bohr, and observed mathematical/resource/operator counters match. Small
domains select no Rys classes; both target domains select the five promoted
classes. These final-build checks are not extra samples in the clean campaign.

An initial validation wrapper forced `candidate` into the native fixture that
explicitly asserts the small-domain `auto` fallback. Its assertion failure is
retained in [validation-harness-failure.json](validation-harness-failure.json).
The corrected wrapper separates those policies. Two derivative-tier neighbor
cases initially skipped for a missing test-enable variable were then run with
that variable enabled; their passing Slurm follow-up is retained separately.

#206 owns fresh stock GPU4PySCF comparisons and any combination with #412;
independent savings must not be added without measuring the combined endpoint.

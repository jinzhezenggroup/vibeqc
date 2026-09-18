# Derivative work diagnostics (#395)

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

The opt-in ledger records actual Boys-series work and generated polynomial,
weight-folding and gradient-scatter operations by angular class and primitive
signature. The normal path keeps diagnostics disabled. No numerical algorithm,
screening threshold, metric cutoff or convergence/force gate changes.

| AOs | Baseline clean median (s) | Candidate, counters off (s) | Change | Updates in every sample |
| --- | ---: | ---: | ---: | ---: |
| 384 | 0.923888040 | 0.924347867 | +0.05% | 3 |
| 768 | 5.298158605 | 5.331490926 | +0.63% | 3 |

Each median has five clean samples in baseline-2, candidate-2, candidate-3,
baseline-3 order. All replays use the same frozen density per size, including
the separate diagnostic runs. Compilation, checkpoint import, priming, traces
and counters are outside clean timing. Both sizes satisfy the declared 2%
normal-path regression gate; this instrumentation is not a speedup claim.

The baseline is #397's frozen final implementation. Its later small-matrix
Jacobi telemetry correction does not affect these 384/768 AO cases. The
candidate source patch applies to #397 head
`23091a4575c3b2bf9288ae00bb173935a51364c3`; [build.json](build.json) records
its exact library, patch and generated-header hashes. All runs use RTX 5090,
CUDA 12.9.86, driver 580.95.05, Release sm_120, one host numerical thread,
batch-one RHF and spherical def2-SVP for both orbital and auxiliary bases.
Full settings and reference hashes are retained in every measurement.

Independent GPU4PySCF gates remain `1e-9` Hartree and `1e-8` Hartree/bohr.
Maximum clean errors are `3.911e-11` and `1.358e-10`, respectively. Baseline
versus candidate and counter-disabled versus counter-enabled energies agree
exactly in the paired observations; the maximum force difference is
`7.816e-14` Hartree/bohr. These comparisons retain normal convergence checks.

## Exact work and measured class activities

[384 AO](work/384.md) and [768 AO](work/768.md) provide readable class tables.
Their [384 JSON](https://github.com/jinzhezenggroup/vibeqc/blob/e215b30685f8a36ef0cc5772c9166837527f64a2/benchmarks/results/issue395-df-work/work/384.json) and [768 JSON](https://github.com/jinzhezenggroup/vibeqc/blob/e215b30685f8a36ef0cc5772c9166837527f64a2/benchmarks/results/issue395-df-work/work/768.json) also retain all
primitive signatures, Boys branches/order, active components, schedules,
resource limits and the independent host-reconstruction checksum. The six
existing executed-domain counters agree across baseline, candidate, enabled
and disabled detailed counters at both sizes.

| Source operation | 384 AO | 768 AO |
| --- | ---: | ---: |
| Active shell tasks | 3,557,376 | 28,385,280 |
| Primitive products / Boys evaluations | 21,976,064 | 175,132,672 |
| Positive-series evaluations | 5,461,916 | 21,650,876 |
| Positive-series iterations | 263,463,122 | 1,095,414,060 |
| Generic axis-polynomial calls | 64,583,424 | 515,349,504 |
| Emitted axis-cache coefficients | 623,227,392 | 4,966,631,424 |
| Coefficient-convolution iterations | 2,925,586,944 | 23,227,654,144 |
| Public weight loads | 56,623,104 | 226,787,328 |
| Nonzero logical public weights | 28,520,448 | 226,787,328 |
| Expansion products / shared fold atomics | 40,680,576 | 322,970,112 |
| A, B, C global atomics, each | 10,672,128 | 85,155,840 |
| Shared-physical-atom gradient updates | 1,585,440 | 6,378,048 |
| Distinct-physical-atom gradient updates | 30,430,944 | 249,089,472 |

Direct fold stores and local orbital accumulations are zero before #392/#393.
At 384, symmetric off-diagonal weights consume two public loads but yield
one logical weight after pair folding. At 768, packed weights are already
folded. All 27 fields are source operations, not hardware instruction or
traffic counts. The small-argument counter is a subset of the series branch.

| Group | 384 primitives | 768 primitives | 384 GPU ms, detailed off | 768 GPU ms, detailed off |
| --- | ---: | ---: | ---: | ---: |
| Pure s/p | 19,125,120 | 152,366,592 | 120.691 | 721.856 |
| Contains d | 2,850,944 | 22,766,080 | 83.514 | 581.698 |
| Seven #394 classes (overlaps above) | 20,061,952 | 159,874,048 | 117.433 | 691.128 |

These are actual class kernel activities in separate Nsight captures, still
intrusive and with the existing six domain counters enabled. Detailed-enabled
activities remain in the work JSON; detailed-disabled activities are in
`profiles/`. Packet durations are not divided among primitive signatures.
CUDA event component intervals in `summary.json` are a separate observable
and may include stream gaps.

The ledger supports the five issue hypotheses with explicit limits:

1. D-containing classes contribute about 13% of primitive products, 58% of
   convolution iterations and 41–45% of detailed-disabled class GPU activity.
   This establishes disproportionate work and time, not a causal instruction
   breakdown.
2. The positive Boys series executes 263 million / 1.095 billion iterations;
   its elapsed-time fraction is not measured separately.
3. Pure s/p folding executes 14,151,808 / 112,562,688 shared FP64 atomics.
4. The 768 workload executes exactly 255,467,520 gradient atomics: nine per
   active shell task, split equally among A/B/C.
5. The seven Rys targets cover about 91.3% of primitive products at both sizes.

## Intrusive costs and validation

| AOs | Traced detailed off → on (s) | Added time | D2H bytes, off → on | Stream drains, off → on |
| --- | --- | ---: | --- | --- |
| 384 | 0.919587 → 0.936168 | 1.80% | 1,200 → 436,656 | 1 → 19 |
| 768 | 5.326386 → 5.429696 | 1.94% | 2,352 → 5,518,128 | 1 → 235 |

Each is a separate single traced observation. The bounded diagnostic buffer
adds exactly 82,944 device bytes and the same host capacity. Register usage
changes even when the sink is null: SSS 96 → 132, PSS 130 → 148 and DSS
252 → 252. SSS theoretical resident threads fall 640 → 384 out of 1,536;
PSS stays 384 and DSS 256. Actual CUDA resource maxima and Nsight resource
rows are retained with their distinct measurement definitions. No achieved
occupancy, contention percentage or hardware-counter claim is made.

Library size grows 198,259,648 → 205,297,872 bytes (+3.55%). Disabled-counter
post-call process residency is unchanged: 8,426,356,736 bytes at 384 and
21,252,538,368 at 768. The [Agent Note](../../../.agents/notes/implemented/performance/2026-09-16-df-shell-work-diagnostics.md)
explains the bounded aggregation and why a second kernel family was deferred.

[validation.json](validation.json) retains commands/counts and original log
hashes: 203 compiler/Libcint/Boys checks, 381 final CPU checks including 19
ledger checks, 120 CUDA Python tests, native full/symmetric/packed and mixed
spherical/Cartesian sparse/partial-panel cases, and Compute Sanitizer with
zero errors. [environment.json](environment.json) records tool/package versions.

## Reproduction

Build before scheduling measurements. Reconstruct the exact candidate in a
clean worktree at `23091a4575c3b2bf9288ae00bb173935a51364c3` by applying
`reproduction/source.patch`. Reconstruct the baseline at
`b29649f895ad53a549bef03341c2349cbe131f6c` with
`../issues388-391-df/reproduction/final-source.patch`. Build each using:

```bash
cmake --preset cuda-release-sm120 -DVIBEQC_BUILD_TESTS=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc
cmake --build --preset cuda-release-sm120 --parallel 4
```

Freeze each library beside its matching `source.patch`. Also copy the
candidate's `build/cuda-release-sm120/generated/generated_df_shell_derivatives.cuh`
beside its library. Run the current checkout's benchmark/reducer code against
those frozen libraries. Configure absolute directories and an interpreter
with the recorded dependencies:

```bash
export VIBEQC_WORK_BASELINE=/path/to/frozen-baseline
export VIBEQC_WORK_CANDIDATE=/path/to/frozen-candidate
export VIBEQC_WORK_OUTPUT=/path/to/fresh/qualification
export VIBEQC_WORK_CHECKPOINTS=/path/to/checkpoints
export VIBEQC_WORK_PYTHON=/path/to/python
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:35:00 bash \
  benchmarks/results/issue395-df-work/reproduction/run.sh checkpoint 768
```

Repeat the finite allocation for `checkpoint 384`, then `clean 384`,
`clean 768`, `profile 384` and `profile 768`. Every GPU test/profiler stays
inside Slurm with its assigned device visibility. Recreated checkpoints may
differ slightly from retained hashes; all policies must replay the same
checkpoint and the runner must still pass the three-update guard. A failure
is not permission to force convergence or loosen the gates.

The original run/validation scripts are retained beside `run.sh`. Job 9720
was deliberately cancelled after all five clean samples and their component
passes completed; job 9721 finished the 768 overhead/profiles. This split was
an allocation boundary, not a numerical failure. `collect_evidence.py` consumes
an artifact root with `work-v1/qualification`, `work-v1/build.json` and
`work-v1/environment.json`, checks all required inputs and identities, and
refuses a nonempty publication destination. It records the 2% gate explicitly.
Logs, binaries, checkpoints and profiler databases remain outside Git; every
retained file is below 1 MiB.

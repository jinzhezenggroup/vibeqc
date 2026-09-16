# Rejected isolated DF optimizations

This records six candidates that failed the production performance gates for
#392/#393. `early-folding/` contains direct shared stores and the first register
variant; `identity-folding/` adds constant tuple extents and unit coefficients.
None demonstrates an endpoint improvement. `serial-grouping/` regresses both
sizes; `block-grouping/` and `warp-grouping/` have lower 384-AO medians but
regress at 768 AOs. See the
[decision note](../../../.agents/notes/rejected/2026-09-16-df-folding-and-c-grouping.md).

The later candidate directories contain source/native-test patches against
`1c3f2ab8d6df6f06bee526b89a807511ef726301`, build and validation identities,
the complete-endpoint summary, and one compact record per AO size. The records
retain every clean sample's timing, energy, forces, errors, iterations and
metric metadata, workload settings and reference hashes, class-level executed
work/resources, and separate disabled-counter Nsight kernel observations.
Detailed signature rows and transient traces are omitted; their original
ledger, trace, database and measurement hashes are retained. No cold or
changed-geometry performance claim is made. The first direct candidate has
native/memcheck evidence but no recorded full Python CUDA suite; the other five
have 120-test CUDA suite records. Per-candidate validation scopes are explicit.

The two `early-folding/*-source.patch` files instead apply to
`c5475c49b331c989fbe0aafb3ce19685d7cb254d`. Their `conditions.json` discloses
compilation overlapping part of the intrusive profiles; clean endpoint timing
was isolated. The register native executable was rebuilt with the subsequent
#395 packet assertions, as recorded in `builds.json`; its library was unchanged.

Reconstruct each candidate in a separate checkout of the base by applying its
`source.patch` and `native-test.patch` (or the named early source patch), then
apply `cuda-test.patch` when present. Use the `cuda-release-sm120` CMake
preset to build `vibeqc` and `vibeqc_df_shell_pairs_tests`. Freeze each completed
library before measuring. Baseline evidence and the runner's scientific
contract are in `benchmarks/results/issue395-df-work/` and
`benchmarks/df_policy_endpoint.py`.

All GPU commands require a finite Slurm allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 bash run-comparison.sh
```

Inside the job, preserve Slurm's device visibility and set `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `PYTHONPATH=python:.`, and `VIBEQC_LIBRARY` to the
selected frozen binary. Generate a common post-cold checkpoint on the baseline
with the runner's `--warm-checkpoint-out`, then pass the same checkpoint to all
variants with `--warm-checkpoint-in` and `--skip-cold`. Use `--aos 384` or `768`,
`--control VIBEQC_DF_SHELL_WORK --policies 0 --expected-iterations 3`, and
`--components-after`. The independent references are
`benchmarks/results/issue377-379-df/gpu4pyscf/water-hexadecamer-2s4-def2-svp-spherical.json`
and `water-32mer-4s4-def2-svp-spherical.json`; pass the matching `--reference`.
Run baseline-2, candidate-2, candidate-3, baseline-3 groups in that order, where
the suffix is `--repeats`. The original paired early-folding run used
baseline-2, direct-2, register-2, register-3, direct-3, baseline-3.
Keep compilation and profiling outside clean timing.
New checkpoints have their own identities; do not claim exact historical
density reproduction unless their retained hashes match.

For intrusive work attribution run a separate `--policies 0 1 --repeats 1
--trace --cuda-profile` capture under Nsight Systems with
`VIBEQC_DF_SHELL_COUNTERS=1`, then reconcile it with the retained
`reproduction/df_shell_work_ledger.py` using the frozen generated header.
Run that file with `PYTHONPATH=python:.` from the reconstructed candidate
checkout, passing `--trace`, `--measurement`, `--generated-header`, `--nsys`
and `--output`. It reads the executed grouping fields and supports both
historical schedule names; the candidate base's original reducer cannot
decode every added diagnostic. Source
counts do not measure atomic contention. Never infer a promotion from them.

Publication changes only evidence and notes. Production selectors and scientific
tolerances remain at the merged #397/#398 baseline. #392/#393 stay open;
the separate [Rys prototype](../issue394-deferred/README.md) is deferred without
an endpoint performance verdict.

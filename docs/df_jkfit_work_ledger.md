# Practical JKFIT derivative work ledger (#437)

`benchmarks.issue437_practical_jkfit_work_ledger` is the Phase-A reducer for
issue #437. It consumes the existing intrusive shell-work ledger plus the exact
matching `vibeqc.df_trace` record. It does **not** alter production DF execution
or enable screening.

The reducer reports, per angular derivative class:

- physical orbital shell-pair and auxiliary-shell counts;
- shell triples considered/executed;
- primitive products considered/executed;
- public response-weight loads and nonzero fraction;
- generated lowering and, when an Nsight-backed shell ledger is supplied,
  schedule, kernel launches, and kernel GPU time;
- logical response-weight bytes; and
- the fraction of the complete `force_response` GPU interval.

The report deliberately does not call logical bytes measured DRAM traffic.
Response-weight magnitude histograms, distance/exponent bins, and measured shell
metadata bytes remain explicit Phase-A gaps until those quantities have direct
instrumentation.

## Collection

First capture the existing separate intrusive component/shell diagnostics. Keep
clean endpoint timing separate, as required by `benchmarks.df_policy_endpoint`.

For the practical 96/464 case, use the canonical explicit basis snapshots:

```bash
export PYTHONPATH=python:.
export VIBEQC_LIBRARY=/absolute/path/to/libvibeqc.so
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export VIBEQC_DF_WEIGHTED_EXECUTION=shell
export VIBEQC_DF_DERIVATIVE_PAIRS=symmetric
export VIBEQC_DF_SHELL_SCHEDULE=compact
export VIBEQC_DF_PRIMITIVE_BUCKETS=packet
BASIS=benchmarks/results/issue206-practical-auxiliary/identity

python -m benchmarks.df_policy_endpoint \
  --aos 96 --repeats 1 \
  --control VIBEQC_DF_SHELL_POLICY --policies auto \
  --cpu-reference \
  --orbital-basis-file "$BASIS/cc-pvdz.json" \
  --auxiliary-basis-file "$BASIS/cc-pvdz-jkfit.json" \
  --components-after --shell-work \
  --output .artifacts/issue437/practical-96.json
```

Run GPU collection only in a finite Slurm allocation and preserve scheduler
device visibility. Export a separate Nsight capture of the intrusive diagnostic
call to SQLite, then reduce the matching force-call trace with the existing
shell ledger:

```bash
python -m benchmarks.df_shell_work_ledger \
  --trace .artifacts/issue437/practical-96.diagnostic-0-auto.jsonl \
  --measurement .artifacts/issue437/practical-96.json \
  --generated-header build/generated/generated_df_shell_derivatives.cuh \
  --orbital-basis-file "$BASIS/cc-pvdz.json" \
  --auxiliary-basis-file "$BASIS/cc-pvdz-jkfit.json" \
  --nsys .artifacts/issue437/practical-96.sqlite \
  --output .artifacts/issue437/practical-96-work.json
```

Repeat with the same orbital basis and an equal auxiliary basis for the 96/96
control, then repeat for the 192/928 and 192/192 water-octamer cells. The small
OH UHF control is 19/93 versus 19/19 and should be collected with a runner that
supports the `oh-def2-svp-spherical-uhf` benchmark case.

Finally combine cells and enforce the headline dimensions:

```bash
python -m benchmarks.issue437_practical_jkfit_work_ledger \
  --cell equal-96 equal \
    .artifacts/issue437/equal-96-work.json \
    .artifacts/issue437/equal-96.diagnostic-0-auto.jsonl \
  --cell practical-96 practical \
    .artifacts/issue437/practical-96-work.json \
    .artifacts/issue437/practical-96.diagnostic-0-auto.jsonl \
  --cell equal-192 equal \
    .artifacts/issue437/equal-192-work.json \
    .artifacts/issue437/equal-192.diagnostic-0-auto.jsonl \
  --cell practical-192 practical \
    .artifacts/issue437/practical-192-work.json \
    .artifacts/issue437/practical-192.diagnostic-0-auto.jsonl \
  --expect equal-96 96 96 \
  --expect practical-96 96 464 \
  --expect equal-192 192 192 \
  --expect practical-192 192 928 \
  --pair water-96 equal-96 practical-96 \
  --pair water-192 equal-192 practical-192 \
  --output .artifacts/issue437/work-matrix.json
```

`--pair` rejects an AO mismatch and verifies that the complete orbital shell metadata and
pair-enumeration mode match before reporting Naux/work ratios. Auxiliary
classes that exist only in JKFIT (for example auxiliary-f classes) are reported
as new classes instead of receiving a manufactured finite ratio.

The resulting JSON is intrusive diagnostic evidence. It is not a clean endpoint
timing claim and is not a screening policy.

Agent: ChatGPT
Model: GPT-5.6 Sol

# CUDA DF component evidence

`VIBEQC_DF_TRACE=/absolute/path.jsonl` enables diagnostic CUDA event intervals,
NVTX ranges (when toolkit NVTX headers are available), transfer/work counters,
and a logical three-center tile ledger. It covers RI-J, RI-K, generated raw and
transformed tiles, DF-HF response weights, exchange response matrix products,
Coulomb response, spectral metric response, weighted generated derivatives,
and the one-electron/overlap-Pulay response.

The disabled route performs no extra CUDA calls, allocations, synchronizations,
NVTX calls, or file writes. An enabled ordinary operation records events on its
existing stream and waits on the final event before writing JSONL. This changes
submission cost and overlap: use separate **unprofiled** endpoint runs for any
performance claim.

## Reproducible force probe

Build a Release library with production compiler settings, then run both probes
through a finite Slurm allocation, preserving the assigned device visibility:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH="$PWD/python" /path/to/python benchmarks/issue206_df_force_probe.py \
  --library "$PWD/build/cuda/libvibeqc.so" --repeats 3 \
  --output /path/to/evidence/unprofiled.json

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH="$PWD/python" /path/to/python benchmarks/issue206_df_force_probe.py \
  --library "$PWD/build/cuda/libvibeqc.so" --repeats 1 \
  --component-trace-dir /path/to/evidence/new-traces \
  --output /path/to/evidence/components.json
```

The default cases are the #206 96- and 192-AO systems. The probe retains energy
parity and iteration-count gates, stores exact library/patch/untracked-file
hashes, preserves each solve's raw JSONL with a hash, and rejects missing or
invalid instrumentation. Output paths should be outside the source tree (or
ignored by Git) so writing evidence does not change the identified checkout.
Trace files must be fresh; the probe never overwrites earlier captures.

For executed graph-node attribution, wrap a separate probe with Nsight Systems:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH="$PWD/python" /group/software/cuda-12.9.1/bin/nsys profile \
  --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --output=/path/to/evidence/df-nsys \
  /path/to/python benchmarks/issue206_df_force_probe.py \
  --library "$PWD/build/cuda/libvibeqc.so" --repeats 1 \
  --component-trace-dir /path/to/evidence/new-nsys-traces \
  --output /path/to/evidence/nsys-components.json
```

Keep the `.nsys-rep`, exported tables and original JSON alongside source and
binary identities. Nsight timing also has instrumentation overhead.

## Interpretation and limitations

- `execution=graph_capture` contains construction counts and host time only.
  It creates no CUDA events and adds no synchronization during capture. Those
  counters **do not count graph replay**, including device tail launches.
- Regions record inclusive host and GPU milliseconds with parent indices.
  `benchmarks/df_component_ledger.py` subtracts only immediate children to
  produce exclusive components. Host and GPU views overlap and must not be
  added together. CUDA events include stream idle gaps between submissions.
- Root exclusive time remains unclassified runtime work. The force attribution
  compares named host intervals and measured synchronization to the same
  profiled pair's force increment. Nuclear assembly, host packing before the
  operation, and work outside the roots remain explicit residuals. A ratio
  over one may reflect variation between the two SCF solves.
- A tile key includes absolute source system, AO-pair and auxiliary ranges,
  derivative coordinate (`-1` for values), and raw/transformed representation.
  `system_offset` locates a single force call within a batched source.
  Production multiplicity exposes repeated generation within each call.
- Generated value byte counts measure logical output work, not device traffic.
  Weighted derivative bytes measure consumed weights; the fused derivative
  kernel does not allocate or write a full nuclear derivative tensor. They
  must not be interpreted as materialized derivative bytes.
- Scratch counters report per-operation allocation sizes. Their sums are not
  live-memory peaks; the #203 resource ledger remains the complete memory gate.
- Each operation is limited to 65,536 regions and logical tile keys. Dropped
  entries, CUDA timing errors, malformed hierarchy, missing timings, truncated
  JSONL and inconsistent logical work totals invalidate evidence. Missing sink
  output is rejected by the probe rather than changing the scientific status.

This instrumentation alone does not complete #282, #283 or #284. Resident and
streamed reuse, force optimizations, occupied-factor exchange, numerical gates,
and the full batch-1/batch-4 #206 matrix require separate implementation and
evidence.

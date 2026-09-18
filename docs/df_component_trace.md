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

## Host eigensolve ledger for #308

`VIBEQC_DF_HOST_TRACE=/absolute/fresh.jsonl` records host scopes and actual
CPU-reference eigensolve invocations. The existing force probe's
`--component-trace-dir` collects this companion file together with the CUDA
ledger. The two clocks are reported separately. Setting either trace variable
implicitly on a clean force-probe invocation is rejected.

Each `reference_eigensolve` leaf records its dimension, reason (`overlap`,
`core_guess`, `iteration`, `seed_validation`, `final_fock`, `reference_export`, `fallback`, or `unspecified`),
item, host wall milliseconds, thread CPU milliseconds and exceptional exit.
Only the actual oracle entry emits this leaf; an intended solve, cache flag or
graph declaration cannot count as executed work. The Jacobi arithmetic and
overlap singularity threshold are unchanged. A small observer interface keeps
the independent reference and initial-guess modules free of runtime/CUDA
dependencies. The runtime owns timers, bounded records and file output.

Independent CUDA `FockPlan.solve()` consumers with either fitted J or K borrow
the same prepared ordinary device provider for setup, host-driven SCF iterations,
finalization and applicable RHF reference export. The `iteration` reason counts
each actual spin solve, including DIIS matrices. These frames do not carry the
compact CUDA solver's physical-state identity and do not authorize retained-state
reuse. Strict warm-seed checks also use that provider for S and each spin's
metric occupations, recorded as `seed_validation`. The existing 1e-7 seed
symmetry tolerance is preserved: inputs outside the device frame's tighter
symmetry contract select the original reference validation before submission
and record `fallback`. No density is repaired and device failures propagate.
Host DIIS and the independent finalization sequence remain in place.
Exact-only CUDA plans and the CPU oracle keep their reference provider. Explicit
reference setup/finalization controls affect only those stages; they never
silently retry a failed device operation.

`benchmarks/df_component_ledger.py` validates the host JSONL and subtracts
immediate children within each root to produce exclusive host phases. Root
and ancestor IDs disambiguate source indices local to different CUDA buckets.
Thread CPU time is `null` when the platform cannot supply it; process CPU time
is never substituted. Host wall time includes waiting and descheduling, so
wall minus CPU time is not automatically all GPU waiting. Concurrent CPU
worker roots overlap: their wall times must not be summed into endpoint time.
Host and CUDA-event times must not be added together either.

If host ablations take different SCF iteration/retry branches, the matrix CLI
still fails. It writes completed samples and the differing branches to a
separate `host-<ao>ao-b<batch>.rejected.json`, with source/library identity and
input controls. The failed manifest links this diagnostic artifact; it has no
timing assessment and is never a passing endpoint. Raw host traces remain
available. Scientific endpoint comparisons that follow the branch gate are
explicitly marked unchecked in the rejected record.

The existing #206 matrix entry point also supports native protocol controls:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:20:00 \
  env PYTHONPATH="$PWD/python" /path/to/python benchmarks/issue206_df_matrix.py \
  --run --host-workloads --case water-tetramer-def2-svp-spherical --batch 1 \
  --library "$PWD/build/cuda/libvibeqc.so" --memory-budget-bytes 1073741824 \
  --energy-only --repeats 5 --host-trace-dir /path/to/fresh-host-traces \
  --output-dir /path/to/host-components
```

It retains cold creation/solve/destruction, fixed-seed replay, energy-only or
energy-plus-force, and a changed-geometry item. Every changed sample restores
the original geometry before starting its timer. Omit `--host-trace-dir` for
separate clean timing; choose batch 4 or the existing 192/384-AO inputs to
extend the domain. Identical A/B configurations are an ABBA protocol control,
not a speedup or an external-engine parity result. Complete traffic/device
work still comes from the separate CUDA/Nsight ledger; the original matched
DF-versus-DF matrix remains the owner of parity acceptance.

For the lazy-initial-density slice, add `--eager-core-ablation` to the host
workload invocation. The baseline selection explicitly requests the old warm
core frame through `VIBEQC_DF_EAGER_CORE_GUESS=1`; the candidate consumes the
same frozen density without that unused solve. Cold density, overlap and
finalization are identical. Actual leaf counts must distinguish the selections
in a separate traced run. Clean interleaved endpoint timings own any savings
claim; the runner rejects an ambient eager-guess flag. This private diagnostic
control is not a different physical initial guess or a fallback policy.

The default `--memory-budget-bytes 0` preserves the original probe's host
resident compatibility route. A positive value, for example `268435456`,
selects the existing bounded generated-source execution. Record and compare
both routes explicitly: the original #206 probe does not exercise source-backed
tile regeneration. Compatibility one-electron/nuclear derivative exports have
their own roots, and host response weights remain labeled as host work.

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
  profiled pair's force increment. Host nuclear assembly, packing before the
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

The final-state integration adds two explicit host workload comparisons:

- `--final-state-ablation` pairs actual forced ordinary-device final rebuilding
  with verified candidate retention (or necessary ordinary-device correction).
  Both selections use the same lazy/cache preparation and frozen density.
- `--combined-host-ablation` pairs eager core guesses, rebuilt overlap,
  reference setup providers and forced reference final correction with the
  combined lazy/cache/device/verified-state path. Both sides enforce the current
  strict physical-state gates. This diagnostic baseline is not the historical
  legacy finalizer, which had a weaker single-rebuild sequence.

Each flag is exclusive with other ablation flags and requires `--host-workloads`.
The trace gates require a candidate read, current physical F evaluation and
validation for every item, even at zero eigensolves. UHF counts two provider
leaves per joint correction; force fixed-point probes also contribute provider
leaves. A rejected probe promoted into correction is counted once. With these
fields present, total provider leaves are `spins * (corrections + checks -
promotions)`, and successful force items satisfy `checks = items + promotions`.
Forces require one verified W construction and one force-response consumer per
item. A necessary cold/changed correction is valid
on either side; omitted checks, accidental reuse in a forced sample or hidden
reference fallback fail the gate. Source/library identities and SCF iteration/
retry branches must match. Run at least five interleaved samples in clean and
separate traced invocations; never multiply ratios from historical binaries to
claim the combined improvement. External parity remains the matched #206 gate.

## Incomplete-stage journal for #308

`VIBEQC_DF_PROGRESS_TRACE=/absolute/fresh.jsonl` writes a separate append-only
journal. Every BEGIN, VALUE and END line is closed immediately. A killed process
therefore leaves its active stage visible without waiting for the enclosing
operation to finish. This guarantees process-exit visibility, not power-loss
persistence (`fsync` is not used). Missing END records and partial last lines
remain incomplete evidence.

The journal covers existing host scopes (including final validation and force
response), one-electron/source setup, metric factorization, raw generation,
metric GEMM, compact SCF and host DIIS retry. CUDA component boundaries wait on
their own event **only when progress tracing is enabled**; this adds intrusive
synchronization beyond ordinary component tracing. Graph construction never
records or waits on CUDA events and ends as `graph_constructed`. A stream END
means submitted work completed; a host END means the scope returned or unwound.
Neither status asserts numerical convergence. Dispatch statuses and device
convergence readbacks remain separate observations.

K records planner AO-pair/auxiliary tiles and executed AO rows/output auxiliary
widths. Source-backed K also reports `executed_raw_auxiliary_tile` and
`raw_tensor_passes_per_k`, the shared dense/occupied policy's full-traversal
prediction per system. With R row blocks and T output blocks, the current
row/column schedule generates R*T tensor-equivalent raw values. This is a
source-work count, not a timing multiplier. Positive-budget plans use the
available four-buffer allowance before choosing these dimensions. The fused
route is restricted to Q=1 when raw staging also holds only P=1; other short
output tails reuse raw P through GEMM.
`fused_source_auxiliary_evaluations` counts logical `(pair,Q,P)` recurrence
work, while `raw_panel_source_auxiliary_evaluations` counts `(pair,P)` work for
the GEMM route. Both describe submitted work within their execution mode;
capture counts are construction templates, not executed evaluations. A killed
kernel's pre-launch counts do not imply it completed. Source-backed J records
its own two raw passes, independently of K's transformed-panel work.
`tile_submission` is one JSON-object string per nonempty tile, written before
launch. Decode its scalar label as JSON to obtain system, pair/auxiliary
begins/counts, derivative coordinate (-1 for values), and transformed flag.
It remains available even if the enclosing CUDA operation never completes.

Compact scopes record the solve epoch, caller-density seed, graph construction
attempts, host graph replays and cumulative per-system device iterations after
synchronization. Retry scopes identify their original caller-density seed and
own iteration numbers. Do not sum cumulative readbacks, multiply graph replay
counts by a configured iteration limit, or report only the final retry's
iteration count as total SCF work. For a killed graph replay the final device
iteration count is unknown. The journal does not inspect every tail-launched
kernel; use an independently bounded device timeline when that detail is needed.
`compact_eigensolve` records provider and the number of submitted eigensystems
(one per system/spin). Its ordinary stream duration has a distinct CUDA-event
boundary; capture records describe only constructed nodes. Executed graph
iteration counts still come from the device readback, not this host scope.

Read a completed or interrupted journal with:

```bash
python -m benchmarks.df_progress_ledger /path/progress.jsonl \
  --output /path/new-progress-summary.json
```

The summary subtracts immediate completed children for exclusive diagnostic
wall intervals, and leaves open scopes explicit. Its observations preserve
parent IDs, execution modes and ordering. Concurrent root wall times must not
be summed into endpoint time. Existing clean #206 runners reject an ambient
progress trace, just as they reject ambient component tracing.

`benchmarks/issue308_stage_probe.py` and `benchmarks/df_stage_probe.cpp` provide
bounded setup/fixed-D J/dense-K/occupied-K experiments. Prepare an input with
`python -m benchmarks.issue308_stage_probe --prepare --case CASE --output INPUT`.
The input stores independently converged PySCF orbitals, overlap, J/K and the
metric rank policy. An optional `--checkpoint` must match the exact geometry
and basis and pass the same physical-state checks. The native probe checks its
full overlap and imported occupied S-orthogonality before constructing a DF
plan. Its D is a diagnostic seed, not a cold-start endpoint.

Compile the probe against the unified Release library, then execute through a
finite Slurm allocation. For example, from the repository root:

```bash
c++ -std=c++20 -O3 -Iinclude -Isrc -I/path/to/cuda/include \
  benchmarks/df_stage_probe.cpp -Lbuild/cuda -Wl,-rpath,"$PWD/build/cuda" \
  -lvibeqc -ldl -o .artifacts/df-stage-probe
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:06:00 \
  env PYTHONPATH=python:. python -m benchmarks.issue308_stage_probe \
  --input INPUT --output NEW_RUN --library build/cuda/libvibeqc.so \
  --probe .artifacts/df-stage-probe --progress --timeout 300
```

Zero probe tile arguments request full dimensions, not a public budget-policy
change. `--ao-pairs 8192 --auxiliary-tile 128` at 768 AOs exercises the previously
reported planner shape and its rounded 7680-pair allocation. Use
`--operation dense` or `--operation occupied` for a bounded K-only probe. The runner retains
source/library identity, actual loaded-library identity from the probe, input
hashes, build cache, all DF diagnostic variables, Slurm visibility, memory
samples, completed rows and timeout/failure disposition. Its memory sampling
makes all its timings diagnostic, even without `--progress`; existing #206
runners own separate clean endpoint timing.

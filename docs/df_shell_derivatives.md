# Generated shell-block DF response

`VIBEQC_DF_WEIGHTED_EXECUTION=shell` selects generated weighted three-center
derivatives across all 64 s/p/d/f shell classes. `shell-sp` retains the original
seven non-SSS s/p classes as a comparison subset. Automatic consumer admission
uses the centralized [work profile](df_tuning.md#derivative-consumer-admission),
separate from device-metric/source/state/resource correctness gates. Small work
and unknown target profiles retain the generic consumer. Serial mappings and
attribution probes retain their explicit fallbacks. The packed occupied-response
layout remains separately qualified; shell admission never grants that layout.

Historical #308 shell-schedule measurements remain in
`benchmarks/results/issue308-shell-schedules/`; they are not current endpoint
latencies. The current work-based decision is recorded in the
[admission note](../.agents/notes/implemented/performance/2026-09-19-df-work-admission.md).

The scalar and shell consumers use the same generated primitive geometry and
Boys routine. The compiler builds each shell class's axis-moment cache from
`axis_polynomial`, the existing Gaussian-moment DAG. Component derivatives use
the same raised/lowered Gaussian identity as the generic generated evaluator.
The two orbital derivatives share
`H_i = sum_jk v_j w_k F_(i+j+k)` for each differentiated axis. Raising the
second orbital center uses the exact polynomial identity
`M_(a,b+1,c) = M_(a+1,b,c) + (A-B) M_(a,b,c)`, so the cache reserves only
the raised-first-center boundary. The coefficient beyond the base degree is
zero, including coincident centers. This reduces polynomial preparation and
shared storage without changing the primitive geometry/Boys DAG.
The auxiliary derivative follows from translation of the two independent
orbital centers. Exponents, normalization and external response weights remain
fixed in this derivative.

The native runtime owns compact angular-class shell lists and AO offsets. It
projects each shell block's public weights through the existing normalized
sparse Cartesian expansions once. One generated subgroup traverses primitive shell
products, reuses geometry/Boys/axis moments across components, and immediately
contracts their derivatives into six independent gradient coordinates. A local
reduction and the existing atom-gradient sink finish the shell block. There is
no nuclear-coordinate derivative tensor or expanded shell-triple task array.

An auxiliary-major response panel may cut through a shell. The host narrows each
class list to intersecting shells; the device treats out-of-panel public weights
as zero. The next panel can revisit the same shell with disjoint weights. There
is no symmetry multiplier. Full shell execution consumes every three-center
weight; `shell-sp` partitions its subset from the generic thread consumer.
Metric derivatives, serial mapping, and the host-weight compatibility adapter
retain the generic generated route. Generic three-center work is excluded only
after the corresponding shell launches have been submitted successfully.

`VIBEQC_DF_SHELL_SCHEDULE` chooses among three compiler-owned variants:

| Selector | Component ownership | Shells per block |
| --- | --- | --- |
| `warp` | 32 lanes per shell | 1 |
| `packed` | 32 lanes per shell | Up to 4, bounded by shared storage |
| `compact` | 4, 8, 16 or 32 lanes according to component count | Up to 128 threads, bounded by shared storage |

`VIBEQC_DF_PRIMITIVE_BUCKETS=on` splits each angular list by exact primitive
count and launches homogeneous `(la, lb, lc, nprim_a, nprim_b, nprim_c)` products.
`packet` batches up to 24 signature ranges into each angular-class launch;
every block still belongs to one homogeneous signature. `off` retains the
angular-only traversal. Primitive counts are uniform runtime parameters; the
generated derivative equations and component schedule are unchanged. The
control is recorded in prepared replay metadata.

The default `auto` selects packets through the centralized primitive-work
profile, with one density term, spherical bases, symmetric/packed pairs and a
full `shell` / `compact` consumer. It requires heterogeneous contraction lengths
within at least one angular class. There is no water shell fingerprint, exact
AO/rank pair or equal orbital/auxiliary dimension requirement. Unknown profiles,
small primitive work, homogeneous contractions and incompatible diagnostic
schedules retain angular-only execution. Explicit `on` and `packet` controls
remain available for qualification. The original
[signature scheduling note](../.agents/notes/implemented/performance/2026-09-15-df-signature-packets.md)
preserves historical alternatives; the current work profile supersedes its
endpoint-specific admission boundary.

Each signature preserves the original shell order, so auxiliary-panel clipping
still uses the same public AO offsets. A symmetric or packed same-signature
orbital product is triangular; different signatures use a rectangle once. Full
mode traverses both orbital orientations and keeps same-signature rectangles.
Grouping applies to the selected shell subset only; generic, dense, metric and
unsupported-state consumers preserve their existing contracts.

These products are implicit task queues. Additional device storage is one
shell-ID array per basis, charged to the response budget. The host retains
compact signature ranges; no shell-triple descriptors or device count/prefix/
scatter buffers are materialized. Metadata is rebuilt within each complete
force call, so its construction and launch-dispatch cost belong to the endpoint.

A packet stores only bounded product ranges and block-count prefixes in kernel
parameters (at most 4 KiB including all arguments). Descending primitive-product
order lets expensive small ranges overlap cheaper ranges in the same grid.
`grid_constant` keeps the read-only packet in parameter storage when a block
selects its range. Full packets flush before additional ranges are admitted.
This adds neither a device allocation nor a synchronization; storage grows
with shells/signatures and never with the shell-triple domain. It does not use
a persistent worker or import Direct J/K screening mathematics.

Large component blocks cycle across the same lanes. Distinct shell groups use
disjoint shared arrays and subgroup masks, so ragged primitive counts and
clipped panels never require an unrelated group to reach a barrier. The
generator reserves 1 KiB for compiler/runtime shared state and keeps total
storage below 48 KiB, including the FFF weights and moment cache. Resource
qualification compares these bounds with every compiled kernel's actual use.

Metadata allocation grows with shell count and is charged before choosing the
response tile from the remaining budget. Static shared storage is bounded by
the generated angular class. The launcher borrows the response owner's stream
and adds no synchronization or allocation of its own during normal execution.
The explicit detailed-work diagnostic below adds readbacks and stream drains.
Selecting the sharded
gradient diagnostic together with shell execution is rejected because that
diagnostic specifically controls the original AO-element sink layout.

Set `VIBEQC_DF_SHELL_COUNTERS=1` and enable the existing component trace to obtain
device execution counters. These diagnostic atomics are disabled for clean
timing:

| Counter | Meaning |
| --- | --- |
| `shell_triples_visited` | Device shell tasks reaching the panel consumer |
| `shell_triples_nonzero` | Tasks with a nonzero projected Cartesian weight |
| `shell_public_weights_nonzero` | Nonzero public weights consumed before projection |
| `shell_primitive_products` | Actually executed primitive geometry/Boys evaluations |
| `shell_cartesian_component_products` | Nonzero Cartesian contributions served by those evaluations |
| `shell_public_weights_consumed` | Public response-weight loads, including both off-diagonal orientations in symmetric mode |

Set `VIBEQC_DF_SHELL_WORK=1` alongside `VIBEQC_DF_SHELL_COUNTERS=1` and
`VIBEQC_DF_TRACE` for the detailed mathematical work ledger. It records actual
Boys positive-series iterations and aggregates fixed generated loop counts per
active primitive/component. Counters are named `shell_ABC_work_FIELD` and
`shell_ABC_pNA_NB_NC_work_FIELD`, preserving both angular class and primitive
signature. A zero primitive signature denotes a heterogeneous angular-only
launch and cannot be used for homogeneous host-domain reconstruction.

The detailed fields include geometry/Boys calls, requested Boys-order sum,
positive-series iterations, polynomial preparation calls, emitted cache
coefficients, convolution iterations, active component products, public weight
loads, expansion products, weight-fold shared atomics/direct stores, A/B/C
gradient atomics and subgroup rendezvous. `boys_small_argument` counts arguments
below `1e-8` within the series branch; it is a subset of `boys_series`, not a
third numerical branch. `boys_large_argument` is the branch at arguments at
least 30. A gradient update counts as shared-atom work when that mathematical
center has the same physical atom as either other center; shared and distinct
counts partition the A/B/C updates. Counts describe emitted source operations,
not hardware instructions, memory transactions, or elapsed-time fractions.
`public_nonzero_weights` counts nonzero logical weights after pair folding:
one symmetric off-diagonal weight consumes two public loads but contributes
at most one such nonzero weight before Cartesian expansion.

Detailed diagnostics borrow a fixed buffer of 24 packet rows, 16 shards and 27
unsigned 64-bit counters: 82,944 device bytes and the same host readback size.
The force-response owner charges both allocations before selecting its panel.
Each packet/group readback drains the stream before reusing that buffer; bytes
and drains enter the response resource statistics. `shell_work_panel_BEGIN_COUNT`
records actual panel repetitions for independent domain reconstruction. All
these allocations, diagnostic atomics and readbacks are disabled when the
control is absent or zero. Intrusive diagnostic times must remain separate
from clean endpoint medians.

For the 384/768 AO headline cases, reduce a single force-call trace with:

```bash
PYTHONPATH=python:. python -m benchmarks.df_shell_work_ledger \
  --trace detailed.jsonl --measurement measured.json \
  --generated-header build/cuda-release-sm120/generated/generated_df_shell_derivatives.cuh \
  --nsys measured.sqlite --output work.json
```

The reducer validates executed shell/signature counts against public basis
metadata and recorded panels, checks generated work and counter conservation,
and retains reconstruction/source/library hashes. It groups pure s/p,
d-containing, auxiliary-f and the seven historical Rys-prototype classes
separately. Explicit `--orbital-basis-file` and `--auxiliary-basis-file` inputs
are checked against the measured snapshot hashes; their shell domains and
partial auxiliary-panel visits are reconstructed independently. The optional
Nsight SQLite export must cover the same capture: its per-class kernel launch
counts must match the component trace. Nsight kernel durations and resource
rows remain separate from CUDA event intervals. A packet shares one duration;
individual signature timing cannot be inferred from its work proportions.
The class record also identifies the selected generated schedule's component
lanes and shell tasks per block, checked against the actual kernel variant and
Nsight block size.

The [work-diagnostic rationale](../.agents/notes/implemented/performance/2026-09-16-df-shell-work-diagnostics.md)
records aggregation choices, counter semantics, measured overhead and the
conditions for reconsidering the disabled-path implementation.

With `VIBEQC_DF_TRACE` enabled, `shell_ABC_pNA_NB_NC` regions retain per-signature
stream-event intervals and their counters retain submitted tasks. Packets retain
`shell_ABC_packet` intervals; their individual
signature task counters remain available, but a shared kernel duration cannot
be attributed to each signature separately. Zero primitive
counts identify heterogeneous angular-only launches. Read the class resource
counters from **counter maxima**, since repeated launches make their sums
meaningless. `shell_resource_values_are_maxima=1` identifies per-operation
resource maxima: registers, static/dynamic shared bytes, resident block/thread limits,
and the device SM thread limit come from CUDA function/occupancy APIs. Their
ratio is theoretical occupancy, not measured achieved occupancy. Nsight kernel
activity and clean complete endpoints remain separate measurements.

`df_shell_metadata` and `df_shell_signature_dispatch` host regions identify
metadata preparation and grouped submission. `shell_signature_id_bytes` and
`shell_signature_groups` describe the added lists/ranges. Submission intervals
can include host waiting and overlap GPU work; do not add them to GPU intervals.
`signature_packet_preparation_ns` measures diagnostic host packet construction,
excluding driver submission; instrumentation contributes to this interval.
Packet counters retain submitted slices, launch count, cumulative argument bytes,
maximum packet payload, and simultaneous host view capacity. The shell signature
counters also retain ID upload bytes and host range capacity. Parameter-copy bytes
are reported separately from explicit H2D copies. Clean timings disable all traces
and counters, and include preparation and submission in the complete endpoint.

Panel-boundary revisits count as separate executions. Counts describe the shell
consumer only; the full `three_center_derivative_weights` and
`metric_derivative_weights` trace counters continue to describe the complete
response. Launch counts alone do not establish reuse or endpoint performance.

The weighted-execution selector is recorded in prepared replay metadata, so a
route change cannot silently reuse a result from a different execution policy.
CPU emission tests compare all 64 classes and coincident/asymmetric centers
against Libcint. GPU checks include transitions back to generic execution,
independent complete RHF/UHF Cartesian/spherical forces, existing auxiliary and
metric/subspace response cases, bounded budgets, and memory sanitization.

Two additional controls isolate the measured response bottlenecks. Response
algebra is intentionally independent of resident/source-backed/streamed storage:
production defaults to the compiler-qualified BLAS contractions, while scalar
execution is retained only as an explicit diagnostic/ablation route.

| Control | Production default | Explicit alternative |
| --- | --- | --- |
| `VIBEQC_DF_RESPONSE_ALGEBRA` | `blas`: parallel charge GEMV and density GEMM | `scalar`: diagnostic/ablation only |
| `VIBEQC_DF_RAW_STAGING` | workload-selected (`pageable` outside promoted staging) | `pinned-panels`: two bounded host panels |

All four selectors override their respective defaults independently. To request
the complete original comparison route, set `VIBEQC_DF_WEIGHTED_EXECUTION=generic`,
`VIBEQC_DF_SHELL_SCHEDULE=warp`, `VIBEQC_DF_RESPONSE_ALGEBRA=scalar` and
`VIBEQC_DF_RAW_STAGING=pageable`. Unset selectors are resolved from the native
source, device and response dimensions; prepared provenance retains the explicit
environment controls and the native source identity.

BLAS execution borrows the existing host-scalar handle and its owning stream.
Transpose flags preserve the original row-major contractions. Terms with
exactly zero exchange coefficients require no exchange products. The spectral
metric reverse map, including retained/discarded subspace motion, is unchanged.

Pinned staging gathers up to 16 auxiliary columns into each of two host panels.
Padded column strides avoid cache-set aliasing during the transpose. Bounded
groups of 128 AO-pair rows reuse source cache lines while writing each output
column contiguously, without allocating another copy buffer. Each
panel's last queued H2D read records an event; the CPU waits for that event before
recycling the panel. The response arena drains before either panel is freed,
including exceptional exits. Actual allocated host bytes are charged to the
host budget independently of the existing device response tile. Source-backed
values keep their device generation route and allocate no host staging panels.
Pinned panels cannot be combined with the original drain/packed upload probes.

Host-gather, event-synchronization and contiguous-copy trace regions distinguish
these operations. The original inclusive upload interval must still not be
interpreted as pure packing or transfer time. The pinned-panel controls do not
reduce the mathematical response work or reconstruct raw values from a
truncated transformed tensor.

Benchmark with `benchmarks.issue308_response_timeline`, retaining exact source
and library identity and the independent reference arrays. Use separate traced
qualification and clean complete-force timings, and compare both 192/384 AO and
batch 1/4. Repeat `--candidate` to sweep named consumers on one fixed post-cold
density. Each candidate primes outside timing, and the runner rejects changed
SCF branches, missing warm starts, numerical parity failures, and missing
selected-consumer counters in traced qualification. Every GPU invocation must
run in a finite Slurm `main` allocation with
`--gres=gpu:5090:1`; preserve the assigned device visibility.

The [resident DF dataflow note](../.agents/notes/implemented/performance/2026-09-16-resident-df-dataflow.md)
retains isolated arithmetic/cache ablations, the comparison with GPU4PySCF's
Rys consumer, unchanged scientific work counters, and binary/resource costs.

## Rys derivative selection

Mathematical lowering comes from the generated architecture/class manifest,
without an AO-count or equal-auxiliary-dimension whitelist. The qualified sm_120
profile selects Rys/compact for 18 s/p/d entries and the five auxiliary-f
classes `003/103/113/203/213`. The latter use the existing cooperative shared-axis
IR and qualified three/four-root evaluators. Missing targets/classes, including
five-root `223`, retain the polynomial fallback.
Additional mathematical availability does not promote an unqualified entry.
`VIBEQC_DF_SHELL_POLICY=legacy` forces the fallback; `candidate` admits candidate
manifest entries for complete endpoint qualification. Consumer, packet and
weight-layout admission remain separate. See [DF tuning](df_tuning.md) and the
[admission evidence](../.agents/notes/implemented/performance/2026-09-19-df-rys-admission.md).

For the original seven low-angular classes `000/001/002/100/101/110/200`, the
node convention is `u=t²`. SSS uses one node `F1(T)/F0(T)` and weight `F0(T)`;
the other six classes share a two-root evaluator with a piecewise Chebyshev table
and an asymptotic branch at `T >= 48`. Gaussian product geometry and moment IR
are shared with the polynomial emitter. Raised/lowered moments produce six
orbital-center derivatives, immediately contracted with folded response weights;
the existing translation identity recovers the auxiliary derivative. FP64,
normalization, shell/primitive scheduling, panel ownership and gradient
scattering remain common to both paths. Root values and reconstructed moments
have independent 75-digit incomplete-gamma checks, including branch boundaries
and extreme arguments. Contracted derivatives use independent libcint and
high-precision differentiation oracles.

The usual shell resource diagnostics include `shell_000_rys_selected` and
per-class Rys evaluation/root/recurrence counts. Polynomial axis-cache and
convolution counts become zero for selected Rys primitives. Recurrence counts
include shared-axis work once per active primitive plus the work of active
nonzero folded components. Clean endpoint
timing must omit diagnostic counters and compare an identical SCF workload.

The [Rys family note](../.agents/notes/implemented/numerics/2026-09-16-batch-df-rys.md)
records evaluator decisions and qualification boundaries. The
[000 Rys qualification note](../.agents/notes/implemented/performance/2026-09-16-000-rys-qualification.md)
retains historical endpoint evidence, and the
[selector retirement note](../.agents/notes/implemented/compatibility/2026-09-16-df-math-selector-retirement.md)
explains compatibility for checkpoints recording the former SSS override.

The [auxiliary-f qualification note](../.agents/notes/implemented/performance/2026-09-19-df-auxiliary-f-rys.md)
records the independent mathematics, complete endpoints and five-root fallback.
For practical paired qualification, `benchmarks.df_policy_endpoint` accepts
`--cpu-reference`, `--orbital-basis-file` and `--auxiliary-basis-file`. The fresh
reference includes auxiliary-basis response; explicit practical inputs retain
the 3e-11 Eh / 3e-11 Eh/Bohr gates for initialization, priming and measurements.
`--shell-work --components-after` captures a separate intrusive ledger after
all clean samples, never inside the promoted timing intervals.

# DF tuning, screening and final-projection reuse

The compiler owns value/derivative mathematics and class schedules. The native
runtime owns allocation, primitive traversal, streams and validated SCF state.
`integral/df_tuning` shares the Direct J/K CUDA compiler adapter, resource parser,
occupancy estimate and finite process execution. Benchmark code may load CPU or
libcint oracles; source generation never imports them or probes a GPU.

## Batch qualification

`tools/benchmark_df_derivatives.py` enumerates every available lowering for
`000 001 002 100 101 110 200`, crossed with warp/packed/compact schedules. Both polynomial and Rys
lowerings cover all seven classes, giving 42 candidates. Each independent CUDA
translation unit instantiates the production shell template; it does not copy
the recurrence. The Rys node convention is `u=t²`: one root for `000`, two for the six
other classes, restricted to ordinary full-range first derivatives. All six
share one generated degree-17 Chebyshev evaluator and its analytic large-T
limit. Source generation needs no high-precision library; the offline table
reproducer is `tools/generate_df_rys2_table.py`. Availability does not qualify
a production choice. The [numerical decision note](../../.agents/notes/implemented/numerics/2026-09-16-batch-df-rys.md) records the evaluator and oracle boundaries.
The linked executable checks full/symmetric/packed and
Cartesian/spherical fixtures against native CPU derivatives, then measures each
real primitive signature in the retained 384/768 work ledgers.

`tools/benchmark_df_values.py` enumerates generic Rys, specialized polynomial,
and specialized Rys values for ten ordered classes of total angular degree at
most two, with 1/4/32 primitive lanes: 90 candidates. It samples eight real shell
triples per primitive signature, retains actual class frequencies, and validates
every component against libcint. Weighted timings are stratified estimates;
they do not represent complete raw export or source-backed consumers.

Both tools compile candidates concurrently and execute all eligible objects in
one finite Slurm allocation. They preserve compile errors/timeouts, resources,
code sizes, numerical errors, all timing samples, and generator/target/toolchain
identity. Missing samples, changed work, failed numerical gates or resource
rejections prevent selection. Rankings remain separate for 384 and 768; a
conflict cannot silently become a combined winner. Neither tool writes the
production manifest. Hardware occupancy is not inferred from theoretical
resource occupancy.

Example from a configured checkout (set the Python environment first):

```bash
python tools/benchmark_df_derivatives.py \
  --profile benchmarks/results/issue395-df-work/work/384.json --pair-mode symmetric \
  --profile benchmarks/results/issue395-df-work/work/768.json --pair-mode packed \
  --directory .artifacts/df-derivatives --nvcc /path/to/cuda/bin/nvcc \
  --generated build/cuda-release-sm120/generated \
  --oracle-library build/cpu/libvibeqc.so --compile-jobs 2
python tools/benchmark_df_values.py \
  --directory .artifacts/df-values --nvcc /path/to/cuda/bin/nvcc \
  --generated build/cuda-release-sm120/generated --compile-jobs 2
```

## Current selectors

| Control | Automatic behavior | Diagnostic overrides |
| --- | --- | --- |
| `VIBEQC_DF_SHELL_POLICY` | Qualified architecture/class mapping, without an AO-size or auxiliary-equality filter | `legacy`, `candidate` |
| `VIBEQC_DF_SHELL_SCHEDULE` | Class-specific manifest schedule | `warp`, `packed`, `compact` |
| `VIBEQC_DF_VALUE_MATH` | Existing generic Rys | `generic`, `polynomial`, `rys`, `candidate` |
| `VIBEQC_DF_VALUE_RAW_MAPPING` | Existing scalar raw export | `scalar`, `subgroup`, `warp`, `candidate` |
| `VIBEQC_DF_FORCE_SCREEN_ABS` | Off | Nonnegative finite absolute force budget, or `off` |
| `VIBEQC_DF_FINAL_PROJECTION` | Reuse under the shared resident RHF work/capacity policy | `off`, `reuse` |

The qualified sm_120 derivative manifest chooses cooperative Rys/compact for
its 18 s/p/d entries and auxiliary-f `003/103/113/203/213`. These five
classes require only the qualified three/four-root DF quadrature. The five-root
`223` class and missing architectures/classes retain polynomial execution;
availability alone never qualifies a new entry. Lowering selection does not
change the separate consumer, weight-layout or primitive-packet admission rules.
See the [cooperative lowering qualification](../../.agents/notes/implemented/performance/2026-09-17-cooperative-df-rys.md)
and [dimension-independent admission](../../.agents/notes/implemented/performance/2026-09-19-df-rys-admission.md) notes.
Value candidates remain unqualified after complete cold
endpoint regressions. The raw `candidate` schedule applies its generated lane
count only to total angular degree at most two; metric and higher classes retain
scalar work. Source-backed value math is frozen when its owner is constructed;
its existing source schedule remains independent. All these controls participate
in checkpoint scheduling identity as optional extensions, preserving older
checkpoint compatibility.

An unqualified derivative campaign may embed a qualified `baseline` profile for
each architecture. `auto` retains that qualified baseline; `candidate` selects
the proposed class mapping. Both arms share one
prepared state and library, so the endpoint runner can interleave identical
frozen-density replays without repeating large initialization or holding two
DF arenas. Both profiles undergo the same mathematical/evidence validation.
Promoted manifests omit the campaign baseline. See the
[campaign baseline note](../../.agents/notes/implemented/performance/2026-09-17-df-campaign-baseline.md).

Mathematical lowering is selected only through the class manifest. Historical
checkpoints may retain retired controls as source provenance; importing their
density requires `allow_warm=True`, and re-export preserves those controls.
See the [selector retirement note](../../.agents/notes/implemented/compatibility/2026-09-16-df-math-selector-retirement.md).

## Derivative consumer admission

`src/scf/df_derivative_policy.hpp` owns the initial conservative sm_120 work
profile. This is an empirical scheduling envelope, not a universal latency
prediction or a mathematical capability declaration:

- Full shell execution requires estimated public weight work `N_AO² * N_aux`
  of at least `2^18`.
- Signature packets additionally require estimated ordered primitive work
  `P_orbital² * P_auxiliary` of at least `2^22`, with contraction-length
  variation within an angular class. `P` sums primitive counts over shells.

Comparisons use overflow-safe ceiling divisions. Unknown architectures and
smaller work retain the generic/angular-only alternatives. The existing
source, metric, derivative schedule, representation, response-state and arena
checks still decide correctness eligibility. The profile does not authorize
new mathematical kernels, change precision or screening, or bypass allocations.
Generated architecture/class manifests remain the separate lowering authority.

The shell and packet rules generalize across AO/auxiliary ratios and non-water
fixtures without storing a molecular histogram. They do not replace the separate
packed occupied-response layout selector. That remaining selector and wider
profile qualification stay under #444/#445; no universal cross-device speedup
is implied.

Schedule lookups query only CUDA compute-capability attributes through
`runtime/cuda_architecture.hpp`. They do not fetch the complete device property
record per auxiliary panel. The helper neither caches device ordinals nor masks
runtime errors; full product identity is queried only where a separate retained
compatibility boundary still requires it.

The [decision note](../../.agents/notes/implemented/performance/2026-09-19-df-work-admission.md)
and [evidence](../../benchmarks/results/issue445-df-work-admission/README.md)
retain small-system losses, holdouts, rejected query overhead, numerical gates
and the measured domain. `benchmarks.df_admission_probe` compares existing
controls against an independent full-force reference; it never edits this
profile or promotes a result automatically.

## Force screening contract

Screening applies only to weighted SSS three-center derivatives. Higher classes,
metric derivatives and unsupported traversal paths retain strict evaluation.
The compiler derives a triangle bound from the same six-channel SSS derivative
IR using `F0(T) <= 1` and `F1(T)/F0(T) <= 1/3` for `T >= 0`. Absolute normalized
primitive coefficients and the actual folded response weight enter the bound.
The auxiliary-center channel is recovered by translation. The sum of the six
orbital-center magnitudes also bounds any shared-atom accumulation.

Each primitive receives a budget of
`target / ((sum orbital-s primitive counts)^2 * sum auxiliary-s primitive counts)`,
rounded toward zero. The ordered domain overcounts symmetric/packed work;
an s auxiliary component belongs to exactly one panel. A factor of two provides
FP64 rounding margin. This analytical construction and its tested finite
numerical domain are not a universal floating-point certificate or a bound on
SCF/metric errors. Independent extreme-exponent and small-value/large-force
fixtures, translation checks and two-step energy finite differences protect
the intended observable.

With counters enabled, considered/skipped/executed SSS primitive counts and
fully skipped shell tasks are reported separately. Shell visits come from the
existing per-class/signature work ledger. Decisions occur before expensive
Boys/Rys evaluation, within existing packets; no global queue is allocated.
Preparation is included in force timing, and decisions in derivative-kernel
timing. Default-off specialization removes the decision branch and extra live
state. Screening remains opt-in because measured endpoint gains are small and
do not justify a universal automatic threshold.

## Final physical K to force handoff

Let `A[P,mu,nu]` be raw three-center integrals and `M` the auxiliary metric.
The resident tensor is `B = A M^(-1/2)`. Final physical occupied K produces
`U[Q,mu,i] = sum_nu B[Q,mu,nu] C[nu,i]` in existing J/K scratch.
Force response consumes `G_raw[P,i,j] = C^T A[P] C` to form metric and packed
derivative weights. For full-rank `M`, exactly:

```text
G_white = C^T U
G_raw   = G_white M^(1/2)
```

The handoff forms these small occupied-space products, then uses the existing
metric Frechet and derivative-weight algebra. When the forward cutoff discards
metric directions, `M^(-1/2) M^(1/2)` is a projector, so this reconstruction would
lose required discarded-direction response. Such plans always retain the exact
raw projection path. Rank-crossing rejection is unchanged.

An exclusive lease binds scratch to the validated final-state token: solve
epoch, density/factor generation, occupations, model/basis/geometry, metric,
device/provider/layout and math policy. Every scratch writer and new solve
revokes it. Only successfully drained final physical RHF K publishes it; force
consumes it once. Source owner identity, full metric rank, buffer capacity and
singleton RHF eligibility must all match. UHF, batches, source-backed or
constrained plans and stale tokens fall back. No extra full-sized U allocation
or D2D copy is introduced. Automatic reuse additionally requires the shared
[occupied work/capacity policy](df_occupied_cuda.md). This policy is independent
of the derivative schedule/packet tuning tables and uses no endpoint or GPU
product-name whitelist.

Qualification and rejected alternatives are retained in the
[decision note](../../.agents/notes/implemented/performance/2026-09-16-df-tuning-and-projection.md)
and [evidence bundle](../../benchmarks/results/issue404-407-df/README.md).

The [auxiliary-f qualification](../../.agents/notes/implemented/performance/2026-09-19-df-auxiliary-f-rys.md)
adds these classes without changing the separate workload-based consumer or
primitive-packet policy. Candidate availability is not itself a promotion;
independent derivative and full-endpoint evidence remains mandatory.

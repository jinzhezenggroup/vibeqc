# DF tuning, screening and final-projection reuse

The compiler owns value/derivative mathematics and class schedules. The native
runtime owns allocation, primitive traversal, streams and validated SCF state.
`integral/df_tuning` shares the Direct J/K CUDA compiler adapter, resource parser,
occupancy estimate and finite process execution. Benchmark code may load CPU or
libcint oracles; source generation never imports them or probes a GPU.

## Batch qualification

`tools/benchmark_df_derivatives.py` enumerates every available lowering for
`000 001 002 100 101 110 200`, crossed with warp/packed/compact schedules. Only
000 currently has Rys derivatives, giving 24 candidates. Each independent CUDA
translation unit instantiates the production shell template; it does not copy
the recurrence. The linked executable checks full/symmetric/packed and
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
| `VIBEQC_DF_SHELL_POLICY` | Generated sm_120 mapping for 384/768 AO with equal auxiliary dimension; other sizes retain legacy | `legacy`, `candidate` |
| `VIBEQC_DF_SHELL_SCHEDULE` | Class-specific manifest schedule | `warp`, `packed`, `compact` |
| `VIBEQC_DF_SHELL_MATH_000` | Manifest choice in its qualified domain; polynomial elsewhere | `polynomial`, `rys` |
| `VIBEQC_DF_VALUE_MATH` | Existing generic Rys | `generic`, `polynomial`, `rys`, `candidate` |
| `VIBEQC_DF_VALUE_RAW_MAPPING` | Existing scalar raw export | `scalar`, `subgroup`, `warp`, `candidate` |
| `VIBEQC_DF_FORCE_SCREEN_ABS` | Off | Nonnegative finite absolute force budget, or `off` |
| `VIBEQC_DF_FINAL_PROJECTION` | Reuse only in the qualified resident 768-AO RHF domain | `off`, `reuse` |

The derivative manifest chooses Rys/compact for 000 and polynomial/compact for
the other six classes. Value candidates remain unqualified after complete cold
endpoint regressions. The raw `candidate` schedule applies its generated lane
count only to total angular degree at most two; metric and higher classes retain
scalar work. Source-backed value math is frozen when its owner is constructed;
its existing source schedule remains independent. All these controls participate
in checkpoint scheduling identity as optional extensions, preserving older
checkpoint compatibility.

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
or D2D copy is introduced. Automatic reuse additionally requires the existing
RTX 5090, 768-AO, 160-occupied resident response admission.

Qualification and rejected alternatives are retained in the
[decision note](../.agents/notes/implemented/performance/2026-09-16-df-tuning-and-projection.md)
and [evidence bundle](../benchmarks/results/issue404-407-df/README.md).

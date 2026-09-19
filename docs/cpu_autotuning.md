# CPU schedule autotuning

Issue #471 adds a bounded CPU tuning layer on top of the compiler-owned
Gaussian-integral DAG and the explicit CPU targets from #469/#470. It does not
introduce a second integral recurrence or a CPU-model lookup table.

## Search space

A tuning candidate is a `CpuTuneSchedule` containing:

- an explicit CPU target: generic scalar, AVX2+FMA, or AVX-512F+FMA;
- the existing `CpuScheduleIR` algebra choices:
  - materialized CSE, inline-single-use, or pressure rematerialization;
  - topological or pressure-aware ordering;
  - separate arithmetic or explicit FMA where the target permits it;
  - small-integer power lowering or native power;
- a complete-shell component tile size.

Candidate enumeration is a deterministic bounded prefix. One default candidate
from every target appears before algebra variants, so a small search limit
cannot accidentally become scalar-only. Larger shell classes also expose
smaller component tiles (for example DPS S can compare full-shell, 16-component
and 8-component call boundaries).

## Static cost model

Every candidate records a labeled static resource payload before compilation:

- FP64 vector width;
- lane utilization and tail fraction for the actual primitive-record count;
- estimated peak live scalar/vector values and live bytes;
- arithmetic/materialization/rematerialization/FMA counts;
- generated source bytes;
- estimated AoS-to-SoA traffic and total logical load/store bytes;
- estimated bounded runtime working-set bytes, including the actual AoS record
  capacity, full-shell/tile result buffers, native lane scratch and live values;
- detected L1-data/L2 sizes when the operating system exposes them, plus
  working-set fit booleans.

The `vibeqc.cpu.static-cost.v2` record includes complete-consumer numeric
storage; the evaluator's numeric reservation is also recorded after loading.
`maximum_working_set_bytes` is enforced without silently increasing it. The
bound is per concurrent endpoint, not the aggregate tuning process: compiler
metadata, loaded code, allocator overhead, and additional parallel workers are
not covered by this per-endpoint estimate.

These are estimates, not measured register counts. The portable C++ path does
not expose a stable compiler register/spill report, so the manifest explicitly
records that register/spill evidence is unavailable rather than inventing a
number.

After compilation, the row also records summed cold compile time, cold
`ctypes` load/validation time, shared-library bytes, compiler artifact keys
and the complete-shell program identity.

## Numerical and performance gates

The tuner requires a caller-supplied independent reference with an explicit
reference identity. Every runtime-compatible candidate executes the complete
shell and must pass the same FP64 value + first-derivative gate before timing.
A wrong same-shaped reference therefore rejects even the scalar baseline and
prevents any selection.

Runtime-incompatible ISA candidates are rejected before their shared libraries
are loaded. On a host without AVX-512F, AVX-512 rows remain negative evidence
rather than executable candidates.

Performance comparison uses the complete `FirstDerivativeCpuLaneShellEvaluator`
consumer, not only the isolated generated function. Baseline/candidate samples
are interleaved and require an improvement larger than both 2% and the measured
MAD-derived noise floor. If no candidate clears that gate, the generic scalar
baseline wins.

The selection identity hashes:

- mathematical IntegralIR;
- workload identity;
- independent-reference identity;
- selected target/schedule;
- complete-shell program identity;
- compiler artifact keys.

Measured wall times are evidence attached to that decision, not part of the
cache identity.

## Threading interaction

After selection, the tuner measures the baseline, selected schedule, and the
fastest numerically valid SIMD candidate across explicit worker counts. This is
diagnostic evidence: it does not assume that more CPU threads improve a SIMD
kernel.

Use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for the retained qualification
so unrelated BLAS/OpenMP pools cannot silently change the comparison.

## Retained #471 evidence

The reproducible artifact is:

`benchmarks/results/issue471-cpu-autotune.json`

It uses the complete PSSS Cartesian shell with 192 contracted primitive records
and an independent native dynamic-Jet value/derivative oracle from the fresh
CPU library.

On node3 the tuner evaluated nine numerically valid generic/AVX2 schedules and
rejected five AVX-512 schedules at the target-compatibility stage.

Representative static/compiled rows:

| Candidate | Lanes | Peak live vectors | Source | Working set | Compile | Load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generic inline | 1 | 29 | 26,239 B | 784 B | 0.337 s | 0.00091 s |
| AVX2 inline | 4 | 29 | 27,125 B | 2,200 B | 0.659 s | 0.00092 s |
| AVX2 pressure-remat | 4 | 29 | 28,772 B | 2,200 B | 0.659 s | 0.00089 s |
| AVX2 explicit FMA | 4 | 29 | 25,403 B | 2,200 B | 0.658 s | 0.00087 s |

The retained v1 manifest/table reports a historical lane-only working-set
estimate; it omitted the evaluator's AoS record buffer and result reservations.
It must not be used as complete-consumer memory-admission evidence. With the
same 192-record capacity, the corrected v2 estimate is 28,456 B for the generic
row and 29,872 B for these AVX2 rows. Those recalculated estimates still fit the
host's 32 KiB L1-data and 512 KiB L2 caches. This corrects resource accounting;
it does not replace or claim to rerun the retained timing measurements.

The complete-consumer result is intentionally different from the isolated
#469 microkernel result. #469 measured about 1.90x raw AVX2 speedup on an FDPS
component. In the #471 complete PSSS consumer, the strongest measured AVX2
candidate improved only about 1.02%, below its roughly 3.04% measured noise gate, so the tuner
**retained the generic scalar baseline**.

This negative evidence is the desired behavior: source size, vector width, and
microkernel throughput do not by themselves authorize production promotion.

For the fastest valid AVX2 schedule, measured complete-consumer throughput was
approximately:

- 1 worker: 854 systems/s;
- 2 workers: 1301 systems/s;
- 4 workers: 968 systems/s.

The two-worker case improved throughput, while four workers already regressed
slightly. Threading × SIMD interaction is therefore measured rather than
assumed.

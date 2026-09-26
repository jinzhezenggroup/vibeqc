# Generated shell-block prototype (#308)

The seven-class s/p prototype executes shared primitive work correctly, but
does **not** materially improve the complete 384-AO endpoint. It remains opt-in
with `VIBEQC_DF_WEIGHTED_EXECUTION=shell-sp`; `generic` remains the default.

Clean complete analytic-force wall time, mean of two frozen-seed warm replays:

| AOs | Batch | Generic | Shell prototype |
| --- | ---: | ---: | ---: |
| 192 | 1 | 0.810 s | 0.756 s |
| 384 | 1 | 11.441 s | 11.552 s |
| 192 | 4 | 3.238 s | 3.003 s |
| 384 | 4 | 48.728 s | 47.321 s |

These are complete batch times. Both routes take identical iteration branches:
`[3]` for batch 1, `[3,2,2,3]` for 192 AO/batch 4, and `[3,3,3,3]` for 384
AO/batch 4. The density is frozen after cold execution and an untimed replay
primes the measured route. Tracing and diagnostic counters are disabled in
these samples. No native compilation overlaps the clean Slurm measurement job.
Two samples establish prototype behavior, not a robust speedup distribution.

Every complete energy and force array passes the unchanged independent
GPU4PySCF reference gates of 1e-9 Ha and 1e-8 Ha/Bohr. Maximum force error is
below 1.1e-10 Ha/Bohr. The exact geometry/settings checks and reference hashes
are retained in each measurement. The steep 192-to-384 scaling remains, so this
evidence does not promote the prototype or close #308/#206.

Separate 384-AO component captures show the three-center CUDA-event interval
decreasing from 3.145 s to 2.786 s. These intrusive intervals include the work
between their events; they are not CUPTI kernel-only sums or promotion samples.
Both warm host traces contain zero CPU-reference eigensolves. Raw traffic stays
at 4,015,521,792 bytes over eight auxiliary panels, and the complete response
still consumes 56,623,104 three-center weights and 147,456 metric weights.

Actual device counters for the prototype show:

| Counter | Executed count |
| --- | ---: |
| Visited/nonzero shell triples | 4,139,776 |
| Nonzero public weights | 26,689,536 |
| Primitive geometry/Boys evaluations | 24,899,328 |
| Cartesian component contributions | 133,373,952 |

The counts include shell revisits at auxiliary-panel cuts. Each shared primitive
evaluation supplies several Cartesian contributions; the remaining SSS and d/f
weights stay with the generic generated consumer. Metadata adds 4,624 device
bytes, plus 40 bytes only when the five diagnostic counters are enabled. The
auxiliary tile and response traffic remain unchanged. Compiled prototype
kernels use 72–114 registers/thread, compared with the generic kernel's 255.
Register counts and reuse counts do not imply an endpoint speedup.

Validation includes 100 CPU emitted-arithmetic checks, 43 distinct GPU
derivative/complete-force cases, and four RHF/UHF Cartesian/spherical transition
memchecks with zero errors. Existing auxiliary-only-center, finite-difference,
metric/subspace, rank-crossing and bounded-budget cases retain their tolerances.
Two newly added UHF references initially reached PySCF's default 50-cycle limit.
Setting its iteration limit to 100 resolved both without changing convergence
or comparison tolerances; all four transition cases then passed on the same
frozen native library.

`measurements/` retains complete arrays and provenance. `summary.json` records
the execution counts, component intervals, sample branches and file hashes.
`reproduction/` retains the measured source patch, library/source identities,
finite Slurm script and GPU state snapshots. Full logs and the binary remain
under ignored `.artifacts/issue308-shell-block/v1`. The implementation and
control contract are described in [the shell guide](../../../docs/developer/df_shell_derivatives.md).

```text
generated capability: shell-shared weighted derivatives for seven non-SSS s/p classes
handwritten scientific CUDA LOC: +80 / -1
runtime CUDA LOC: +166 / -8
legacy production path removed: no
retained duplicate reason: generic generated execution is the explicit comparison route and covers SSS, d/f, metric and serial consumers
```

The bridge's allocation and layout adapters remain conservatively classified as
scientific method glue. There is no new handwritten derivative evaluator and no
line reclassification. Both consumers use the same compiler-owned geometry,
Boys and axis-moment definitions. The next stage must generalize schedules and
address the independently measured response reductions and staging costs.

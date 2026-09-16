# DF tuning, screening and final-projection evidence (#404–407)

The production changes select derivative 000 Rys/compact in the bounded
sm_120 384/768 domain and reuse final occupied-K scratch in the existing
RTX 5090 resident 768-AO RHF domain. Screening and value candidates remain
diagnostic. No value acceleration is claimed: two complete cold comparisons
rejected promotion, and the investigation was closed at the user's request
rather than continuing to tune a losing direction.

Final automatic-selector confirmation (`384-default.json`, `768-default.json`)
uses a separately frozen final library and couples both baseline controls
against both defaults, with five interleaved repeats per arm:

| AOs | Legacy/off median (s) | Automatic median (s) | Reduction |
| ---: | ---: | ---: | ---: |
| 384 | 0.924076 | 0.910025 | 1.52% |
| 768 | 4.059656 | 3.880987 | 4.40% |

Every final sample takes three SCF updates. Maximum independent energy/force
errors are 8.65e-12 Ha / 1.62e-10 Ha/bohr. The final diagnostic confirms no
projection reuse at 384 and one successful handoff at 768, with products
769 → 3. These combined measurements replace any estimate from adding the
individual ablation percentages below.

## Complete clean ablations

Each derivative/projection row has five interleaved samples per arm, the same
frozen density, exactly three SCF updates and strict independent GPU4PySCF
energy/force gates of 1e-9 Ha / 1e-8 Ha/bohr. Intrusive component timing and
profiling are separate. These individual ablations must not be added together.

| Change | AOs | Median before → after (seconds) | Decision |
| --- | ---: | ---: | --- |
| Derivative class policy | 384 | 0.920860 → 0.907066 | Qualify bounded default |
| Derivative class policy | 768 | 4.054524 → 3.980638 | Qualify bounded default |
| Final occupied projection | 384 | 0.923678 → 0.930817 | Fallback; no 384 promotion |
| Final occupied projection | 768 | 4.055600 → 3.970709 | Qualify bounded default |
| Screening, off → 1e-4 | 192 | 0.178733 → 0.179758 | Keep off by default |
| Screening, off → 1e-4 | 384 | 0.905205 → 0.904931 | Keep off by default |
| Screening, off → 1e-4 | 768 | 3.971840 → 3.955689 | Small gain; keep opt-in |
| Screening, extended WATER8, off → 1e-4 | 192 | 0.153761 → 0.153916 | Held-out limitation |

Screening includes off, 1e-8, 1e-6 and 1e-4, with five repeats each. Complete
raw samples, all forces, independent errors, iteration/residual diagnostics,
resource diagnostics and separate component records are in `*-404.json`,
`*-406.json`, and `*-407.json`. `timings.json` is a derived convenience table.

The retained U already occupies 754974720 bytes at 768 AO. Reuse adds no U
allocation/copy; occupied projection products fall from 769 to 3 and their
counted FLOPs from 175154135040 to 61303947264. Metric/weight/pseudo-density
components remain individually visible. At screening 1e-4, 47507808 of
55656960 SSS primitives and 4184584 shell tasks are skipped; higher classes
remain strict. Independent endpoint force errors stay below 3e-10.

## Negative value result

`rejected-value-v2.json` retains the interrupted first combined candidate:
global subgroup scheduling and inlined specialized evaluators regressed the
384-AO cold endpoint. `rejected-value-v4.json` retains the complete revised
smoke, with evaluator call boundaries and low-class-only subgroup scheduling.
Cold time still increased from 10.840220 to 11.600897 seconds; changed-geometry
time increased from 9.867199 to 11.800585 seconds despite fewer updates
(12 versus 9). Cold branches also differ (20 versus 18 updates). This one-sample
smoke rejects promotion; it is not a precise regression estimate or a matched
SCF performance claim. No 768/source-backed performance qualification or
automatic value promotion is claimed.

The final batch changes some specialized Rys preferences back to generic,
showing why the original isolated estimates were insufficient. The frozen
diagnostic manifest records the original mapping used for the rejected
complete candidate; it remains `qualified:false`. Neither this follow-up
ranking nor the original ranking changes defaults automatically.

## Candidate evidence and reproducibility

- `derivative-batch.json` plus its two `*-results.json` files retain 24/24
  eligible compilations and 612/612 passing numerical rows; maximum error
  3.47e-17. The rerun reproduces all seven selected class choices.
- `value-batch.json` plus its two result files retain 90/90 eligible
  compilations and 2592/2592 passing rows; maximum error 1.90e-12. The original
  mapping's evidence uses the `initial-value-` prefix.
- Compilation metadata retains duration, return status/timeouts, parsed
  registers/stack/shared/spills, theoretical occupancy, source/object bytes,
  generator/toolchain/target identity, and raw-stream hashes. Raw streams,
  binaries and Nsight databases remain outside Git.
- Value scores use real frequency-weighted stratified shell samples, not full
  endpoint timing. 384/768 rankings and conflicts remain separate. Derivative
  measurements use every retained signature and independent CPU derivative
  fixtures across public representations and pair layouts.

`reproduction/*-source.patch.gz` reconstructs measured source against
`e41c1dcd448df9cb66775dde617ceb42cf1570c4` using `gzip -dc FILE | git apply`.
Both compressed and exact uncompressed hashes are in `source-snapshots.json`.
Qualification uses frozen library
`ae09e7a2fb51edd9de67aef6584534a310a530ea1f3cd602b1ca2a89937e8821`;
the revised value smoke uses
`210ab54e50d47d78b5eff2def82dffdd884c9131071602382addf7acf7bc7e30`.
Later final-selector confirmation records its own source/library hashes.
Library/module growth is reported separately from tensor storage: generated
diagnostic families add code even when their automatic policy stays disabled.

Use the commands in [DF tuning](../../../docs/df_tuning.md) for candidate batches.
For clean endpoints use `benchmarks.df_policy_endpoint` with the retained
scientific settings, `--repeats 5 --expected-iterations 3 --components-after`,
`--control VIBEQC_DF_SHELL_POLICY --policies legacy candidate` or
`--control VIBEQC_DF_FINAL_PROJECTION --policies off reuse`. Frozen checkpoint
hashes and starting-density hashes are recorded in every endpoint JSON; the
original local checkpoint paths are provenance, not portable dependencies.
To reproduce on a fresh machine, create a checkpoint with
`--warm-checkpoint-out` from a strict cold solve, then give the same checkpoint
to both arms and retain the newly observed identity/branch. Do not describe a
different regenerated seed as byte-identical to these measurements.

Every real GPU command must run through a finite allocation, for example:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 python -m benchmarks.df_policy_endpoint ...
```

## Transfers, synchronization and simultaneous memory

The four captures in `profiles/` compare legacy/off and candidate/reuse with
identical three-update work. Each pair has identical transfer and synchronization
counts:

| AOs | H2D bytes | D2H bytes | D2D bytes | Stream / event synchronizations | Sampled device high-water bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 384 | 7130292 | 5902575 | 0 | 7 / 432 | 8573157376 |
| 768 | 28416308 | 28326391 | 4718592 | 16 / 343 | 21338521600 |

768 also has one context synchronization per capture. The existing D2D copy is
unchanged; projection reuse introduces none. The high-water mark samples the
actual process every 100 ms, including opaque CUDA/module retention, and is a
lower bound rather than an exact allocation peak. Native resource estimates,
separate process-resident observations and component counters are retained in
the endpoint JSON. `reproduction/reduce_profiles.py` documents the reduction.

Final-library regressions pass 144 GPU Python and five GPU native tests,
including stale/failed solve lease revocation. See `validation.json` for
test/sanitizer evidence and `ownership.json` for the
reviewed native CUDA delta. Mathematical assumptions, exact fallbacks and
decisions are in the [note](../../../.agents/notes/implemented/performance/2026-09-16-df-tuning-and-projection.md).

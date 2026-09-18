# 000 Rys versus polynomial after #399

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

Only the one-root 000 first-derivative class has a generated Rys implementation.
`VIBEQC_DF_SHELL_MATH_000=rys` selects it explicitly; `auto` and `polynomial`
retain the polynomial path. No higher-angular class is implemented or promoted.

## Complete warm endpoints

Five interleaved repeats per arm, one identical frozen density per size, an
untimed prime before every sample, and exactly three strict SCF updates in every
sample. Both arms run the merged #399 exchange policies automatically.

| AOs | Polynomial median (s) | 000 Rys median (s) | Wall-time reduction |
| ---: | ---: | ---: | ---: |
| 384 | 0.922749193 | 0.907209149 | 1.68% |
| 768 | 4.059822947 | 3.984740416 | 1.85% |

These are complete energy-and-force warm calls, including SCF and final physical
state selection. They exclude initialization and priming. They are not cold
solves or energy-only measurements. At 384, the real default uses dense SCF;
this is a different baseline from #399's explicitly occupied-SCF ablation.

Separate Nsight captures, with component and work instrumentation disabled, sum
all kernels across every repeated panel:

| AOs | 000 polynomial → Rys (ms) | 000 speedup | All three-center shell kernels (ms) | GPU-time reduction |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 24.121 → 8.277 | 2.91× | 202.835 → 189.844 | 6.40% |
| 768 | 133.505 → 54.971 | 2.43× | 1296.892 → 1231.666 | 5.03% |

These are one diagnostic CUDA-activity window per arm/size, not the five clean
endpoint samples. All three-center work in this domain uses shell kernels;
the activity sum excludes the host gaps included in component-event intervals.
`kernel-comparison.json` retains all 18 angular classes' times and launch counts.

The 000 lowering has a measurable but modest complete-endpoint benefit. Kernel
savings do not justify a claim that the whole force calculation is several times
faster. This slice retains explicit selection and stops at 000; any next slice
starts with 100/001 in a separate change.

## Scientific contract

RHF, batch one, spherical def2-SVP, identical orbital and auxiliary bases,
FP64, metric relative cutoff `1e-10`, screening `1e-12`, energy convergence
`1e-12`, density convergence `1e-10`, and maximum 100 updates. The independent
GPU4PySCF reference must match geometry, method, representation, basis
fingerprints and settings before applying the unchanged `1e-9` Ha / `1e-8`
Ha/bohr endpoint gates.

All paired clean energies are identical. Maximum paired force differences are
`1.137e-13` Ha/bohr at 384 and `8.527e-14` at 768. Maximum errors against the
independent reference are approximately `8.43e-11` and `1.59e-10` Ha/bohr.

One quadrature node `F1(T)/F0(T)` and weight `F0(T)` integrate the six raised
s-Gaussian orbital derivatives generated from the existing moment IR. The common
finish function recovers auxiliary-center translation. Small arguments use fixed
Taylor polynomials, ordinary arguments use erf/exp, and `T >= 40` uses analytic
asymptotics with omitted relative F1 tail below `4e-17`.

The independent 75-digit incomplete-gamma tests validate nodes, weights and
defining moments with relative error below `5e-14`, including randomized,
boundary-neighbor and extreme arguments through `1e300`. The exact emitted
arithmetic runs on both host C++ and the GPU. Independent libcint derivative
contractions cover asymmetric/coincident centers and mixed primitive lengths.
Native pure-SSS fixtures span 1–7 primitives, packet flushes and response layouts.

## Work, resources and artifact cost

Per active 000 primitive product, polynomial computes a Boys sequence through
order one, prepares three specialized axes, stores nine cache coefficients and
executes six coefficient-convolution iterations. Rys evaluates one root/weight
and produces six direct moment states, with zero axis-cache values or convolution
iterations. Generic `axis_polynomial` calls were already zero for polynomial 000;
the eliminated axis work is its specialized preparation.

Both paths retain geometry, normalization, response folding, shell and primitive
scheduling, panel ownership, and gradient scattering. The detailed work reducer
independently reconstructs the host shell/signature domain and rejects differences
in shell visits, active primitives, Cartesian components, weights and panels.

| AOs | Shell visits | Primitive products | Active component products | 000 products selected as Rys | Derivative panels |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 384 | 3,557,376 | 21,976,064 | 122,438,784 | 7,002,240 | 1 |
| 768 | 28,385,280 | 175,132,672 | 973,859,328 | 55,656,960 | 13 |

All work-domain columns are equal between arms; only the last class-selection
count changes from zero to the displayed value. Basic component diagnostics,
separate from both clean endpoints and detailed work, sum
`three_center_derivative_contraction` to **200.30 → 191.54 ms** at 384 and
**1311.28 → 1225.18 ms** at 768. These event intervals include launch gaps.

000 registers/thread decrease **132 → 107**, static shared memory **8960 → 6656
bytes**, and the theoretical resident-thread limit **384/1536 → 512/1536**, or
**25% → 33.3%**. These are resource limits, not achieved hardware occupancy.
No persistent GPU buffer is added.

The sm_120 library grows from **205,371,680 to 205,785,296 bytes**, an increase of
**413,616 bytes (0.201%)**. Six additional 000 kernel entries cover the existing
three schedules in panel/packet forms. Their combined SASS size is 402,560 bytes;
`code-size.json` records each complete entry, including inlined libdevice and
diagnostic code. Handwritten scientific CUDA grows by **0 lines** and runtime
CUDA by **42 lines** against merged #399.

The executed compact 000 packet entry shrinks **83,840 → 66,176 SASS bytes**.
Nsight records no per-thread local memory for either 000 entry. The 100 ms device
memory sampler observes the same warm peak in both arms: **21,304,967,168 bytes**
at 768 and **8,573,157,376 bytes** at 384. The shared process runs 768 first, so
384 residency includes retained runtime/module caches. The initialization/warm
sampled high-water is **24,584,912,896 bytes**; these are lower-bound samples,
not allocator-exact peaks. The retained `time -v` report describes the Nsight
launcher, not Python application RSS.

Both arms also retain identical transfers and stream synchronization counts:

| AOs | H2D bytes | D2H bytes | D2D bytes | Stream synchronizations |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 7,130,292 | 5,902,575 | 0 | 7 |
| 768 | 28,416,308 | 28,326,391 | 4,718,592 | 16 |

## Evidence and reproduction

- `timings.json` and `measurements/` retain all clean samples, complete numerical
  outputs, convergence records, frozen density identity and source/library hashes.
  Detailed-work timings are explicitly marked intrusive and are not headline results.
- `components/` contains separate basic resource/component passes. Event intervals
  include host launch gaps; sum all repeated derivative panels, not only the first.
- `work/` and `work-comparison.json` retain every class/signature's independently
  checked source-operation counts. They are not hardware instruction counts.
- `profiles/` contains separate Nsight CUDA-activity sums and memory sampling.
  These are different calls from the work traces; no cross-call event join is made.
- `code-size.json`, `ownership.json` and `validation.json` bind binary/resource
  costs and verification to the measured artifacts. Logs, libraries, checkpoints
  and profiler databases remain outside Git.

Base: merged #399, `a90973d7aa43776348e0cdcb51927db01bab5cb4`. The library uses the
production `cuda-release-sm120` preset, fast compilation off, CUDA 12.9.1 and one
host numerical thread. Polynomial scalar/geometry emission remains byte-identical
with SHA-256 `77ca66e9fcef8892fa706e2cd7a1f07708906bf2ac8d5dbde3597a2dee9bb1dd`.

The frozen 768 density hash is
`09b2516e07b8483b47ac9607b9c30abc3485d2dfe7da92ca8f541017614738d9`;
384 is `90c4beddd055f1ffe2c600298ef060dc6ad286cbc17d04a909d0698006177e47`.
The recorded checkpoints come from the retained issues388–391 ablation workflow.
Use those checkpoints to reproduce fixed-work samples; a new cold solve is a
different input. Exact drivers and local input paths are retained in `reproduction/`.

Freeze the library as `.artifacts/issue394-000/v1/libvibeqc.so` alongside its
`source.patch` and a `libvibeqc.so.0` symlink. Copy the recorded drivers into
`.artifacts/issue394-000/`, then run:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:40:00 bash .artifacts/issue394-000/qualify.sh
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 bash .artifacts/issue394-000/profile.sh
```

Preserve Slurm's device visibility. Do not rebuild, run other GPU jobs or enable
component/work counters during clean timing. Run the retained collectors from
the repository root with `PYTHONPATH=python:.` after qualification; the shell
ledger additionally uses the matching generated header. The decision and future
scope are recorded in the
[Agent Note](../../../.agents/notes/implemented/performance/2026-09-16-000-rys-qualification.md).

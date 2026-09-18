# DF primitive-signature scheduling after #383

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

Bounded signature packets reduce the complete clean warm energy-and-force
endpoint by **6.8% at 384 AOs and 12.1% at 768 AOs** against the refreshed exact
#383 baseline, with unchanged numerical work. This is a VibeQC before/after
claim. It is not a matched GPU4PySCF completion claim for #206.

Baseline: `422796f04553953a65e45bf1f0e41642f482a591`. The original dirty
prototype was inspected in `vibeqc-issue382`, then copied into an isolated
worktree because that checkout had another active editing session. Nothing in
that original checkout was reset or committed. The inherited full-mode
same-signature triangle bug was corrected before qualification.

## Clean complete endpoint

The retained WATER27 family uses neutral singlet RHF, spherical def2-SVP in
both orbital and auxiliary spaces, FP64, relative metric cutoff `1e-10`,
screening `1e-12`, energy tolerance `1e-12`, density RMS tolerance `1e-10`,
and at most 100 SCF iterations. All 384/768 samples take three iterations.
The basis fingerprints, full geometries, metric diagnostics and final physical
residuals are stored with every clean record. Independent energy/force gates
are `1e-9 Ha` / `1e-8 Ha/Bohr`; tighter retained small-case gates also pass.

Each policy starts from the same frozen engine-local post-cold density. The
runner alternates policy order, primes every transition outside the timer,
then measures the entire energy-and-force call, including signature metadata
construction, parameter submission, transfers and synchronization. Priming
time is retained. Clean samples have no trace or work counters; components
and memory sampling run separately. Compilation never overlaps timings.

| AO | Exact #383 refreshed | Final `off` | Final `auto` | Saving vs #383 |
|---:|---:|---:|---:|---:|
| 384 | 2.8373781401 s | 2.8301425790 s | 2.6438998189 s | 6.82% |
| 768 | 7.3130372302 s | 7.3454994122 s | 6.4286604410 s | 12.09% |

All five raw samples, seconds (full precision remains in JSON):

- 384 baseline-refresh/auto: `[2.8351651551, 2.8385651682, 2.8345405390, 2.8381361160, 2.8373781401]`.
- 384 final/off: `[2.8258307180, 2.8276608621, 2.8301425790, 2.8335880919, 2.8393844620]`.
- 384 final/auto: `[2.6451623200, 2.6403458992, 2.6438998189, 2.6413935081, 2.6473617249]`.
- 768 baseline-refresh/auto: `[7.3132072741, 7.3088836230, 7.3248334071, 7.3130372302, 7.3089384069]`.
- 768 final/off: `[7.3276920610, 7.3454994122, 7.3419194110, 7.3552412461, 7.3541654111]`.
- 768 final/auto: `[6.4203073920, 6.4317989340, 6.4289268688, 6.4286604410, 6.4272883481]`.

The separate uniform and packet ablations are preserved without pooling runs:

| Stage / median seconds | 384 AO | 768 AO |
|---|---:|---:|
| baseline/auto | 2.8376091281 | 7.3161978922 |
| uniform/off | 2.8354614980 | 7.3095727509 |
| uniform/on | 2.9938412202 | 6.9732736349 |
| packet/off | 2.8361796821 | 7.3585682379 |
| packet/on | 2.9992165798 | 7.0100060320 |
| packet/packet | 2.6527854190 | 6.4454080961 |

Individual homogeneous launches are rejected as a universal default: they
regress 384 AO despite improving 768 AO. The historical s/p accumulate unroll
regression (7.31 → 8.48 s) was supplied by the issue author and was neither
repeated nor included in this implementation; its raw samples and provenance
are in [rejection-history.json](rejection-history.json).

Small final off/auto medians are 0.121364/0.120143 s (96 AO, two iterations)
and 0.206985/0.207013 s (192 AO, three iterations). Auto leaves both on the old
traversal. Complete cold observations in baseline-refresh are 13.135/102.265 s
at 384/768; the final run starts with `off` and records 13.341/102.233 s. Those
are single cold observations, not repeated packet cold or changed-geometry
speed claims. Metadata is reconstructed within each force call; this change
adds no geometry cache. Batch, constrained-memory and changed-state correctness
are covered by the retained test suite; broader performance qualification
remains under #206.

## Histogram, scheduling and component attribution

The exact input shell bins are:

| `(angular, primitives)` | 384 shells per basis | 768 shells per basis |
|---|---:|---:|
| `(0,1)` | 64 | 128 |
| `(0,3)` | 32 | 64 |
| `(0,5)` | 16 | 32 |
| `(1,1)` | 48 | 96 |
| `(1,3)` | 16 | 32 |
| `(2,1)` | 16 | 32 |

[histogram.json](histogram.json) retains every angular class, primitive
signature, task count, primitive product, min/max/CV and modeled warp/block
padding. The heterogeneous warp primitive-slot efficiency is about 46%.
This is a trip-count padding model, **not measured GPU stalls or a predicted
speedup**. Individual-signature profiles retain actual per-signature kernel
durations; packet profiles time the complete class kernel and cannot resolve
each signature's duration independently.

The implementation supplies homogeneous runtime primitive bounds to blocks
and groups up to 24 signature ranges into one kernel argument packet. It
orders ranges by descending primitive product and uses block prefixes to
select a range. There is no per-triple task list, device queue allocation,
count/scatter pass, persistent worker or extra explicit synchronization.
The existing compiler-generated derivative math owns all equations and both
entrypoints share one contraction helper. No Direct Schwarz screening is used.

Separate Nsight GPU kernel sums (milliseconds):

| AO | #383 3c / launches | Individual 3c / launches | Packet 3c / launches | Tasks | Primitive products |
|---:|---:|---:|---:|---:|---:|
| 384 | 536.790699 / 144 | 696.117355 / 987 | 344.401844 / 144 | 3,631,488 | 22,475,520 |
| 768 | 2732.621566 / 234 | 2386.576652 / 1596 | 1810.545576 / 234 | 28,385,280 | 175,132,672 |

The 192/922 ms component savings explain the roughly 193/884 ms complete
endpoint savings against the refreshed exact binary. These are independent
runs, not additive measurements from the same trace. Every Nsight class/grid
is checked against an independent host reconstruction and the device work
counters. Total endpoint kernel launches remain 16,297/8,489. Explicit CUDA
sync API counts are unchanged (384: 220 event + 7 stream; 768: 12 stream).

The final event-instrumented force-stage intervals (one sample per policy)
are 2.408 → 2.225 s and 3.656 → 2.749 s; corresponding three-center intervals
are 537.171 → 343.860 ms and 2732.832 → 1817.705 ms. Do not substitute those
intrusive samples for the clean endpoint samples above.

## Resources and bounds

| Resource | 384 AO | 768 AO |
|---|---:|---:|
| Additional signature ID uploads | 1,536 B | 3,072 B |
| Host signature range capacity | 512 B | 512 B |
| Simultaneous host signature views | 1,920 B | 1,920 B |
| Packet payload / launch | 2,888 B | 2,888 B |
| Cumulative packet-field argument payload | 415,872 B | 675,792 B |
| Signature slices / force call | 987 | 1,596 |
| Diagnostic packet construction | 0.917084 ms | 1.238278 ms |
| Full endpoint H2D #383 → packet | 4,022,650,548 → 4,022,652,084 B | 3,652,291,892 → 3,652,294,964 B |
| Full endpoint D2H, unchanged | 5,902,575 B | 28,320,179 B |
| Final auto resident process bytes | 8,451,522,560 | 21,277,704,192 |
| Sampled peak across final policies | 8,659,140,608 | 24,572,329,984 |

Construction timing excludes driver submission/backpressure and includes trace
overhead; the clean endpoint includes the whole cost. A static assertion keeps
all kernel arguments below 4 KiB. `__grid_constant__` prevents a private packet
copy per thread. Kernel parameter payload is reported separately from explicit
H2D copies. Loaded packet code adds about 24 MiB process residency; the frozen
library grows from about 287 to 352 MiB. Peaks are 100 ms process-sampler lower
bounds, not exact allocator high-water marks. Slurm accounting is disabled, so
no historical MaxRSS or billed CPU values are inferred.

Per-launch registers, static/dynamic shared memory, local memory and duration
are retained in `*-nsys-summary.json`; final occupancy API limits are in
`final/summary.json`. The standalone occupancy calculator was checked against
every uniform native result. Dynamic shared memory is zero. The table below
shows the 768-AO classes (same launch resource limits apply at 384):

| Class | Registers #383 → packet | Static shared B | Theoretical occupancy #383 → packet | GPU ms #383 → packet |
|---|---:|---:|---:|---:|
| 000 | 96 → 102 | 12800 | 41.7% → 33.3% | 304.824 → 134.462 |
| 001 | 124 → 130 | 22528 | 33.3% → 25.0% | 156.724 → 66.464 |
| 002 | 254 → 254 | 17792 | 16.7% → 16.7% | 56.448 → 30.358 |
| 100 | 130 → 138 | 18688 | 25.0% → 25.0% | 319.555 → 137.957 |
| 101 | 154 → 158 | 9088 | 25.0% → 25.0% | 335.799 → 244.942 |
| 102 | 254 → 254 | 7424 | 16.7% → 16.7% | 159.428 → 138.202 |
| 110 | 138 → 138 | 7360 | 25.0% → 25.0% | 205.365 → 130.978 |
| 111 | 126 → 128 | 7712 | 33.3% → 33.3% | 149.958 → 141.527 |
| 112 | 240 → 246 | 12896 | 16.7% → 16.7% | 92.256 → 79.344 |
| 200 | 254 → 254 | 13184 | 16.7% → 16.7% | 145.875 → 66.582 |
| 201 | 254 → 254 | 6752 | 16.7% → 16.7% | 188.550 → 144.087 |
| 202 | 240 → 240 | 11168 | 16.7% → 16.7% | 63.273 → 53.679 |
| 210 | 254 → 254 | 5408 | 16.7% → 16.7% | 175.920 → 142.871 |
| 211 | 240 → 248 | 11744 | 16.7% → 16.7% | 199.305 → 150.077 |
| 212 | 250 → 254 | 19808 | 16.7% → 16.7% | 75.658 → 68.513 |
| 220 | 240 → 240 | 8096 | 16.7% → 16.7% | 34.835 → 26.485 |
| 221 | 252 → 254 | 18080 | 16.7% → 16.7% | 51.094 → 36.288 |
| 222 | 254 → 254 | 30752 | 16.7% → 16.7% | 17.756 → 17.730 |

Nsight Compute could not access hardware counters (`ERR_NVGPUCTRPERM`). These
are **theoretical occupancy limits**, not achieved occupancy. The experiment
supports a scheduling benefit, without attributing an exact fraction to
divergence, register pressure or long-tail overlap. Historical uniform trace
resource attributes were sums; the summary normalizes by observed launch
counts. Final traces mark per-operation maxima explicitly.

## GPU4PySCF alignment

The fresh reference uses PySCF 2.14.0, GPU4PySCF 1.8.1 and CuPy 14.2.0.
Both runs use one RTX 5090, CUDA 12.9.1 and driver 580.95.05. Hardware, clocks,
power/state where captured, source hashes and versions are retained in the
raw records. [reference/conditions.json](reference/conditions.json) audits
the actual metric decomposition rather than assuming package defaults.

| Condition | VibeQC | GPU4PySCF | Interpretation |
|---|---|---|---|
| Geometry, RHF, spin/charge, spherical bases | Same retained family | Same retained family | Matched model |
| Orbital and auxiliary basis | Explicit def2-SVP | Explicit def2-SVP | Same full public ranks 384/768 |
| Metric / 3c precision | FP64 | FP64 | Matched |
| Energy stopping | `1e-12` | `1e-12` | Same numeric threshold |
| Other SCF stopping | density RMS `1e-10` | orbital gradient `1e-10` | Different criteria; residuals retained |
| Screening control in large DF runs | `1e-12` | actual DF optimizer `1e-14` | Explicit difference |
| Metric route | eigensystem, relative cutoff `1e-10` | observed successful Cholesky | Full rank in both; different algorithms |
| Unused GPU4 fallback | N/A | absolute cutoff `1e-7` | Not the executed decomposition |
| Warm seed | frozen own post-cold density | frozen own post-cold density | Same policy, not byte-identical cross-engine seeds |

The metric minimum/maximum eigenvalues are 0.093706/1699.097 (384) and
0.093048/2399.433 (768); all modes exceed the native threshold. Consequently
the full-rank DF observable matches, while the numerical policies above remain
different. No threshold or force definition was changed for the optimization.

Five host-only traced native force-stage samples have medians 2.160378 s /
2.739618 s; the fresh clean GPU4 complete force-only medians are 0.399352 s /
1.346694 s. The native values are intervals inside full endpoints, while GPU4
repeats complete gradients from one converged state, synchronized and returned
to host. These are explicitly different timing scopes, useful for locating
remaining work, not a fully matched force-only speed claim.

The initial standalone reference's complete SCF branches are `[1,1,1,1,1]`
at 384 and `[1,1,2,5,1]` at 768, versus native three-iteration endpoints.
Those complete medians are not presented as a cross-software speed ratio.
The additional unchanged Direct J/K protocol results are documented separately
below; they supersede the standalone runner for protocol comparisons.

## Same benchmark procedure as Direct J/K

At the user's request, Slurm 9685/9686 reran DF with the **unmodified**
`benchmarks/compare_gpu4pyscf_batch.py` and `real_molecule_gate.py` used by
Direct J/K. Both engines receive five ABBA-interleaved warm samples, fixed
engine-local post-cold densities, an untimed priming replay and synchronized
energy-and-force timers. The method selector is `--density-fitting cuda`.
The large comparisons preserve #383's `1e-12` screening and strict DF gates;
the small matrix retains its original `1e-14` screening and per-case criteria.

The shared-runner native before/after confirmation is:

| AO | Exact #383 | Final auto | Saving | Native iterations |
|---:|---:|---:|---:|---:|
| 384 | 2.8477298380 s | 2.6479740900 s | 7.01% | 3 / 3 |
| 768 | 7.3427257640 s | 6.4312485692 s | 12.41% | 3 / 3 |

All five native samples, seconds:

- 384-baseline.json: `[2.8486918579, 2.8477298380, 2.8463450561, 2.8509043299, 2.8470007151]`.
- 384-candidate.json: `[2.6452330898, 2.6536716190, 2.6471079472, 2.6479740900, 2.6493606542]`.
- 768-baseline.json: `[7.3079176969, 7.3342644488, 7.3427257640, 7.3453541100, 7.3615586550]`.
- 768-candidate.json: `[6.4061765210, 6.4312485692, 6.4450113522, 6.4310880110, 6.4493267662]`.

Large-case maximum errors across **all** repeats satisfy `1e-9 Ha` /
`1e-8 Ha/Bohr`. GPU4's 384-AO SCF branch is 1 in every sample; 768-AO branches
are `[2,1,1,2,4]` alongside baseline and `[1,7,4,4,1]` alongside the candidate.
No large-case branch overlaps native 3. The complete raw GPU4 samples and
component intervals remain in [direct-protocol/summary.json](direct-protocol/summary.json)
and its linked records; no unmatched median is promoted into a matched speed
ratio. This establishes **procedure identity**, not identical solver policies
or convergence branches.

The unchanged small DF matrix passes 96 batch 1 and 192 batch 1/4. Its
96 batch-4 force gate fails at `3.20874e-11` against `3e-11`; exact #383
reproduces it at `3.20914e-11`. The five paired native baseline/candidate
batch-4 forces differ by at most `5.51e-14`, with identical energies. These
failures are retained unchanged and remain an open #206 gate, alongside the
separate Direct96 failures. They do not invalidate the passing single-system
policy-ablation gates above, and are not relabeled as passes.

## Validation and remaining gates

- Two native oracle executables, 112 focused GPU Python tests, 190 CPU
  contracts/structure tests, and 18 HF resource GPU tests pass.
- Shell full/symmetric/packed, partial panels, Cartesian/spherical, RHF/UHF,
  corrected-state and generic/dense fallback coverage remains in the tests.
- Memcheck covers both native oracles and the ragged packed response test;
  synccheck covers the native shell oracle. All report zero errors.
- All final DF samples pass the independent gates, including original
  `3e-11` / `3e-11` (96 AO) and `1e-10` / `5e-10` (192 AO).
- A pre-existing HF-resource expected value was stale by 342 bytes:
  both #383 and the final library return 807,419,800, not 807,419,458.
  The test now includes the #381 sparse transform reserve
  `2*(5*47 - 8*8)`. No production resource accounting changed.
- **The original Direct96 external gates fail** at both batch 1 and 4 on
  exact #383 and the candidate. Their `3e-11` force gate is unchanged.
  Direct192 batch 1/4 passes. Paired baseline/candidate native Direct96
  forces differ by at most `7.4082e-13`, energies by `3.4107e-13`, with
  comparable native timings. A tighter GPU4 gradient-threshold diagnostic
  still fails the original force gate; batch 4 also has nonconvergence.
  [direct-audit.json](direct-audit.json) retains this limitation. This PR
  does not certify or close #206's Direct acceptance requirement.

Automatic packet selection is limited to the measured RTX 5090, full compact
shell execution, one density term, exact six-bin spherical histograms in both
bases, and the dense symmetric 384/384 or existing qualified packed occupied
768/768 response. Other domains keep the previous traversal. Explicit
`off/on/packet` are diagnostic policies and participate in replay identity.

The conservative CUDA ownership census increases by **147 scientific-glue
LOC and 281 runtime LOC**. `df_gradient_bridge.cu` retains its whole-file
scientific-glue classification; its growth is metadata, selection and dispatch.
There is no new derivative formula and no reclassification to hide that growth.
The generated derivative SHA remains
`ca1e422037afc0e9b4a9bcb05b6d5f2c6b3b739e9707703b5b950b2acb410a69`.
The packet-profile and final derivative objects are byte-identical; library,
object, source-patch and test identities are in [qualification.json](qualification.json).

## Reproduction and evidence layout

Use the committed `cuda-release-sm120` preset with the pinned CUDA 12.9.1
toolchain and the Python stacks recorded above. Build in an isolated checkout;
do not compile concurrently with GPU timings. To reconstruct a historical
candidate, create a worktree at the exact base and apply one of
`reproduction/{uniform,packet,final}-source.patch` before building. Do not apply
these patches on top of the submitted final source. The exact baseline needs
no native/Python patch. Frozen binary hashes identify the measured local builds;
fresh rebuilds may differ in provenance/path metadata.

```bash
cmake --preset cuda-release-sm120
cmake --build --preset cuda-release-sm120 --parallel 6
export PYTHONPATH=python:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/build/cuda-release-sm120/libvibeqc.so"
# Use a fresh output directory; the runner refuses to overwrite evidence.
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python -m benchmarks.df_policy_endpoint \
  --aos 768 --control VIBEQC_DF_PRIMITIVE_BUCKETS --policies off auto \
  --repeats 5 \
  --reference benchmarks/results/issue377-379-df/gpu4pyscf/water-32mer-4s4-def2-svp-spherical.json \
  --output .artifacts/issue385-reproduce/768-clean.json
```

Repeat with 384 and `water-hexadecamer-2s4-def2-svp-spherical`. For the exact
baseline, use its built library and `--control VIBEQC_DF_DERIVATIVE_PAIRS
--policies auto`. For diagnostics use fresh output names, `--trace` plus
`VIBEQC_DF_SHELL_COUNTERS=1`, or `--host-trace`. Keep those runs separate from
clean timing. Run every GPU test/profiler through finite Slurm allocations,
preserving Slurm's device visibility. The frozen scripts include the complete
Nsight capture/export commands, occupancy model and source-identity checks;
historical absolute interpreter/library paths must be replaced locally.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python benchmarks/compare_gpu4pyscf_batch.py \
  --case water-32mer-4s4-def2-svp-spherical --batch 1 --repeats 5 \
  --density-fitting cuda --max-iterations 100 \
  --energy-tolerance 1e-12 --density-tolerance 1e-10 \
  --reference-gradient-tolerance 1e-10 --screening-tolerance 1e-12 \
  --maximum-energy-error 1e-9 --maximum-force-error 1e-8 \
  --output .artifacts/issue385-reproduce/768-direct-protocol.json
```

`baseline`, `baseline-refresh`, `uniform`, `packet` and `final` preserve
separate stages. `reference` records the standalone GPU4 probes and conditions;
`direct*` records the unchanged Direct gate and diagnostic comparisons.
JSON basis metadata is losslessly interned: replace each `basis_metadata_ref`
with its `basis_metadata_catalog` entry to recover the original parsed object.
The publisher verifies that roundtrip for every record and records original
and retained SHA-256 values in [manifest.json](manifest.json). Numerical arrays,
timings, errors, branches and sample order are unchanged. Historical measured
source bytes use `.txt` to prevent formatters changing their identities.
Binaries, logs, full timelines and profiler databases remain local; reproducing
the retained numerical/performance claims does not require them.

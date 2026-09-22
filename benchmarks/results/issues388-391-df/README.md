# Resident DF dataflow, derivative algebra and DIIS (#388–#391)

> **Historical supporting data:** bulky reports from this campaign remain in
> existing Git history, with [checksum-verified snapshot recovery](../retention-checkout/README.md).
> The summary below and compact records remain here. Restore the complete
> snapshot before running historical scripts or verifying its original
> manifests; those manifests describe the original snapshot, not this reduced
> checkout. No measurements, rejected cases or acceptance thresholds changed.

The promoted changes retain raw FP64 three-center values in existing J/K
capacity, share generated derivative intermediates, batch exact occupied
response products, and parallelize DIIS residual dots. The normal warm
energy+force endpoints improve at both qualified sizes with unchanged gates.

| AOs | #386 baseline median (s) | Final median (s) | Warm SCF updates, every sample |
| --- | ---: | ---: | --- |
| 384 | 2.641318227 | 0.920400906 | 3 → 3 |
| 768 | 6.436193604 | 5.308092445 | 3 → 3 |

These are five clean repeats per implementation, each replaying its own
frozen post-cold density. Baseline and final cold densities are not assumed
identical. Cold calls are single observations: 384 takes 13.104 → 10.782 s
(20 → 20 updates), and 768 takes 101.993 → 85.514 s (21 → 24 updates).
Cold timing is not a matched-work performance claim. The isolated comparisons
below use an identical exported density and record every actual update count.
No cross-engine endpoint speed ratio is inferred.

## Qualification and retained records

Base source is `b29649f895ad53a549bef03341c2349cbe131f6c` (#386). Final runtime
patch SHA-256 is
`18e8949fbd159301ac8e27fe63c206f393f367bd2710523c5a99025066db2f06`.
The frozen library, native source identity, runner hash, input/reference hashes,
Slurm job and controls appear in each measurement. The patch reconstructs
`src/` and `python/`; this PR also contains benchmark and test changes.
A subsequent review correction records the already-allocated Jacobi solver
workspace size for `nbf <= 32`. It changes telemetry only and does not affect
the 384/768 AO solver paths or the frozen measurements below.

Measurements use RTX 5090, CUDA 12.9.86, driver 580.95.05, GCC 11.4, Release
sm_120, fast compiler mode off, one host numerical thread, spherical def2-SVP
with the same orbital and auxiliary basis, batch one, RHF, FP64, metric cutoff
`1e-10`, screening `1e-12`, energy convergence `1e-12` and density convergence
`1e-10`. Every sample passes independent retained GPU4PySCF energy/force gates
of `1e-9` Hartree and `1e-8` Hartree/bohr. Final normal maximum errors are
`3.911e-11` and `1.358e-10`, respectively. Source/model/shape validation precedes
the numerical comparison. No precision, screening, metric or force tolerance
was loosened.

- [timings.json](timings.json): raw seconds, medians, branch identities, errors
  and density residuals; complete result arrays remain under `measurements/`.
- `components/`: separate intrusive passes, executed stream counters, native
  reservations and process residency. `inclusive_regions` sums **all** repeated
  regions, including all 13 derivative panels; nested regions are not additive.
- `profiles/`: reduced Nsight activities, host wait attribution, transfer bytes,
  sampled memory, environment and unavailable-counter evidence.
- [generated-work.json](generated-work.json): symbolic derivative loop counts
  tied to the final diagnostic trace hash.
- [builds.json](builds.json): measured binary/object sizes and hashes.
- [validation.json](validation.json): validation counts and original log hashes.
- [ownership.json](ownership.json): physical CUDA additions/removals and
  unchanged lines reclassified by the reviewed semantic inventory.
- `rejected/`: incomplete or unmatched experiments and rejection reasons.
- `reproduction/`: measured runtime patches, exact run scripts and reducers.

Component and workload profile records are bundled in `384.json` and
`768.json` under their respective directories. Each bundle's `records` keys
preserve the original logical record paths. Bundling changes only storage
layout; all values and ordering within each record are retained.

The [Agent Note](../../../.agents/notes/implemented/performance/2026-09-16-resident-df-dataflow.md)
explains the ownership contract, algebra, fallback boundaries and rejected
designs. [Current documentation](../../../docs/developer/df_occupied_cuda.md) describes
the supported execution controls.

## Isolated clean comparisons

Every row below has five interleaved samples per policy and three SCF updates
in every sample. All policies receive an untimed replay at transitions.
Common cold controls are resident exchange `legacy` and DIIS dots `serial`.
The 768 density SHA-256 is
`09b2516e07b8483b47ac9607b9c30abc3485d2dfe7da92ca8f541017614738d9`.

| Change | 384 before → after (s) | 768 before → after (s) |
| --- | ---: | ---: |
| Raw reuse, `off` → `auto` | 0.955107 → 0.922374 | 5.574276 → 5.336021 |
| Response batching, `off` → `auto` | 0.921551 → 0.913283 | 5.517331 → 5.335179 |
| Resident K, `legacy` → `auto`, raw reuse off | 0.940694 → 0.949764 | 5.600309 → 5.570335 |
| DIIS, `serial` → `auto` | 0.942967 → 0.919556 | 5.703966 → 5.627464 |
| Response storage, `panel` → `jk-scratch` | 2.542280 → 0.921271 | already resident |
| Energy only, K `legacy` → `auto` | 0.453513 → 0.455179 | 3.500410 → 3.451702 |
| Energy only, DIIS `serial` → `auto` | 0.476339 → 0.455411 | 3.585720 → 3.499338 |

At 768, matched DIIS comparisons hold K at `legacy`: default new K causes
serial DIIS to take five updates versus auto's three. That separate complete
comparison is retained as an unmatched diagnostic, and the energy-only guard
rejection is retained. It is not used for the table above. Energy-only runs
enforce `--expected-iterations 3`; they retain normal convergence checks rather
than forcing acceptance after an arbitrary number of steps.

Resident K alone is slightly slower at 384. It also frees the tensor needed
by raw reuse, whose saving exceeds that cost. Full-Gram K and triangular K
have similar 768 timings; a twofold SYRK timing improvement is not claimed.
The flattened dense-K variant regresses both sizes and is retained only as
an explicit diagnostic. Do not add savings across independent rows.

## Raw reuse and exact response work

The 768 raw-reuse ablation changes bulk uploads `1 → 0`, upload bytes
`3,623,878,656 → 0`, and transpose elements `452,984,832 → 0`. Separate component
times are upload `235.329 → 0` ms, transpose `13.707 → 0` ms and force-response
GPU inclusive `1832.118 → 1578.889` ms. Complete force-stage host timing is
`2.019655 → 1.770545` s; the clean endpoint comparison is in the table above.
The retained raw allocation replaces a former scratch tensor. No second full
tensor is added to setup peak or persistent capacity. Raw values remain
untruncated so discarded metric directions contribute to the exact derivative.

For `D=w C C^T`, response forms `T_Q=C^T A_Q C`, transforms it with the same
metric inverse, then expands `C U_P C^T` into public derivative weights.
The 768 projection count `1536=2a` becomes `769=1+a`: a concatenated first
GEMM and a batched second GEMM. Submissions fall from 1536 to two while
`175,154,135,040` FLOPs remain. Pseudo-density products `3072=4a` become
`2317=13+3a`, with 52 BLAS submissions across 13 shell-aligned panels and
unchanged `126,835,752,960` FLOPs. The metric and weight GEMMs retain the
occupied-space representation and their exact spectral reverse map.

Separate 768 batching component times (ms) are:

| Component | Serial | Batched |
| --- | ---: | ---: |
| Occupied projection | 276.067 | 138.194 |
| Metric GEMM | 23.428 | 23.420 |
| Weight GEMM | 19.996 | 19.997 |
| Pseudo-density expansion | 146.483 | 81.157 |
| Derivative consumer | 1283.949 | 1282.199 |

Projection staging uses 754,974,720 bytes of existing scratch. Rectangular
outputs write 301,989,888 FP64 elements per force call (2,415,919,104 bytes),
then the packed handoff consumes 1,814,298,624 bytes. These are logical traffic
proxies, not measured DRAM transactions. Batching changes lifetime/layout and
factor reuse without approximating the mathematical products. Full response
orientation and packed symmetrization have independent RHF/UHF tests.

At 384, automatic borrowing retains dense response algebra but executes its
all-Q products once: `6144 → 768` products. It removes repeated panel staging
and some duplicate shell-boundary primitive visits, without changing screening.
Tight-capacity, batch and generated/source paths keep their explicit fallbacks.

## Generated derivative ablations

Only generated derivative mathematics differs among these three matched
libraries. Their common runtime predates the final dense-K panel carry fix and
uses the earlier flat dense formulation; do not compare their absolute times
to final normal timings as if only derivative code differed.

Two owners at 768 exceed device capacity. Sequential ABCCBA reconstruction
uses two forward and three reverse samples per variant, verifies identical
starting density hashes and retains three updates in every sample.

| AOs | Legacy endpoint (s) | Shared H (s) | Shared H + center identity (s) |
| --- | ---: | ---: | ---: |
| 384 | 1.128207 | 1.079105 | 1.065016 |
| 768 | 5.980266 | 5.593880 | 5.435645 |

The corresponding 768 derivative totals are 1823.164, 1446.670 and 1285.599 ms.
Sharing `H_i=sum_jk v_j*w_k*F_(i+j+k)` halves generated other-axis products
from 46,587,727,872 to 23,293,863,936. The raised-center identity removes an
axis-cache boundary. All variants visit 28,385,280 shell triples,
175,132,672 primitive products and 226,787,328 public weights; packet launch
count remains 234. The final normal derivative consumer is 1281.658 ms.

GPU4PySCF's corresponding fused Rys kernel is about 665 ms in the retained
[reference profile](../issue382-packed-df/reference/768-profile-summary.json).
Its root-wise products differ from VibeQC's general Hermite/Boys coefficient
convolution. Both share primitive preparation, other-axis products and a
horizontal center identity. This change removes measured duplicate work but
does not establish cross-engine parity or attribute every remaining cycle.

Matched library sizes are 203,146,168 → 198,378,424 → 198,255,544 bytes;
derivative objects are 75,736,928 → 70,971,104 → 70,850,064 bytes. The historical
baseline has line information and is excluded from this size comparison.
All three 768 process-resident snapshots are 21,237,858,304 bytes. No additional
specialization family is introduced. Registers and theoretical occupancy
have tradeoffs: SSS resident threads rise 512 → 640; PPP falls 512 → 384.
Per-class resource counters remain in component records. NCU failed with
`ERR_NVGPUCTRPERM`; achieved occupancy, stalls, branch efficiency and actual
DRAM traffic are not claimed.

## SCF attribution and memory

The new occupied K projects directly from pair-major B into `U[mu,i,Q]` and
forms a Gram product over `(i,Q)`, removing the full gather/per-Q output and
reduction traversal. The legacy resident case already used one full panel and
one transform with diagonal projection reuse. New projection GEMM dimensions
are `(m,n,k,batch)=(768,160,768,768)` and the final Gram is
`(768,768,122880)`. Two occupied builds decrease about 415.55 → 368.34 ms.
Two required dense seed/final-state K builds still take about 1.71 s; RI-J is
about 18 ms. These costs remain in complete endpoint and energy-only timings.

DIIS reuses completed residual scratch for deterministic partial dots, with
the same normalization, pivot gate and circular-history retirement. Host-only
Nsight traces hold legacy K and three updates fixed. Provider host time is
1503.300 → 1424.318 ms; three 16-byte D2H **cudaMemcpyAsync** calls account for
1499.636 → 1420.586 ms. Their actual copy work totals about 0.0013 ms. Queued
DIIS kernel overlap falls 125.583 → 43.760 ms; earlier K work also overlaps.
The unfenced 1.5-s wait must not be compared directly to the old component-fenced
203-ms interval. DIIS explains most of that earlier excess over roughly 52 ms
of eigen GPU work. Provider selection and retained workspace are unchanged.
Final-state validation remains separately visible at roughly 1.17 s.

| Final normal quantity (bytes) | 384 | 768 |
| --- | ---: | ---: |
| Native resident reservation | 1,937,915,779 | 15,002,532,751 |
| Native peak reservation | 2,398,506,527 | 18,651,054,635 |
| Post-call process residency | 8,426,356,736 | 21,252,538,368 |
| Sampled process high-water, default profile | 8,770,289,664 | 24,647,827,456 |
| Actual endpoint H2D, default profile | 7,130,292 | 28,416,308 |
| Actual endpoint D2H, default profile | 5,902,575 | 28,320,179 |

Native capacity, post-call residency and sampled process peak are different
observables. The 100-ms sampler includes opaque runtime allocations and can
miss short peaks. Its legacy-K comparison peaks are 8,770,289,664 and
24,672,993,280 bytes. No hidden growth is inferred from a component-local
budget. Separate profile windows enable CUDA event tracing and memory sampling;
they are intrusive attribution, never clean timing.

The host-tensor streamed compatibility route still builds bounded CPU panels,
uses pageable H2D and drains before reuse. It is absent from these resident
traces. Source-backed generation remains the production alternative. This PR
retains correctness coverage for these paths and makes no constrained-memory
speed claim; pinned/double-buffered compatibility staging needs its own
accounting and endpoint qualification.

## CUDA ownership accounting

The DIIS partial-dot and update kernels are method-specific scientific code;
launch wrappers remain runtime. Against `b29649f`, physical edits add/remove
172/23 scientific lines and 61/23 runtime lines. Another 151 unchanged lines
move from runtime to scientific classification. The resulting semantic deltas
are **+300 scientific** (11,723 → 12,023 including retained oracle/exception
code) and **−113 runtime** (9,686 → 9,573). Total maintained growth is 187 lines;
reclassification is not physical retirement. Derivative equations remain
compiler-owned. The `scf_tensor` ownership ledger records the generated DIIS
replacement and numerical/resource/endpoint gates needed for retirement.

Reproduce [ownership.json](ownership.json) using
`tools/compare_cuda_ownership.py` with an unchanged checkout of `b29649f` and
its `docs/cuda_ownership_current.json` as the baseline, and this checkout and
its regenerated inventory as the candidate. This accounting correction leaves
the qualified runtime source and binary unchanged.

## Reproduction

Build before scheduling clean measurements. From this PR's checkout, with the
recorded Python dependencies and CUDA 12.9 available:

```bash
cmake --preset cuda-release-sm120 -DVIBEQC_BUILD_TESTS=ON \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc
cmake --build --preset cuda-release-sm120 --parallel 4
mkdir -p .artifacts/issues388-391/final
cp build/cuda-release-sm120/libvibeqc.so .artifacts/issues388-391/final/
git diff b29649f895ad53a549bef03341c2349cbe131f6c --binary -- src python \
  > .artifacts/issues388-391/final/source.patch
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:25:00 bash \
  benchmarks/results/issues388-391-df/reproduction/final-continuation.sh
```

Run scripts preserve the original interpreter/CUDA paths; change those paths
to equivalent installations when reproducing elsewhere. Use fresh output
directories because the runner refuses to overwrite results/checkpoints.
Slurm's device visibility must remain unchanged. All real-GPU commands,
including profilers, native tests and sanitizers, run inside finite allocations.

`final-carry-ablations.sh` records job 9713's 768 comparisons and the intentional
iteration-guard stop. `final-continuation.sh` records job 9716's matched DIIS,
384 controls and normal endpoints. `profile-final.sh` records job 9715's
profiles, sanitizers and 150 CUDA Python tests. `derivative-ablations.sh`
records job 9714. Wrap each script in the same finite `srun` form; do not run
benchmark scripts alongside compilation.

To reconstruct derivative libraries, create separate clean worktrees at the
base revision, apply the corresponding `derivative-*-source.patch`, build each
with the same Release preset and no line information, and copy each library
beside its patch under `.artifacts/issues388-391/derivative-*`. Run the current
benchmark runner against those frozen libraries. Reconstruct the historical
baseline at the base revision with `CMAKE_CUDA_FLAGS=-lineinfo`. The baseline
run script and build hashes retain that distinction.

Raw logs, temporary restart checkpoints, libraries, objects and profiler
databases remain outside Git. The reviewed records preserve numerical samples
and hashes; every retained file stays below the 1 MiB limit.

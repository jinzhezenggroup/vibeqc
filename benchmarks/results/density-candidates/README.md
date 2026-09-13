# Registered D/C candidates and larger workloads (#235 / #303)

This directory retains independent current orbital states and fixed-grid
LDA/PBE XC energy/potential references for seven workloads. They span
24–192 AOs, compact larger bases, water tetramer/octamer systems, and diffuse
RKS/UKS OH inputs. PySCF 2.14.0 / Libxc 7.0.0 supplies the converged PBE
producer state and independent AO/feature/E/V values. Production never
factorizes an arbitrary D to construct these factors.

The fixed-input grid is the declared unscreened `GridSpec(2,16,32)` with all
points and weights retained. Its coarse radial rule lies inside the audited
XC domain; this is not a molecular quadrature-convergence claim. Both LDA
and PBE are evaluated on the same supplied state; only the PBE reference
producer is self-consistent. Screening differences on the water clusters
are reported separately from equal-mask arithmetic gates.

The extended runner registers the actual prepared D/C executors through
`vibeqc.autotune.dft_density_candidates`. It uses 256-point / 16-orbital
tiles, 128/256 MiB device budgets and a 256 MiB host budget. Total E/V timings
include GPU AO/features, explicit transfers, native CPU XC and full potential
assembly. A four-system measurement replays independently owned replica
states serially through one prepared owner; it makes no fused batching or
concurrent GPU execution claim. Its extra source/output storage composes
with that owner's resource requests under #203.

The companion native measurement invokes the existing LDA/PBE RKS loop on
water/def2-SVP and water/def2-TZVP, using complete `GridSpec(16,8,16)` energy-only
endpoints. Both algorithms use the same cold core guess or the same strict
warm density. PBE retains the explicit production tail-v2 LDA fallback.
Provider/grid setup is recorded separately. The invocation bridge contains
input tables and calls the native solver, with no second SCF loop or XC
formula. Full per-iteration traces and complete DFT forces are unavailable,
so these measurements do not qualify for #168's energy-plus-force promotion.

Reproduce in a clean checkout after building a matching Release CPU library:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /path/to/pinned-pyscf-python \
  tools/generate_density_workload_references.py /fresh/reference-directory

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VIBEQC_NVCC=/group/software/cuda-12.9.1/bin/nvcc \
  .venv/bin/python tools/benchmark_density_sources.py \
  --workload-matrix benchmarks/results/density-candidates/inputs \
  --library build/cpu/libvibeqc.so --cache .artifacts/fresh-candidate-cache \
  --output .artifacts/fresh-candidate-evidence --samples 5

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  tools/benchmark_native_rks_density.py \
  --workload-matrix benchmarks/results/density-candidates/inputs \
  --library build/cpu/libvibeqc.so --output .artifacts/fresh-native-evidence \
  --samples 5
```

The fixture manifest validates each archive, input contract and numeric block.
It also binds the exporter and existing Cartesian/spherical basis adapter.
`inputs_hash` identifies mathematical inputs; a fixture's `identity` is the
content hash of its complete reference snapshot, including producer provenance
and runtime. Executable candidate/workload identities use the actual source,
grid and prepared contract, and do not include producer runtime. Re-exporting
references can change the snapshot identity without changing those inputs.
Only the 192-AO octamer archive exceeds the repository's review-size threshold;
its exact-hash retention exception covers the unique scientific reference
arrays. Full molecular grid-by-AO or grid-by-orbital tensors are not retained.

## Retained results

The [GPU publication](gpu/publication.json) measures clean source
`9968ce24bab98fd62bf068c0640600192e3bdd75` under Slurm job 9412 on RTX 5090.
The [native publication](native/publication.json) measures clean source
`55ac2dc01181dfed83ca5aae5516cb50dd20f593` on AMD EPYC 7K62 with one
OpenMP/OpenBLAS thread. The latter revision only adds runtime CPU provenance;
scientific source/library identities match across both runs. All measurements
use five interleaved repeats per route, with setup outside the repeated endpoint.

| Evidence | Cases | Timing rows | Passing block gates |
| --- | ---: | ---: | ---: |
| GPU features + native XC E/V, batch 1 | 36 | 360 | Combined below |
| Four independent states, serial shared-owner replay | 12 | 120 | Combined below |
| Combined GPU record, including six independent feature fixtures | 48 | 480 | 392 |
| Complete native energy-only RKS, cold/common warm | 8 | 80 | 240 |

GPU maximum energy/potential-entry errors are `3.55e-14` / `5.33e-15`.
Native maximum energy/density-entry differences are `4.13e-13` / `3.90e-9`,
and the maximum final physical residual is `1.16e-9`. The native gates are
respectively `1e-9`, `1e-8`, and `1e-8`. Every C warm start records one original-D
fallback for the external starting density; subsequent factors describe the
current state. Cold C starts and D runs record no fallback.

Maximum composed GPU-workload capacities are 76,482,128 host bytes and
110,540,544 device bytes. Native observed numeric capacity reaches 28,331,424
host bytes. These include the documented numeric buffers and provider allowances;
they are not measured whole-process RSS or allocator peaks. Per-case plans,
source packing/upload costs, active AO distributions and screening differences
remain in the full record.

The table shows illustrative median milliseconds at the 128 MiB device budget.
All cases and five-sample spreads are in [gpu/summary.json](gpu/summary.json) and
[native/summary.json](native/summary.json); raw samples are retained with them.
A larger D/C ratio means a shorter C time in this particular measurement.

| Endpoint | D ms | C ms | D/C |
| --- | ---: | ---: | ---: |
| GPU/native XC water SVP PBE, dense | 17.409 | 19.298 | 0.902 |
| GPU/native XC CO2 TZVP PBE, dense | 24.288 | 28.205 | 0.861 |
| GPU/native XC water octamer PBE, dense | 306.090 | 513.832 | 0.596 |
| GPU/native XC water octamer PBE, local | 285.485 | 437.462 | 0.653 |
| Four serial water tetramer PBE states | 369.165 | 514.788 | 0.717 |
| Native RKS water TZVP PBE, cold | 1252.747 | 1019.645 | 1.229 |
| Native RKS water TZVP PBE, warm | 361.794 | 311.534 | 1.161 |

The bounded GPU C implementation loses on these total XC measurements even
though its feature-count estimate is lower. Native CPU C improves these
energy-only examples. Different backends/grids/scopes cannot be combined into
a universal winner, and neither publication passes a performance-promotion gate.
No complete-force, GPU SCF or native UKS endpoint is claimed. #163 retains
complete-force integration; #168 retains profile selection and promotion.

Reconstruct either summary without running hardware:

```bash
.venv/bin/python tools/summarize_density_candidates.py \
  benchmarks/results/density-candidates/gpu/evidence.json \
  --output .artifacts/gpu-summary.json
.venv/bin/python tools/summarize_density_candidates.py \
  benchmarks/results/density-candidates/native/verification.json \
  --output .artifacts/native-summary.json
```

Both publication manifests are produced by the existing `tools/evidence.py`
publisher and validated in the retained-evidence regression. The GPU envelope
also has an exact-content size exception because all timing rows, numerical
gates and resource/source identities are needed to audit the result.

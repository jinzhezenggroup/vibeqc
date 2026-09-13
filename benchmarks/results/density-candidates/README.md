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
Only the 192-AO octamer archive exceeds the repository's review-size threshold;
its exact-hash retention exception covers the unique scientific reference
arrays. Full molecular grid-by-AO or grid-by-orbital tensors are not retained.

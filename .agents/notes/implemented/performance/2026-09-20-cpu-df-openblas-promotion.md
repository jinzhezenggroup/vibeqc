# Decision: auto-enable OpenBLAS after CPU DF hot-path promotion

Status: implemented
Date: 2026-09-20

## Problem

The first OpenBLAS vertical slice accelerated isolated GEMMs but did not improve complete CPU DF energy+force endpoints because runtime Jet forward AD dominated force preparation and the main RI-K contractions still used scalar loops. The earlier endpoint audit therefore correctly retained the scalar build default at that stage.

## Decision

Supersede the scalar-default portion of `../architecture/2026-09-20-cpu-dense-linalg-provider.md` after migrating the actual CPU DF hot paths:

- lower the existing compiler-owned DF first-derivative IR to host C++ for s/p/d/f instead of carrying dynamic Jet AD through production CPU force preparation;
- use the dense-LA provider for RI-K (`B_Q D`, then `(B_Q D) B_Q^T`) from 16 AO upward;
- use the same provider for the dense exchange force-response contractions from 16 AO upward;
- preserve scalar/no-provider paths as deterministic fallbacks and independent numerical coverage;
- change `VIBEQC_CPU_LINALG_PROVIDER` build default from `scalar` to `auto` so an available qualified OpenBLAS provider is used by these explicitly provider-parallel dense regions.

The DF metric eigensolver remains on the scalar schedule for now. Matrix-function response keeps its conservative external-provider threshold of 128 pending #471 endpoint-aware tuning.

## Invariants

- CPU and CUDA DF derivatives come from the same compiler-owned scientific IR.
- No runtime automatic differentiation is needed for production s/p/d/f CPU DF first derivatives.
- Unsupported higher-angular cases retain the existing Jet fallback.
- Builds without OpenBLAS remain correct and use scalar dense-LA fallbacks.
- OpenBLAS global thread control is serialized when thread-local control is unavailable; redundant set/restore is avoided when the requested count already matches.
- Scientific SCF/DF code does not call vendor BLAS symbols directly.

## Evidence

All timings below were pinned to CPU 47 on node3 (AMD EPYC 7K62), with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. Scalar and OpenBLAS builds used identical source except for the configured dense-LA provider.

### RI-K contraction

The synthetic contraction includes the required AO-pair-major to auxiliary-major packing cost.

| AO | Aux | scalar | OpenBLAS | speedup |
|---:|---:|---:|---:|---:|
| 16 | 18 | 0.163 ms | 0.0444 ms | 3.7x |
| 24 | 26 | 0.787 ms | 0.124 ms | 6.4x |
| 43 | 45 | 5.01 ms | 0.635 ms | 7.9x |
| 64 | 66 | 27.1 ms | 5.12 ms | 5.3x |
| 96 | 98 | 154 ms | 16.0 ms | 9.7x |

### Existing-integral RHF DF gradient contraction

| AO | Aux | scalar | OpenBLAS | speedup |
|---:|---:|---:|---:|---:|
| 16 | 18 | 0.403 ms | 0.248 ms | 1.6x |
| 24 | 26 | 1.98 ms | 0.708 ms | 2.8x |
| 43 | 45 | 13.2 ms | 3.24 ms | 4.1x |
| 64 | 66 | 82.8 ms | 17.0 ms | 4.9x |

### Generated CPU DF derivatives

Replacing production dynamic Jet AD with host code generated from the existing DF derivative IR reduced the native density-fitting test runtime from roughly 0.16 s to 0.07 s in the scalar build. Compiler/derivative qualification after the host lowering: 96 passed, 244 skipped; compiler structure: 248 modules, 0 dependency errors.

### Complete CPU DF energy+force endpoints

| endpoint | scalar | OpenBLAS | speedup |
|---|---:|---:|---:|
| water/def2-SVP RHF, 5 repeats | 0.0806 s | 0.0675 s | 1.19x |
| water/def2-TZVP RHF, 3 repeats | 0.6368 s | 0.4798 s | 1.33x |
| OH/def2-SVP UHF, 5 repeats | 0.0475 s | 0.0385 s | 1.23x |
| water tetramer/def2-SVP RHF, 2 repeats | 25.96 s | 8.91 s | 2.91x |

All endpoint pairs converged with identical SCF iteration counts. Energies agree at ordinary FP64 roundoff; force norms agree to approximately 1e-11 or better in the measured cases.

For context, the pre-codegen scalar water/def2-SVP complete endpoint was about 1.17 s, so the generated CPU derivative lowering is itself the dominant improvement. OpenBLAS becomes a net endpoint win only after those dominant non-BLAS costs are removed.

## Consequences

OpenBLAS is now a justified automatic build-time provider when available, rather than a microbenchmark-only option. The measurements also clarify the next optimization frontier: larger systems remain dominated by raw three-center integral work and data layout/packing, so persistent auxiliary-major packing and further generated/batched integral scheduling should follow.

## Revisit when

- #471 can replace fixed AO-size cutoffs with workload-aware tuning;
- provider packing is retained across repeated SCF iterations;
- MKL/BLIS providers are added;
- multi-thread provider ownership is benchmarked under isolated CPU allocations.

## References

- #674
- #681
- #471

---
Agent: ChatGPT
Model: GPT-5.6 Sol


## Follow-up hotspot removal: generated CPU DF values

After derivative codegen removed the force-side Jet bottleneck, profiling exposed an
asymmetry: CPU energy-only DF still used the legacy generic value evaluator, so
water/def2-TZVP energy-only (~1.36 s) was slower than the energy+force path (~0.48 s).

The same compiler-owned DF value IR and Rys tables now have a host C++ lowering for
production s/p/d/f values. Unsupported higher-angular cases retain the legacy fallback.
Generated value/derivative headers are isolated behind `integrals/generated_df_cpu.cpp`
so ordinary `s_integrals.cpp` edits do not recompile the large generated Rys tables.

The prepared DF owner also retains an optional Q-major orthonormalized tensor when
OpenBLAS is available. RI-J, RI-K, and the metric three-center transform use the common
dense-LA boundary; direct/oracle callers retain bounded temporary/scalar fallbacks.

Pinned node3 energy-only measurements (CPU 47, one BLAS/OMP thread):

| endpoint | prior OpenBLAS | generated-value OpenBLAS | speedup |
|---|---:|---:|---:|
| water/def2-SVP RHF | 0.14371 s | 0.01863 s | 7.71x |
| water/def2-TZVP RHF | 1.36270 s | 0.11006 s | 12.38x |

The no-BLAS scalar build also improves to 0.02441 s (SVP) and 0.18011 s (TZVP),
confirming that host codegen, not BLAS alone, removes the dominant value-side cost.

Persistent Q-major reuse plus RI-J/metric-transform lowering is a smaller but consistent
follow-up: the 96-AO water-tetramer complete energy+force endpoint improved from about
8.91 s to 8.80 s, while smaller force endpoints moved only at sub-percent scale.

Local qualification completed before GitHub handoff:
- scalar/OpenBLAS DF native tests pass;
- CPU value/derivative emitter host-only checks pass (5 selected tests);
- generated-value endpoint energies retain the same SCF iteration counts and agree with
  generated-derivative endpoints at normal FP64 roundoff.

Final full-repository CI is delegated to GitHub because the node3 remote-command monthly
quota was exhausted during the last local qualification pass.

---
Agent: ChatGPT
Model: GPT-5.6 Sol

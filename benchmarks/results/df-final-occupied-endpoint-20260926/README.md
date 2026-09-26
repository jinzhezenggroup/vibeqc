# Final occupied K and retained metric-root qualification

The explicitly selected packed-single/fitted occupied RHF endpoint improves on the same-binary dense-final/spectral control. The 768-AO frozen warm median drops **26.8%**, from **13.443 s to 9.843 s**; moved-warm drops **31.0%**, from **11.648 s to 8.042 s**. Direct remains faster on these warm endpoints.

This does not change the default DF storage/response planner. It qualifies safe final occupied K and root reuse inside the already admitted explicit route. PR #939 is not used; its [assessment](https://github.com/jinzhezenggroup/vibeqc/pull/939#issuecomment-5844952104) found no retained endpoint win in the available experiments.

## Common protocol

- RTX 5090, Slurm `main --gres=gpu:5090:1`, 8 OpenMP/OpenBLAS/MKL threads, one shared sm_120 Release native library, FP64, HF shell AOT enabled. Optional **DFT** stationary-force AOT disabled for this build; it is not the RHF provider.
- Water16mer / water32mer: 48 / 96 atoms, 384 / 768 spherical def2-SVP AOs, 1856 / 3712 cc-pVDZ-JKFIT auxiliaries. The second atom moves +0.001 Bohr along z.
- Complete energy plus analytic forces; energy tolerance `1e-12`, density tolerance `1e-10`, screening `1e-12`, maximum 100 iterations. Independent GPU4PySCF for **each approximation**, with orbital-gradient tolerance `1e-10` and direct screening `1e-14`. Direct and DF are not numerically gated against one another.
- Five forward/reverse control cycles from one frozen post-cold or post-move DF density. Direct uses its own frozen density and five repeats. The controls are same-binary ablations, not a separate historical build. Clean timings exclude tracing; diagnostic replays retain actual source/GEMM/byte work.
- Cold includes `prepare_batch` plus first execution; imports, library/probe loading are outside timing. Cold and geometry updates are single observations from independent owners, not repeated paired speedup estimates. Iterations are retained; the public CUDA HF Fock-count API returns `null`, so DF Fock work is reported only from separate traces.

The common tolerances above are public inputs; effective warm convergence policies differ in this measured binary. Direct retains a density/geometry-qualified energy baseline and uses a FP64 energy comparison guard (about `8.64e-12 Eh` here). DF resets that baseline to infinity and uses the unguarded `1e-12 Eh` comparison. A [subsequent iteration diagnosis](../../../.agents/notes/implemented/performance/2026-09-26-df-qualified-one-step-warm.md#historical-five-step-diagnosis) reproduces five DF iterations despite satisfying the density-step criterion in the first iteration. These timings include that implementation difference; they do not isolate equal-work J/K performance. The later [shared-acceptance qualification](https://github.com/njzjz-bot/vibeqc/blob/b2e57efe9af86bcaf08936c5a2ca287942658a27/benchmarks/results/hf-unified-acceptance-20260926/README.md) unifies the native comparison rule; the [current comparison](../df-one-step-warm-20260926/README.md) additionally qualifies one-step warm reuse; this historical record retains its original binary and observations.

## Clean warm medians (seconds)

| AO | Phase | Dense final, spectral | Occupied final, spectral | Occupied final, root | Direct | DF / direct SCF iterations |
|---:|---|---:|---:|---:|---:|---|
| 384 | warm | 1.203216 | 1.030408 | 1.005487 | 0.765967 | 3 / 1 |
| 384 | moved-warm | 1.246960 | 1.082900 | 1.061671 | 0.766210 | 3 / 1 |
| 768 | warm | 13.442742 | 10.254464 | 9.842618 | 3.056365 | 5 / 1 |
| 768 | moved-warm | 11.648289 | 8.440280 | 8.042058 | 3.058349 | 3 / 1 |

All three DF controls have matching iteration counts in these frozen comparisons. The extra one-repeat observations from independently created control owners remain in the receipt but are not pooled into these medians.

## Cold and geometry updates: single observations

Each cell is seconds followed by SCF iterations in parentheses. Matching iteration counts do not establish identical final-correction work or identical starting densities across independent owners.

| AO | Phase | Dense final, spectral | Occupied final, spectral | Occupied final, root | Direct |
|---:|---|---:|---:|---:|---:|
| 384 | cold | 6.592997 (18) | 5.778377 (18) | 5.736061 (18) | 11.659581 (24) |
| 384 | moved | 4.987226 (9) | 4.346083 (9) | 4.332523 (9) | 7.379937 (16) |
| 768 | cold | 50.639260 (24) | 47.432606 (24) | 47.042524 (24) | 41.714829 (26) |
| 768 | moved | 64.932746 (8) | 39.476472 (8) | 39.104632 (8) | 27.272040 (16) |

## Why it improves, and why direct still wins warm

The separate 768-AO warm trace records one physical final Fock: **4.044 s** with dense final K versus **0.856 s** with occupied final K. Both rebuild the physical Fock and perform strict validation. Five SCF K builds remain occupied; final occupied K makes six total instead of five occupied plus one dense. Single-B never publishes a raw response projection lease.

On the same occupied-final path, force response falls from **4.541 s to 4.124 s**. Applying the retained symmetric metric root replaces two root GEMMs with one, reducing metric-transform work from **1,410,963,865,600 to 705,481,932,800 FLOPs** and removing a **760,217,600-byte** copy. Response scratch remains **1,980,551,200 bytes** within the 5 GB allowance. Both controls still use 58 fitted projection panels / 116 projection BLAS calls and borrow the same **8,769,110,016-byte** B owner.

These component times are intrusive, inclusive diagnostic intervals; do not sum parents with children or substitute them for clean wall times. Remaining DF work includes several large FP64 occupied-K contractions and a roughly 4.1-second force response. In the frozen original geometry it also takes 5 SCF iterations versus direct's 1. The optimized 768-AO warm endpoint therefore remains **3.22x direct**; moved-warm remains **2.63x**. No general claim that DF now beats direct is made.

## Numerical and robustness validation

All **124** retained native complete endpoints, including diagnostic/cold/moved and all repeats, pass. Maximum errors across both methods and sizes are **2.642e-10 Eh** and **1.528e-10 Eh/Bohr**, against unchanged `1e-8` / `1e-7` gates.

- Host lowering/adjoint, delayed-upload lifetime and benchmark gate tests: **131 passed**. Checkpoint compatibility: **45 passed, 2 skipped**.
- GPU molecular matrix: **39 passed**, including all **8 new single-B final-K cases** (orbital-sized and practical auxiliaries, exact/forced-corrected/dense controls, cold/warm/moved). Both native final-snapshot and occupied-response/adversarial suites pass; the latter also covers truncated metrics and rejects stale identity, mismatched density and insufficient capacity.
- Compute Sanitizer memcheck: **0 errors** for the native response/identity suite and both forced-correction molecular cases. A separate 96-AO direct/DF smoke run also passed complete E/F checks.
- **Existing regression, not hidden:** six 256 MiB packed-cache replay assertions in `test_df_packed_values_cuda.py` fail because an unchanged packed geometry regenerates its source. The identical six failures reproduce with the pre-change stack library (`af4a10da...32e`, source `d4cc69a3...ecc`); its other six replay cases pass. Numerical gates reached by those cases pass. This PR neither suppresses nor weakens these assertions. The receipt pins the baseline library and failure-log hashes.
- Compiler, CUDA ownership, SCF/cross-method structure, generated manifest and pre-commit checks pass.

## Provenance and reproduction

Native source base: `013675d52d3b2857fc425c96c9a3bf7a3883ce21` plus [measured-source.patch](measured-source.patch). Measured native library SHA-256: `edad6753260ad527708dac3a43d0c371bcf025ec3256b98ffbba803cea5b4c89`. The patch pins the dirty measured checkout; subsequent evidence/documentation does not change those native bytes. [receipt.json](receipt.json) retains every clean/diagnostic sample, aggregate work counts, all gates and raw-artifact hashes. [independent-reference.json](independent-reference.json) preserves independent energy/force arrays and complete geometry/settings identity. Raw traces remain ignored local artifacts, without external publication.

Slurm jobs: 11796 independent references, 11797 regression matrix, 11798 baseline reproduction / memcheck / smoke, 11799 large endpoints. CUDA 12.9.1; package versions and the loaded library's actual kernel profile are in the JSON records.

The receipt and independent-reference records contain all workload observations
inline. They reconstruct identically to the earlier split companions, which are
recoverable from commit `b2e57efe9af86bcaf08936c5a2ca287942658a27`.

Build the measured source or this PR with a Python environment containing the compiler dependencies, PySCF, GPU4PySCF and CuPy:

```bash
cmake --preset cuda-release-sm120 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DVIBEQC_BUILD_TESTS=ON -DVIBEQC_ENABLE_AOT_SHELLS=ON \
  -DVIBEQC_ENABLE_STATIONARY_FORCE_AOT=OFF
cmake --build --preset cuda-release-sm120 --target vibeqc \
  vibeqc_df_final_snapshot_tests vibeqc_df_occupied_response_tests -j8
export VIBEQC_LIBRARY="$PWD/build/cuda-release-sm120/libvibeqc.so"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:35:00 bash benchmarks/results/df-final-occupied-endpoint-20260926/reproduce.sh
```

Use a fresh output directory for each reproduction. The script preserves Slurm device visibility and rejects reuse of existing result directories.

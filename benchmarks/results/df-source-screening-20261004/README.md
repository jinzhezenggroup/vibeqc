# DF generation-time force screening

The complete 96-atom RHF DF energy-and-force warm endpoint falls from
**4.564 s to 3.315 s (27.4% less time)** when whole-shell force
screening is enabled. This is an incremental same-binary comparison on the
RTX 5090, separate from the earlier occupied-response optimization in
[the preceding qualification](../hf-df-96-20261003/README.md).

| Complete phase | Screening off, s | Screening on, s | Time reduction | SCF iterations |
| --- | ---: | ---: | ---: | ---: |
| Cold, including preparation | 38.662 | 37.550 | 2.9% | 24 / 24 |
| Frozen warm, median of five | 4.564 | 3.315 | 27.4% | 1 / 1 |
| Changed geometry, reconvergence | 31.462 | 30.228 | 3.9% | 13 / 13 |
| Changed geometry, frozen warm median | 4.585 | 3.336 | 27.2% | 1 / 1 |

The two arms use one Release sm_120 library in node3 Slurm job **12192**.
Five frozen warm repeats and five moved-geometry warm repeats are retained
per arm. Cold and reconvergence have one observation per arm. No sample is
selected or normalized by SCF work. Maximum off/on force difference across
all 28 calls is **4.23e-13 Eh/Bohr**. Independent energy/force gates remain
`1e-8 Eh` and `1e-7 Eh/Bohr`.

## What changes

The compiler generates geometry-only radial Gaussian Coulomb-norm bounds.
The force consumer folds the actual Cartesian response weights, checks a bound
on the complete shell contribution, and returns before primitive derivative
recurrences when its allocated error budget permits omission. Through-f,
Cartesian/spherical, RHF/UHF and full/symmetric/packed consumers are covered.

The qualified sm_120 automatic shell domain defaults to a `1e-10 Eh/Bohr`
absolute force-component omission budget. `GENERATIVEQC_DF_SHELL_SCREEN_ABS=off`
retains strict evaluation. Generic, unsupported and capacity-limited consumers
also stay strict. Optional norm storage is admitted after existing response
tiles are fixed; screening never shrinks the tile. The older SSS primitive
screen remains independently off.

This change screens **three-center derivative generation**. Forward three-center
values, metric whitening, retained dense/packed B storage, metric derivatives
and SCF iterations are unchanged. It does not implement fully sparse forward DF.
The analytical envelope has conservative FP64 headroom and invalid-value
fallbacks; it is not an interval-arithmetic certificate.

A separate final-binary counter replay skips **248,071,633 of 342,802,304
primitive products (72.37%)** and **67,578,393 of 101,861,760 active shell
tasks (66.34%)**. All 58 auxiliary panels remain. Norm metadata occupies
1,190,400 bytes and takes 2.840 ms to prepare. Three-center derivative GPU
time falls from 1,955.433 to 701.513 ms; complete force-response GPU time
falls from 3,096.283 to 1,848.935 ms. These intrusive observations are excluded
from the clean endpoint medians.

## README figure

[hf.svg](hf.svg) is a fresh six-size Direct/DF comparison from node3 Slurm
job **12189**, using the default screening policy and separate GPU4PySCF
processes. All **156 native calls** (including 12 diagnostic calls) and
**144 independent reference calls** pass. Maximum native errors are
**4.64e-11 Eh** and **1.49e-10 Eh/Bohr**.

At 96 atoms/768 AOs the plotted DF warm medians are **3.318 s** for GenerativeQC
and **11.232 s** for GPU4PySCF. The figure preserves actual reference iteration
counts and marks varying-work repeats. It does not assert equal SCF work.

The protocol uses nested 3/6/12/24/48/96-atom water clusters, spherical
def2-SVP, RHF, complete analytic forces and eight OMP/OpenBLAS/MKL threads.
DF uses cc-pVDZ-JKFIT (3712 auxiliaries at 96 atoms), explicit `packed-single`
values, occupied fitted response and FP64 BLAS. Native energy/density/direct
screening thresholds remain `1e-12 / 1e-10 / 1e-12`; GPU4PySCF uses
energy/gradient thresholds `1e-12 / 1e-10`, full Fock and screening `1e-14`.

## Resource boundary and validation

At 99 atoms/792 AOs, the current clean baseline and both final-binary
screening arms reject the workload **before SCF with out-of-memory status**
under the same existing admission policy (node3 Slurm job 12192). An initial
off attempt in job 12189 failed identically. These are resource-boundary
observations, not performance samples. The earlier 99-atom success in the
preceding qualification used an older source baseline and is not attributed
to this build. Screening has not changed the forward value-storage limit.

The new tests also cover 3 MiB strict fallback and a 2,000,000-byte clipped-panel
case. Screened and strict arms retain identical auxiliary panel counts. Batch
geometry replacement includes bringing a remote water from 24 to 5 Bohr to
detect stale spatial bounds. Two finite-difference steps check force sign and
geometry-dependent regeneration.

- 118 selected host tests and 77 generated-envelope/SSS-bound tests pass.
  The independent libcint matrix includes all 64 s/p/d/f shell triples,
  signed contractions, self norms and shared-atom derivatives.
- All 30 new molecular GPU tests pass locally and independently on n2
  (RTX PRO 6000 Blackwell, Slurm job 2180). These independent-device results
  do not enter the RTX 5090 timing plot.
- Final-binary memcheck passes locally; n2 memcheck and synccheck include
  orbital f shells, and racecheck covers spherical packed def2-SVP. All report
  zero errors/hazards.
- After preserving PR head `23df34f35`, the integration rebuild passes the
  30 new GPU tests, all 14 complete 96-atom calls and the corrected native
  budget-fixture regression. Its [integration receipt](integration.json.gz)
  is separate from the figure and screening A/B.
- The broader local regression records **165 passed and 17 failed**. All 17
  failed cases reproduce on clean baseline `7d1152038` with the same leading
  assertions: diagnostic counters, BLAS-route attribution, scratch/residency
  assumptions and old insufficient-budget expectations. None is a numerical
  energy/force failure. This broader suite is not claimed as passing.

## Retained evidence

- [summary.json](summary.json), [samples.json.gz](samples.json.gz),
  [references.json.gz](references.json.gz) and [work.json.gz](work.json.gz)
  preserve all six-size observations, independent reference arrays and semantic
  work. `summary.json` binds each compressed part by SHA-256 and size, including
  its decoded JSON identity.
- [comparison.json.gz](comparison.json.gz) and
  [comparison-summary.json](comparison-summary.json) retain every 96-atom A/B
  scalar sample, independent oracle and numerical difference.
- [screening-work.json.gz](screening-work.json.gz) retains the separate
  intrusive counter replay; its times are excluded from clean medians.
- [resource-boundary.json.gz](resource-boundary.json.gz) preserves the larger
  resource observations, including unsuccessful attempts.
- [validation.json](validation.json) records exact binaries, source hashes,
  compiler/cache settings, test cases, sanitizer receipts and all matched
  baseline failures. [measured-source.patch.gz](measured-source.patch.gz)
  reconstructs the measured dirty source from `7d1152038`.

Routine logs, XML, retries and binaries remain in ignored `.artifacts/`;
the repository retains compact evidence rather than those run products.
See [the current contract](../../../docs/developer/df_tuning.md) and
[the decision note](../../../.agents/notes/implemented/performance/2026-10-04-df-generation-time-force-screening.md).

## Reproduce

Use PySCF 2.14.0, GPU4PySCF 1.8.1, NumPy 2.4.6, CuPy and the compiler
dependencies. Matplotlib is only needed for plotting. Build with the preceding
qualification's [ccache-enabled sm_120 instructions](../hf-df-96-20261003/README.md#reproduce).

```bash
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=python:.
export HF_BENCHMARK_PYTHON="$(command -v python)"
export HF_BENCHMARK_OUTPUT="$PWD/.artifacts/df-screening-reproduction"
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:35:00 bash benchmarks/run_hf_acceptance_benchmarks.sh
python -m tools.render_hf_acceptance_benchmarks \
  --raw-directory "$HF_BENCHMARK_OUTPUT" --destination .artifacts/df-screening-figure

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:10:00 bash -lc '
  for screen in off 1e-10; do
    HF_DF_SHELL_SCREEN_ABS="$screen" "$HF_BENCHMARK_PYTHON" \
      benchmarks/results/df-source-screening-20261004/screen_endpoint.py native \
      --nested-water --aos 768 --route df --repeats 5 \
      --reference "$HF_BENCHMARK_OUTPUT/768/reference-df/results.json" \
      --output "$HF_BENCHMARK_OUTPUT/ab-$screen"
  done'
```

The small A/B wrapper is necessary because the acceptance comparator clears
inherited DF controls before each endpoint. For the 99-atom resource check,
apply the retained [larger harness patch](../hf-df-96-20261003/larger-harness.patch)
in an isolated checkout, generate its independent `--aos 792` reference under
Slurm, and use the same wrapper. Preserve assigned `CUDA_VISIBLE_DEVICES` and
retain any failed admission as a resource observation, not a timing sample.

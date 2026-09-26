# RHF direct and DF: VibeQC versus GPU4PySCF

![Complete warm energy and force endpoints](hf.svg)

All four series share one axis. Lines show medians of **all five** complete warm
energy-plus-analytic-force calls; bars show min/max. Crosses expose individual
repeats whose SCF iteration counts vary. The plot compares endpoint cost, without
dividing by iterations or implying equal J/K work.

## Warm medians, seconds

| Spherical AOs | VibeQC direct | GPU4PySCF direct | VibeQC DF | GPU4PySCF DF |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 0.082217 | 0.880455 | 0.013620 | 0.207237 |
| 48 | 0.091444 | 0.940502 | 0.027437 | 0.223901 |
| 96 | 0.130440 | 1.124118 | 0.065085 | 0.260715 |
| 192 | 0.262644 | 1.605668 | 0.181975 | 0.530312 |
| 384 | 0.766056 | 2.381099 | 0.803204 | 1.771504 |
| 768 | 3.048253 | 4.042483 | 6.174827 | 11.254863 |

VibeQC direct and DF take one iteration in every original and moved warm repeat.
GPU4PySCF direct takes one iteration in the plotted original-geometry repeats;
GPU4PySCF DF takes one through 192 AOs, {1, 2, 3} at 384 and {2, 5, 7} at 768.
Those variable repeats remain in the plotted medians and ranges.

At 768 AOs, native DF is 2.03× direct. Its separate trace records one SCF occupied
K and one final occupied K; baseline retention adds zero J/K builds. Force-response
scratch is 1,980,551,200 bytes and borrowed fitted B is 8,769,110,016 bytes.
These clean timings do not provide a component-time split. Public native Fock
counts remain `null` where unavailable; GPU4PySCF `get_veff` counts include the
pre-loop Fock, and DF diagnostic work remains in [work.json](work.json).

The optional same-binary warm-reuse controls stay outside the README plot:
384-AO DF is 0.803204 s with reuse versus 1.005950 s without it; 768-AO DF is
6.174827 s versus 8.021126 s (23.0% less time). Disabled controls take three steps
and use separate owners with their own converged frozen seeds.

## Protocol and acceptance

- RTX 5090, 32,607 MiB, driver 580.95.05, CUDA 12.9.1; eight OpenMP/OpenBLAS/MKL
  threads. Neutral RHF, spherical def2-SVP, nested water32mer prefixes with
  3–96 atoms / 24–768 AOs. DF uses cc-pVDZ-JKFIT. Exact coordinates and basis
  identities are retained in [references.json](references.json).
- Native energy tolerance `1e-12 Eh`, density tolerance `1e-10`, screening
  `1e-12`, maximum 100 iterations. Both native methods share finite-value FP64
  energy comparison with `16*epsilon*max(1,|E|,|previous_E|)`, density-step RMS,
  and physical pre-DIIS `max|FDS-SDF| <= min(1e-8,density_tolerance)` checks.
  Strict final Fock/determinant validation remains mandatory.
- GPU4PySCF 1.8.1 / PySCF 2.14.0 / NumPy 2.4.6 uses its own stopping logic,
  energy tolerance `1e-12 Eh`, orbital-gradient tolerance `1e-10`, full Fock
  builds and direct screening `1e-14`. Equal public tolerances do not imply
  identical internal stopping policies.
- Separate sequential engine processes prevent concurrent resident DF tensors.
  Cold includes preparation and the first synchronized complete execution;
  imports/library probes are excluded. Five warm calls use the frozen post-cold
  density. The second atom then moves +0.001 Bohr along z, reconverges normally,
  and supplies a frozen density for five moved-warm calls. Geometry refresh is
  timed. Diagnostics are excluded from clean medians.
- Native DF explicitly selects packed-single storage, occupied exchange/response,
  fitted occupied source, derivative schedule `qualify`, response algebra `blas`,
  final exchange `auto` and occupied metric `auto`. Response allowances are
  64 MiB through 96 AOs, 256 MiB at 192, 1,000,000,000 bytes at 384 and
  5,000,000,000 bytes at 768. Default storage/response planners are unchanged.
- One-step reuse requires exact same-owner density/Hcore/S/X/occupation/nuclear
  energy matches from a completed strict singleton occupied-RHF endpoint.
  Cold, changed geometry, UHF, batches, no-DIIS and unqualified states retain
  normal iteration and bounded fallbacks. See the
  [current contract](../../../docs/developer/df_occupied_cuda.md).

All **184 native endpoints** pass independent **1e-8 Eh / 1e-7 Eh/Bohr** gates;
maximum errors are **2.6421e-10 Eh / 1.4052e-10 Eh/Bohr**. This includes 168
clean and 16 diagnostic calls, including the disabled controls. Each approximation
has its own reference; direct and DF are not gated against each other.
All 144 reference endpoints are retained and rechecked as well.

The native calls were measured under Slurm **11809** (15-minute limit). The
GPU4PySCF results are the unchanged measurements from **11803**, under the
identical scientific/frozen-warm protocol and hardware setup. They were not
rerun for this figure correction. All repeats are included; the reducer checks
protocol and raw-reference hashes before validating each native endpoint.

## Evidence and validation

[summary.json](summary.json) retains cold/warm/moved/moved-warm medians, iteration
and reference build-count sets, common native identity, and hashes of three files:

- [samples.json](samples.json): all native scalar observations in named columns;
  reconstruct each row with `dict(zip(table["columns"], row, strict=True))`.
- [references.json](references.json): all independent original/moved force arrays,
  protocols, identities, raw hashes and reference timing/work samples.
- [work.json](work.json): all grouped DF diagnostic counters, including controls.

The shared Release sm_120 HF-AOT library SHA-256 is
`7ed5127ad22e919dbdc3e055ec8b2efd0c0725c4ebd620e80d546b8065483c3e`.
Its native/compiler/benchmark execution sources match commit
`b2e57efe9af86bcaf08936c5a2ca287942658a27`; only docs, rendering and host tests
changed afterward. [validation.json](validation.json) preserves the build/test
receipts and the exact historical source-patch identity.

Four native suites, 44 existing GPU molecular cases and four new warm-cache
cases pass. The new cases use independent PySCF gates of 1e-9 Eh / 1e-8 Eh/Bohr.
Compute Sanitizer reports zero errors for the final-snapshot/cache suite.
The final host endpoint/retention group passes 18 tests. Earlier validation
receipts preserve test-fixture and sanitizer-PATH retries without relaxing
production gates. The separate final-K/root
[qualification](../df-final-occupied-endpoint-20260926/README.md) retains its
controls and the previously disclosed six baseline packed-cache replay failures.

The superseded three-step native benchmark and original split records are
recoverable from commit `b2e57efe9af86bcaf08936c5a2ca287942658a27` using
`git show <revision>:<path>`. Consolidation preserves all current scalar values,
reference arrays and work counters exactly; historical diagnoses remain in
Agent Notes. Full raw logs, traces and binaries stay in ignored local artifacts.

## Reproduce

Use Python with compiler dependencies, PySCF, GPU4PySCF, CuPy and Matplotlib.
Build and run from the repository root:

```bash
cmake --preset cuda-release-sm120 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DVIBEQC_BUILD_TESTS=ON -DVIBEQC_ENABLE_AOT_SHELLS=ON \
  -DVIBEQC_ENABLE_STATIONARY_FORCE_AOT=OFF
cmake --build --preset cuda-release-sm120 --target vibeqc -j8
export VIBEQC_LIBRARY="$PWD/build/cuda-release-sm120/libvibeqc.so"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_BENCHMARK_PYTHON="$(command -v python)"
export HF_BENCHMARK_OUTPUT="$PWD/.artifacts/hf-df-reproduction"
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:35:00 bash benchmarks/run_hf_acceptance_benchmarks.sh
PYTHONPATH=python:. python -m tools.render_hf_acceptance_benchmarks \
  --raw-directory "$HF_BENCHMARK_OUTPUT" \
  --destination .artifacts/hf-df-figure
```

To reproduce this retained figure from the original local artifacts, use
`--raw-directory .artifacts/df-warm-one-step-20260926/endpoints`,
`--reference-directory .artifacts/df-unified-acceptance-20260926/endpoints`, and
`--include-warm-controls`. The latter retains the optional 384/768-AO controls
in evidence, never in the figure. Fresh controls use the native benchmark's
`--disable-warm-reuse` option and output directories `<aos>/df-disabled`, under
a finite Slurm GPU allocation. The renderer requires all six sizes and five
repeats by default and preserves Slurm's assigned device visibility.

# HF DF occupied symmetry qualification

On the same RTX 5090, the complete 96-atom RHF DF energy-and-force warm
endpoint drops from **5.574 s to 4.488 s (19.5% less time)** relative to
master `b9c626d15b2c19c21a848cb3625932d57f754a71`. Five frozen warm repeats
and five moved-geometry warm repeats retain one SCF iteration each. No timing
is divided by iterations or selected for matching work.

| Complete phase | Baseline, s | Candidate, s | Time reduction | SCF iterations |
| --- | ---: | ---: | ---: | ---: |
| Cold, including prepare | 46.228 | 38.316 | 17.1% | 24 / 24 |
| Frozen warm, median of five | 5.574 | 4.488 | 19.5% | 1 / 1 |
| Changed geometry, reconvergence | 35.814 | 31.197 | 12.9% | 13 / 13 |
| Changed geometry, frozen warm median | 5.576 | 4.500 | 19.3% | 1 / 1 |

The primary comparison is candidate then baseline in n5 Slurm job **1369**.
An earlier baseline in job **1367** gives 5.557 s warm, followed by the
response-only candidate at 5.085 s. Cold and reconvergence have one observation
per arm and are not statistical latency estimates.

## Compact checkout evidence

The repository checkout retains the material needed for routine review without
keeping a second copy of every raw observation:

- [summary.json](summary.json) records the six-size direct/DF medians, endpoint
  counts, maximum numerical errors and native identity.
- [validation.json](validation.json) records tests, sanitizers, source/build
  identities, 99/108-atom resource qualification and the disclosed baseline
  failures.
- [hf.svg](hf.svg) is the rendered six-size direct/DF comparison used by the
  top-level README.
- [larger-harness.patch](larger-harness.patch) is the small deterministic
  99/108-atom reproduction extension.

The complete pre-compaction raw bundle remains recoverable from Git commit
`f4faedbd02a428ba9ac2eca467e88fada53866fe`, the immediate parent of the
storage-only CI fix. This follows the repository retention policy: historical
payloads stay recoverable from Git history while the tracked checkout remains
within the aggregate evidence budget. The removed large records have these
exact identities:

| Historical path | Bytes | SHA-256 |
| --- | ---: | --- |
| `comparison.json` | 293370 | `9dce78c397fa09c600f8ea1b3064e4e311fc9ea178581de5832bb91b0c8c80f7` |
| `larger.json` | 279491 | `5a14b744a5c3abccee61d50ceac4826b38c8e77ab4c6f168e46b9ee1258696c4` |
| `references.json` | 230576 | `0f7243dd0c8746721e84d2165c61efd9470f96b5ce8c9d074159d1a8b5713f92` |
| `work.json` | 151016 | `7d1ceb7450dda487888660352e5846e6e09c13c50027b0ee4df21c3d9103ec37` |

Smaller raw sample, integration and reconstruction-patch files from the same
qualification are recoverable from that commit as well. Removing them from the
current checkout does not alter the implementation, benchmark claims or test
results.

## README figure

[hf.svg](hf.svg) is a fresh six-size direct/DF comparison from node3 Slurm
job **12090**, using one final Release sm_120 native library and separate
GPU4PySCF processes. It includes 156 native calls (12 diagnostic DF calls)
and 144 independent reference calls. Every call passes `1e-8 Eh` energy and
`1e-7 Eh/Bohr` force gates; maximum native errors across the entire campaign
are **4.73e-11 Eh** and **1.44e-10 Eh/Bohr**.

At 96 atoms/768 AOs the plotted DF warm medians are **4.575 s** for
GenerativeQC and **11.234 s** for GPU4PySCF. These are complete engine-local
frozen warm endpoints with their actual SCF work. Node3 and n5 timings are not
combined into one speedup estimate.

The figure uses nested 3/6/12/24/48/96-atom water clusters, spherical def2-SVP,
RHF, full analytic forces and eight OMP/OpenBLAS/MKL threads. DF uses
cc-pVDZ-JKFIT (3712 auxiliaries at 96 atoms), explicit `packed-single` values,
occupied fitted response, the qualified derivative schedule and FP64 BLAS.
Native energy/density/screening thresholds are
`1e-12`/`1e-10`/`1e-12`; the independent reference uses
energy/gradient thresholds `1e-12`/`1e-10`, full Fock and direct screening
`1e-14`.

## Larger size and resource boundary

The 99-atom/792-AO campaign has 28 baseline/candidate endpoints. Warm time
changes from 6.259 s to 5.690 s and all numerical gates pass. The value planner
reduces the auxiliary tile from 128 at 96 atoms to 12 at 99 atoms, so the
generated K partials are not admitted there; K correctly retains SYRK while the
compact response still provides a 9.1% warm saving. Cold/moved iterations
remain 58/13 and all warm repeats remain one iteration.

Both versions reject the 108-atom/864-AO attempt before SCF with out-of-memory
status under the same measured value/response budget policy. This is a resource
boundary, not a successful timing or a measured whole-process VRAM limit. The
complete raw 99/108-atom record is in the historical bundle above.

## Implementation and provenance

The response roots `r*(r+1)/2` occupied pairs and forms a weighted triangular
metric Gram. Large packed K uses a generated triangular FP64 tiled product with
deterministic reduction slices. Both reuse existing disjoint storage; no device
allocation, precision relaxation, iteration shortcut or oracle work is added.
Exact final-state/resource gates and bounded BLAS/spectral fallbacks remain
explicit. See the
[decision note](../../../.agents/notes/implemented/performance/2026-10-03-df-symmetric-occupied-products.md)
and [current contracts](../../../docs/developer/df_occupied_cuda.md).

The final compatibility build passes 274 host tests with both BLAS interfaces
and 34 molecular GPU tests. The complete identities and sanitizer receipts are
in [validation.json](validation.json). The raw reconstruction patches and full
observation tables are intentionally historical rather than duplicated in the
tracked checkout.

## Reproduce

Use a Python environment with compiler dependencies, PySCF 2.14.0,
GPU4PySCF 1.8.1, CuPy and NumPy 2.4.6. Matplotlib is needed only for rendering.

```bash
ccache --version
export CCACHE_BASEDIR="$PWD"
cmake --preset cuda-release-sm120 \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DGENERATIVEQC_BUILD_TESTS=ON \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON \
  -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF
cmake --build --preset cuda-release-sm120 --target generativeqc -j8
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export CUDA_PATH=/group/software/cuda-12.9.1
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_BENCHMARK_PYTHON="$(command -v python)"
export HF_BENCHMARK_OUTPUT="$PWD/.artifacts/hf-df-reproduction"
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=00:35:00 bash benchmarks/run_hf_acceptance_benchmarks.sh
PYTHONPATH=python:. python -m tools.render_hf_acceptance_benchmarks \
  --raw-directory "$HF_BENCHMARK_OUTPUT" \
  --destination .artifacts/hf-df-figure
```

For an A/B study, build the baseline in a separate checkout with identical
flags and retain a native run from each library against the same independently
computed DF reference. Preserve assigned `CUDA_VISIBLE_DEVICES`; do not mix
measurements from different nodes.

For larger-size qualification only, apply
[larger-harness.patch](larger-harness.patch) in an isolated checkout, then run
the same reference/native commands with `--aos 792` or `864`. The 108-atom
case must remain a rejected resource-boundary observation rather than a
performance sample.

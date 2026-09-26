# #308 host-eigensolve baseline, first #206 increment

No optimization is included. This records the baseline before #309/#310/#311,
using `15d6936390723edf9e9eb0c91fecde4390490573` plus the exact
`measured-source.patch` retained in `raw-evidence.zip`. Every sample identifies that patch and the actual
shared-library SHA-256. The library's embedded scientific source identity was
also checked against the checkout: see `summary.json`.

The first domain is RHF, spherical def2-SVP, 96 AOs, batch 1, 1 GiB DF allowance,
on Slurm's RTX 5090. Each row retains ten measurements in five A/B pairs;
both selections are identical native configurations used as a protocol control.

| Endpoint | Clean median, seconds | CPU reference solves per item |
| --- | ---: | --- |
| Cold energy, including prepared ownership creation/destruction | 0.443143 | overlap 1, core guess 1, final Fock 1 |
| Unchanged-geometry warm energy | 0.232660 | overlap 1, core guess 1, final Fock 1 |
| Changed-geometry energy | 0.408515 | overlap 1, core guess 1, final Fock 1 |
| Warm energy plus complete force | 0.791356 | 3 in the separate cold force trace; warm force counts not profiled here |
| Changed-geometry energy plus force | 0.963408 | not profiled here |

Each changed sample restores the original geometry before its timer. A
separately prepared cold calculation checks the changed endpoint, including
forces; unchanged and cold endpoints are checked too. Warm updates are frozen.
These same-model checks detect replay/cache errors; independent numerical
acceptance remains with the matched #206 matrix and existing scientific tests.

The separate host-profiled run measures about 216 ms of thread CPU time in
the three reference eigensolves during warm energy. Host wall, CPU clocks,
CUDA events and CUPTI kernel intervals overlap; do not add them into elapsed
time. Profiled endpoints are diagnostic and are not speedup samples.

`profile-summary.json` covers a separate whole-process CUPTI capture of one
cold energy and one cold force call, including graph nodes. It records exact
transfer bytes, synchronization API durations and kernel launches. Both calls
take 34 SCF iterations: 68 executed device Fock assembly launches, plus four
physical finalizer J/K applications recorded by the dual ledger. Capture-time
graph declarations do not count as executed SCF work. The earlier graph-level
diagnostic capture is not used to infer kernel counts.

`energy-clean.json`, `energy-profiled.json` and `force-clean.json` preserve all
samples, energies, forces, convergence/iteration diagnostics, inputs, device
identity and setup/destruction costs. `raw-traces.json` retains the original
JSONL text and hashes; `dual-ledger.json` retains the cold host/CUDA breakdown.
`files.json` pins these records and the measured-source patch. Routine test
logs, binaries and the large transient profiler database are not published.

Validation: 24 CPU native tests; 68 hardware-free protocol tests; 9 CPU trace
tests (2 CUDA cases skipped there); 24 GPU Python tests including RHF/UHF
actual-call counts, occupied/dense replay and resource regressions; the native
HF executable. Both GPU call-count cases pass memcheck with zero errors and
zero leaked bytes. All GPU execution used finite Slurm allocations.

The remaining 192/384-AO, batch-4, constrained-memory and RHF/UHF representation
matrix is still required, along with the independent reference gates and
individual optimization ablations. This slice makes no speedup, external
parity or issue-completion claim.

## Evidence retention and reproduction

The standard archive retains the eight original result, trace, profile and
source-patch files without changing their bytes. `raw-evidence.manifest.json`
pins the archive and every member; the original `files.json` remains inside
it. All members were restored and compared byte for byte before removing the
expanded copies. `summary.json` and this README remain directly reviewable.
The retention change reruns no scientific experiment and changes no result.

```bash
python -m tools.unpack_evidence benchmarks/results/issue308-host-baseline \
  --output /tmp/issue308-baseline-evidence
```


Apply the measured patch to a clean checkout of the baseline revision. Build
Release with CUDA 12.9.1, architecture 120, and AOT shells disabled, then use
the existing #206 entry points. Output directories must be fresh and outside
the measured checkout. The Python environment needs the project's benchmark
dependencies. Preserve the scheduler's device visibility.

```bash
cmake -S . -B build/cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVIBEQC_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DVIBEQC_CUDA_ARCHITECTURES=120 -DVIBEQC_ENABLE_AOT_SHELLS=OFF -DVIBEQC_BUILD_TESTS=ON
cmake --build build/cuda --target vibeqc -j 6
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:20:00 \
  env PYTHONPATH=python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python benchmarks/issue206_df_matrix.py --run --host-workloads \
  --case water-tetramer-def2-svp-spherical --batch 1 --library build/cuda/libvibeqc.so \
  --memory-budget-bytes 1073741824 --energy-only --repeats 5 \
  --output-dir /tmp/issue308-energy-clean
```

Run a second invocation with `--host-trace-dir /tmp/issue308-host-traces` and a
new output directory for diagnostic timing. Omit `--energy-only` in a third
clean invocation to retain complete forces. The CUPTI/dual-ledger capture is:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  env PYTHONPATH=python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /group/software/cuda-12.9.1/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none --output=/tmp/issue308-cupti \
  .venv/bin/python benchmarks/issue206_df_force_probe.py \
  --case water-tetramer-def2-svp-spherical --library build/cuda/libvibeqc.so \
  --memory-budget-bytes 1073741824 --repeats 1 \
  --component-trace-dir /tmp/issue308-dual-traces --output /tmp/issue308-dual-ledger.json
```

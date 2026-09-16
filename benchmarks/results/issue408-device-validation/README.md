# Device final-state validation and W (#408)

CUDA DF uses device algebra under the existing shared final-state selector.
Physical J/K, retained C/epsilon and the actual returned D retain their existing
identity checks. FP64 residuals, metric orthogonality, density reconstruction,
idempotency, commutator, electron/spin traces, energy and requested canonicality
are checked before acceptance. W is constructed only when requested and is the
matrix handed to the force consumer. Bounded correction and explicit reference
controls remain effective.

## Complete warm comparisons

`VIBEQC_DF_REFERENCE_FINAL_VALIDATION=1` selects CPU validation/projection/W and
the host physical-Fock adapter; `0` selects the normal device path. Both arms use
the same frozen library, numerical settings and input density. Every table
entry is the median of five clean, interleaved samples with an untimed prime at
each transition. Each arm performs **two SCF iterations at 96 AO and three at 192/384/768 AO**.

| AO | Endpoint | CPU reference (s) | Device (s) | Reduction |
| ---: | --- | ---: | ---: | ---: |
| 96 | Energy only | 0.011507 | 0.009812 | 14.7% |
| 96 | Energy + forces | 0.115924 | 0.111722 | 3.6% |
| 192 | Energy only | 0.060174 | 0.039779 | 33.9% |
| 192 | Energy + forces | 0.187345 | 0.165255 | 11.8% |
| 384 | Energy only | 0.456084 | 0.283945 | 37.7% |
| 384 | Energy + forces | 0.912209 | 0.712288 | 21.9% |
| 768 | Energy only | 2.193397 | 1.014133 | 53.8% |
| 768 | Energy + forces | 3.888060 | 2.602397 | 33.1% |

These are complete `strict=True` warm endpoints, including ordinary validation
and requested forces. They exclude preparation/checkpoint restore and external
oracle comparison. They are not cold-solve or cross-engine speed claims.
Smaller 96/192-AO cases check for regressions. All individual samples and full
energy/force arrays are retained in `measurements/`, with basis metadata interned
once per file without changing numeric values or sample ordering.

RHF water clusters, spherical def2-SVP, identical orbital/auxiliary basis,
`Naux=NAO`, batch one, unlimited DF allowance, metric cutoff `1e-10`, screening
`1e-12`, energy convergence `1e-12`, density convergence `1e-10`, maximum 100
iterations. The independent GPU4PySCF reference geometry/basis/model identities
are checked before applying unchanged **1e-9 Eh / 1e-8 Eh/Bohr** external gates.

Maximum observed clean errors across both arms and all sizes: **1.5e-11 Eh** for energy and **1.62e-10 Eh/Bohr** for a force component.

## Separate diagnostics

`components/` retains host exclusive/inclusive regions, GPU component groups,
original trace hashes, transfers, explicit synchronization counters and final
work observations. These intrusive samples are excluded from the clean table.
Host/GPU intervals overlap; parent and child intervals must not be added.

| AO / endpoint | Validation host, CPU → device (ms) | Validation GPU (ms) | W host, CPU → device (ms) | W GPU (ms) |
| --- | ---: | ---: | ---: | ---: |
| 384 / energy | 159.155 → 4.914 | 4.362 | Absent | Absent |
| 384 / forces | 159.507 → 4.942 | 4.390 | 9.965 → 0.802 | 0.691 |
| 768 / energy | 1178.254 → 23.503 | 20.717 | Absent | Absent |
| 768 / forces | 1179.142 → 23.122 | 20.320 | 91.589 → 6.110 | 5.795 |

Both arms perform one physical final Fock evaluation and zero final eigensolves,
density corrections or candidate rejections in these warm samples. Energy-only
contains neither W construction nor a force stage. Negative and forced-rebuild
tests separately exercise rejection and real bounded correction.

| AO | Validation H2D / D2H (bytes) | Explicit validation syncs | W D2H (bytes) / syncs | Actual workspace (bytes) |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 3,538,944 / 112 | 1 | 1,179,648 / 1 | 10,663,024 |
| 768 | 14,155,776 / 112 | 1 | 4,718,592 / 1 | 42,516,592 |

These are the new validation/W providers' explicit counters, not a CUDA-wide
API census. They exclude library-internal synchronization and optional trace
event waits. No full F/C matrix is downloaded merely for validation. The
reference route downloads C/epsilon/D; its historical dense J/K transfer counters
are incomplete, so no complete reference-route transfer total is inferred.
The 768-AO retained J/K component separately reports 28 D2H bytes and two explicit
synchronizations on the device arm; its reference arm additionally downloads J/K.

## Memory and qualification

| AO / endpoint / policy | Resident device after (MiB) | Sampled warm device peak (MiB) | Sampled warm host peak (MiB) |
| --- | ---: | ---: | ---: |
| 384 / energy / CPU reference | 8022.0 | 8022.0 | 2227.9 |
| 384 / energy / device | 8034.0 | 8034.0 | 2227.9 |
| 384 / forces / CPU reference | 8022.0 | 8150.0 | 2553.2 |
| 384 / forces / device | 8034.0 | 8162.0 | 2553.2 |
| 768 / energy / CPU reference | 20230.0 | 20230.0 | 5345.3 |
| 768 / energy / device | 20272.0 | 20272.0 | 5345.3 |
| 768 / forces / CPU reference | 20230.0 | 20268.0 | 7937.5 |
| 768 / forces / device | 20272.0 | 20310.0 | 7937.5 |

Memory is sampled in separate intrusive warm calls after priming each arm.
Device readings use process-level `nvidia-smi` samples at intervals of at least
50 ms; host RSS uses `/proc/self/statm`. Sampled peaks can miss short transients.
Readings include live prepared owners and opaque runtime caches; they are not
allocator-exact peaks. Lifetime host high-water includes initialization and is
retained separately in `memory/`. The new workspace is serialized across items
and spins and charged to tile admission. Its actual bytes and conservative
reservation are distinct from process memory.

Final qualification: **30 CPU native suites**, **83 CPU Python checks** (one explicit skip), **9 GPU native suites**, **127 focused GPU molecular checks**, and **15 independent-provider/minimal direct-SCF checks plus 29 unchanged Direct/DF derivative checks** pass. Both final validation and retained-frame tests pass memcheck (including leak checks) and initcheck with zero errors.

Qualification covers RHF/UHF (including empty beta), Cartesian/spherical,
multiple/inactive/failed items, changed geometry, bounded budgets, force/energy
replay, independent diagnostics and the actual W consumed by forces. Shared
analytic/adversarial tests include ill-conditioned metrics, values around the
acceptance thresholds, stale identities/device generations, nonphysical Fock,
invalid occupations, nonfinite/overflowing products and exhausted corrections.
The direct-SCF energy/force regression matrix and numerical gates are unchanged.
`qualification.json` binds commands, totals and log hashes to the tested builds.

The rejected fixed reduction allowance and uninitialized diagnostic padding are
recorded in `rejected.json` and the
[Agent Note](../../../.agents/notes/implemented/performance/2026-09-16-device-final-validation.md).
Only the final, sanitizer-clean build supplies the accepted comparisons here.

## Reproduce

Measured source base: `b9df84f3fdee2fc8ddf4236a6f238bac3bb71409`, dirty.
`reproduction/source.patch` reconstructs the native/Python runtime;
`reproduction/support.patch` reconstructs the benchmark runner and tests;
`reproduction/metadata.patch` restores the CUDA ownership inventory used by checks.
`qualification.json` records native source identity, library/executable hashes,
software and hardware. This qualification uses **Release, CUDA 12.9.1, sm_120,
AOT shells OFF, fast compile OFF**, with one host numerical thread.

These patches preserve the exact measured sources, including their original
CuMetal build limitations. Use the current repository for the reviewed portable
implementation. Later source fixes are described in the Agent Note; regenerating
historical patches would break the recorded source identities.

From the repository root, build with the recorded configuration:

```bash
cmake -S . -B build/cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/group/software/cuda-12.9.1/bin/nvcc \
  -DCUDAToolkit_ROOT=/group/software/cuda-12.9.1 \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DVIBEQC_ENABLE_AOT_SHELLS=OFF
cmake --build build/cuda -j 8
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 bash benchmarks/results/issue408-device-validation/reproduction/benchmarks.sh
```

Set `PYTHON` to an environment with the repository dependencies and PySCF, and
make the CUDA runtime libraries available through the usual library search path.
The script honors `VIBEQC_LIBRARY`, `OUTPUT` and `CHECKPOINT_DIR`; preserve Slurm's
device visibility. If freezing a library, retain its `libvibeqc.so.0` symlink.

Exact checkpoint/density hashes are retained in every measurement. The measured
384/768 seeds came from the earlier issues388–391 ablation workflow; the 96/192
seeds were generated by this run. Those transient checkpoints are not published.
Without them, the portable script creates a fresh post-cold density and freezes
it across both arms, reproducing the protocol rather than the archived density
hash or exact timings. External oracle inputs remain versioned under
`benchmarks/results/issue377-379-df/gpu4pyscf/`.

`reproduction/measured-*` preserves the exact historical drivers and memory
sampler, including local paths. `reproduction/warm_memory.py` reproduces separate memory sampling
with `--aos`, `--checkpoint` and `--output` inside a finite Slurm allocation.
The portable script declares two expected iterations at 96 AO and three at the
other sizes. A fresh seed that changes this work must be qualified as a separate
experiment with its own explicit expected count. The historical 96/192 records
retain `expected_iterations: null` because those original runs omitted that CLI
flag; all their recorded samples, primes and diagnostics nevertheless agree on
two/three iterations. We preserve that original metadata instead of claiming
the flag was set retroactively.
`reproduction/collect.py` publishes only complete runs and verifies sample count,
interleaving, unchanged numerical gates, identical density hashes and work counts.
New publications require a positive declared expected count and matching
per-policy iteration arrays across energy and force endpoints.
Routine logs, binaries, checkpoints and raw diagnostic traces stay in `.artifacts/`.

# External basis contracts: issue #169

Scientific source: `4a6ae98afa5ac209578b10263bc4c1e504925979`. The final Slurm 9015 run uses the release
sm_120 native library identified in [build.json](build.json). The six raw case
records use the shared `vibeqc.validation` envelope and record exact source,
library, runner, fixture and complete orbital/auxiliary basis identities.
The archive commit adds evidence only. A subsequent host metadata fix converts
accepted NumPy integer scalars before serializing occupation-error metadata;
`metadata-fix.json` verifies that the scientific native objects are unchanged.
The final focused CPU/GPU and sanitizer logs include that additional case.

All six water/Fe24+/FeH25+ cases pass in Cartesian and real-spherical bases:
300 independent energy-plus-force endpoints, 240 interleaved phase timings,
80 ragged batch item checks and four isolated malformed-coordinate checks.
Maximum absolute errors across these checks are 2.501e-12 Eh and
6.679e-11 Eh/Bohr, against the independent PySCF 2.14.0 references.
The gates remain 1e-8 Eh and 1e-7 Eh/Bohr. BSE data/version/license, NumPy 2.4.6,
original/changed geometries and generator hashes are in the fixture manifest.
Two independent generations produced byte-identical files.

CPU bundled/imported STO-3G outputs and expanded mathematical inputs are exact.
CUDA input identities are exact; repeated force reductions differ in the last
few bits and are checked with a separate 1e-12 replay gate. This does not change
the independent HF accuracy gates. Fe uses an original small diagnostic basis
with two active all-electron electrons. These tests do not establish neutral
transition-metal chemistry or basis-set/model accuracy. Real Fe cc-pVTZ g shells,
synthetic auxiliary h shells and Au def2-TZVP's 60-core-electron ECP are retained
as data and explicitly rejected for execution.

## Timing boundary

Each of five trials alternates CPU/CUDA order and measures a fresh one-shot
calculation, new prepared plan, cold execution, fixed replay, changed geometry
and changed replay. Every call returns energy and forces to host before the
clock stops. Cold below includes native plan preparation plus first execution;
import cost and one-shot costs are separate fields in the raw records.
All changed replays explicitly resubmit the changed coordinates. Ragged batches
preserve item order, active-electron counts and detached result provenance.

The following medians are milliseconds, shown as **CPU / CUDA**. They describe
this existing executor on small diagnostic inputs; no schedule or speed
promotion is claimed.

| Case | Cold | Fixed replay | Changed geometry |
| --- | ---: | ---: | ---: |
| fe_h_ion_cartesian | 10299.899 / 226.225 | 10264.494 / 197.412 | 10312.646 / 218.843 |
| fe_h_ion_spherical | 10285.830 / 428.742 | 10285.943 / 383.201 | 10256.105 / 423.019 |
| fe_ion_cartesian | 6846.829 / 19.715 | 6852.878 / 6.857 | 6854.594 / 15.473 |
| fe_ion_spherical | 6881.109 / 26.418 | 6888.659 / 1.435 | 6879.675 / 22.957 |
| water_cartesian | 3400.803 / 152.042 | 3407.858 / 132.478 | 3403.695 / 170.502 |
| water_spherical | 3480.076 / 151.938 | 3471.226 / 132.358 | 3470.594 / 170.902 |

The public HF API does not expose allocator peak measurements, so the evidence
marks peak memory unavailable. Build task timings/toolchains are retained in
`build.json`; this change introduces no new GPU kernel or schedule.

## Verification and reproduction

- `python-tests.log`: 895 passed, 181 optional skips.
- `native-tests.log`: all 11 native CPU suites pass, including integer-overflow,
  exponent, coordinate and coefficient-normalization guards.
- `affected-tests.log`: 45 final basis/AO/source-identity checks pass.
- `gpu-tests.log`: eight imported-basis GPU tests plus native source identity pass.
- `memcheck.log`: all eight GPU tests pass, zero errors and zero leaked bytes.
- `metadata-cpu-tests.log`: 47 final focused CPU checks pass.
- `metadata-gpu-tests.log`: ten final GPU/source-identity checks pass.
- `metadata-memcheck.log`: nine final GPU tests pass, zero errors and leaked bytes.
- `precommit.log`: every repository hook passes.

Build with the documented release preset, then run on the allocated GPU:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:30:00 env PYTHONPATH=python:. \
  VIBEQC_LIBRARY="$PWD/build/cuda-release-sm120/libvibeqc.so" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/validate_external_basis.py --cuda --output build/basis-evidence
```

Run the opt-in tests with `VIBEQC_BASIS_CUDA_TEST=1`, selecting
`tests/python/test_external_basis_cuda.py`, through the same finite Slurm
allocation. Preserve Slurm's `CUDA_VISIBLE_DEVICES`. See
[the basis contract](../../../docs/external_basis.md) for normalization,
provenance, capability and future higher-l/ECP integration requirements.

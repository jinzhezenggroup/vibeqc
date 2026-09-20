# Stuttgart RLC Na/K complete DFT forces

This qualification adds complete public LDA/PBE RKS/UKS force evidence for the
exact PySCF 2.14.0 `stuttgart-dz` Na/K orbital/ECP records with STO-3G H. It
covers nine-AO s/p NaH/KH, neutral singlets and +1 doublets, Cartesian and real
spherical representations, CPU and NVIDIA CUDA. Production scientific code and
public angular/resource limits are unchanged. The older raw/HF qualification
is retained separately in `../ecp-stuttgart-171/`.

## Numerical and lifecycle gates

All 48 new tests passed without skips: 32 complete analytic/finite-difference
cases and 16 exact-budget mixed-batch replay/recovery cases. The same discrete
24 x 8 x 16 XC grid is used by the native method and independent PySCF
full-grid-response oracle. Energy agreement must be within 2e-8 Eh, analytic
forces within 1e-7 Eh/bohr, and both reconverged directional differences
(steps 3e-4 and 1e-4 bohr) within 2e-7 Eh/bohr. Translation and representation
checks retain their 1e-9 Eh/bohr gates.

| Maximum observed error | CPU | CUDA |
|---|---:|---:|
| Analytic force (Eh/bohr) | 5.65791e-11 | 5.65719e-11 |
| Directional finite difference (Eh/bohr) | 2.49365e-11 | 2.41534e-11 |

Prepared PBE batches pair NaH/KH with H2 (RKS) or H (UKS), use exact planned
host/device budgets, and verify warm reuse, changed-geometry agreement with a
fresh calculation, malformed-item isolation and subsequent recovery. CUDA runs
with CPU scientific fallback entrypoints disabled. The spherical doublet
Na/K replay cases also run under Compute Sanitizer memcheck. Final test receipts,
including 24 existing CUDA cases, 5 native ECP CTests, 90 adjacent Python cases
and 2 sanitizer cases (zero memory errors), are in `summary.json`. All passed
without skips. The first CTest invocation lacked the CPU test executable; it
was built and the complete five-test set rerun successfully. The native library
digest stayed unchanged, and the initial failure and recovery logs are retained.

Each hydride force executes 131766 primitive records, 6561 ordered quartets
and 53775360 two-grid ECP pair samples. CPU additional numeric staging is
41829216 bytes. CUDA additional host/device bounds are at most 8107736 /
211279320 bytes. These remain inside the existing 256 MiB host / 512 MiB device
force staging caps; they do not bound process RSS, compiler memory or the
pre-existing native SCF owner. The complete planned peaks and native CUDA ledger
are retained per batch in `endpoints.json`.

## Source and runtime identity

Measured commit: `df1584039eb1d904ae651d66b9e73e43e61f7efb`, based on
`439776a`. Later documentation/evidence changes do not change measured scientific
or test inputs. The verified source-input manifest and native-library digest are
bound in `source-identity.json`. `raw-evidence-manifest.json` hashes the original
build, test, JUnit, driver and toolchain records; the complete raw archive is
retained outside Git and identified by SHA256 in `summary.json`.

Release CUDA 12.9 build, RTX 4090 / sm_89, AOT shells off, pinned PySCF 2.14.0,
one OpenMP/OpenBLAS/MKL thread per test process. Both CPU and CUDA cases load the
same explicitly selected native library. Compute Sanitizer is from CUDA 12.8.
Exact compiler, driver and package details are in the retained metadata.

Endpoint timings are single sequential measurements, not speed comparisons.
First-use source generation/compilation is included where it occurs (the first
CPU/CUDA LDA calls took about 42 / 162 seconds). Later calls reuse verified
compiler caches. Cold prepared batches are not cold compiler-cache measurements.

## Reproduction

Install the pinned `reference-test` extra and select an actual allocated NVIDIA
GPU. For the measured architecture:

```sh
cmake -S . -B build-stuttgart-dft -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DVIBEQC_ENABLE_AOT_SHELLS=OFF
cmake --build build-stuttgart-dft --parallel 6
export PYTHONPATH=python VIBEQC_LIBRARY=$PWD/build-stuttgart-dft/libvibeqc.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VIBEQC_PROFILE=off
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc VIBEQC_ECP_CUDA_TEST=1 VIBEQC_ECP_CUDA_TARGET=sm_89
export VIBEQC_STATIONARY_CACHE=$PWD/.cache/stuttgart-dft
python -m pytest tests/python/test_ecp_stuttgart_dft.py -q --junitxml=stuttgart-dft.xml
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest 'tests/python/test_ecp_stuttgart_dft.py::test_stuttgart_budgeted_replay_and_recovery[pbe-uks-spherical-Na-cuda]' 'tests/python/test_ecp_stuttgart_dft.py::test_stuttgart_budgeted_replay_and_recovery[pbe-uks-spherical-K-cuda]' -q
python -m pytest tests/python/test_ecp_public_cuda.py tests/python/test_ecp_stationary_cuda.py -q
ctest --test-dir build-stuttgart-dft -R vibeqc_ecp_ --output-on-failure
```

Use the actual allocated GPU's architecture if different. CPU CI executes the
24 CPU cases and explicitly skips the 24 opt-in CUDA cases; it does not substitute
for the no-skip physical NVIDIA run. `tests/python/test_ecp_stuttgart.py` pins
parameter hashes and source provenance; no external parameter tables are copied
into production or this report.

This does not establish other Stuttgart elements/families, arbitrary molecular
states, continuum-grid accuracy, spin-orbit terms, r2SCAN ECP forces or CUDA d
orbitals. No general performance claim or scientific-code retirement is made.

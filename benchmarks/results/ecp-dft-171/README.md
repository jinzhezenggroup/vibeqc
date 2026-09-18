# Semilocal ECP energy qualification

This slice enables the existing LDA/PBE RKS/UKS energy consumers for supported
ECP Gaussian AO values and first spatial jets. It qualifies installed PySCF
2.14.0 LANL2DZ Na with STO-3G H, neutral NaH (RKS, two explicit electrons) and
its +1 doublet cation (UKS, one explicit electron), nine AOs, Cartesian and
spherical representations, CPU and CUDA. Complete DFT forces remain rejected.

## Numerical gates

Independent PySCF/Libcint/Libxc solves use the same explicit default grid as
VibeQC: 48 radial, 16 polar, 32 azimuth points per atom, unit radial scales,
equal-radius Becke partition, no pruning. XC is LDA_X+LDA_C_PW or PBE. Two
independent initial guesses must reach the same reference energy. Native
energy tolerance is 1e-12, density tolerance 1e-10, maximum iterations 150.
The reference uses energy tolerance 1e-13 and gradient tolerance 1e-10.

| Maximum across 16 recorded endpoints | Result | Gate |
|---|---:|---:|
| Total energy error | 1.242e-9 Eh | 1e-8 Eh |
| Individual nuclear/one-electron/Hartree/XC component error | 7.295e-10 Eh | 2e-8 Eh |
| Physical residual RMS | 3.289e-11 | 1e-9 |

The independent ECP residual expectation is at least 0.0215 Eh in magnitude;
these are not accidental all-electron or vanishing-ECP cases. The tests also
check core-adjusted electron populations and nuclear repulsion, energy
component accounting and explicit rejection of force requests. These results
validate matched discretized energies, not continuum XC quadrature convergence.

## Execution and resource gates

- New ECP/DFT tests: CPU 12 passed; CUDA 12 passed. Each includes eight energy
  endpoints and four exact-budget mixed ECP/all-electron batch tests.
- Mixed batches check cold/warm replay, changed-geometry independent energies,
  failed-item isolation, restoration and the CUDA allocation ledger.
- Existing CPU DFT/resource regression: 39 passed.
- Existing CUDA resource regression: 7 passed. Six DFT tests initially skipped
  because their separate opt-in variable was absent; all six passed in the
  retained enabled rerun. Both original and supplemental logs are retained.
- ECP/KS preflight and model checks: 47 passed, 4 expected GPU skips on CPU.
- Native CTest: CPU 31/31; CUDA ECP 3/3.
- Compute Sanitizer: spherical PBE RKS/UKS energy endpoints, 2 passed, 0 errors.
- Local publication checks: 170 ECP/IR/ownership/structure tests passed;
  compiler/SCF dependencies, CUDA ownership, Ruff and formatting passed.

No native scientific, compiler scientific or native runtime lines changed.
There is no performance claim or schedule promotion.

## Source identity and build reuse

Measured candidate: baseline `b78215ef0bf2cb43c91234c9c2dc852151605468` plus
the four overlays in `source-identity.json`, as committed in `180fbc0`. Those
original qualification bytes were checked against the recorded SHA256 hashes.
The review follow-up changes only the unsupported-label preflight fixture from
g to h, because #446 permits local label g with f projectors. Numerical endpoint
fixtures and runtime code are unchanged; the retained hashes identify the
original measured overlay. Later documentation and result files do not affect
the measured code.

The run reused the existing Release CPU/CUDA libraries originally built on
`072f6def38802696aa86138c4c5cc8dc4305ac97`. All 604 relevant native/header/compiler/
generator/build input files compared byte-for-byte with the candidate; there
were no line-ending differences. Both binaries also matched their prior
qualification hashes. New tests, native CTest and new endpoint measurements
were executed against these exact binaries; this was not a new native build.
The full per-file receipt is hashed in the compact source identity and retained
in the raw archive. CPU uses the declared independent native reference; CUDA
uses the existing native ECP/KS paths without a PySCF runtime fallback.

Environment: RTX 4090, CUDA 12.9, sm_89, AOT shells off, GCC 11.4, one
OpenMP/OpenBLAS/MKL thread per numerical process. Sanitizer is CUDA 12.8.
Exact Python, NumPy, PySCF, source and binary hashes are in the reports.

## Reproduction

Install the pinned `reference-test` extra and build this source in Release:

```sh
cmake -S . -B build-ecp-dft-cpu -DVIBEQC_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-ecp-dft-cpu --parallel 4
cmake -S . -B build-ecp-dft-cuda -DVIBEQC_ENABLE_CUDA=ON -DVIBEQC_ENABLE_AOT_SHELLS=OFF -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-ecp-dft-cuda --parallel 4
export PYTHONPATH=python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VIBEQC_PROFILE=off
export VIBEQC_LIBRARY=$PWD/build-ecp-dft-cpu/libvibeqc.so
python -m pytest tests/python/test_ecp_dft.py -q -k cpu
python tools/qualify_ecp_dft.py --device cpu --output cpu-endpoints.json
export VIBEQC_LIBRARY=$PWD/build-ecp-dft-cuda/libvibeqc.so VIBEQC_ECP_CUDA_TEST=1
python -m pytest tests/python/test_ecp_dft.py -q -k cuda
python tools/qualify_ecp_dft.py --device cuda --output cuda-endpoints.json
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/python/test_ecp_dft.py -q -k 'cuda and spherical and pbe'
VIBEQC_DFT_CUDA_TEST=1 VIBEQC_RESOURCE_CUDA_TEST=1 python -m pytest tests/python/test_dft_cuda.py tests/python/test_ks_resources.py -q -k cuda
```

Select the architecture matching the allocated GPU. Source installation and
native libraries must come from the same revision; do not silently substitute
an unrelated installed library.

## Retention and limits

`cpu-endpoints.json`, `cuda-endpoints.json`, `summary.json`, `toolchain.json`
and `source-identity.json` are the compact review evidence. The 27 raw records
and their individual hashes are inventoried in `raw-evidence-manifest.json`.
The operator-retained `ISSUE-171-dft-results.tar.gz` has SHA256
`74d637704e31a043ad63aa4633fe817cf0d52a15a7fc1aa67a99efbe72787c64`.

This does not close #171 C2 or qualify DFT forces, other ECP parameter families,
nonlinear core corrections, arbitrary functional compositions, hybrids, DF/ECP,
higher AO jets or performance leadership. PySCF parameter tables are loaded
from the test installation, not redistributed.

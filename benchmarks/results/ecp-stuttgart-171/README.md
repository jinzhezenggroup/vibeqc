# Stuttgart RLC Na/K qualification

Bounded test-only qualification for the exact PySCF 2.14.0 Stuttgart RLC
(`stuttgart-dz`) Na/K orbital/ECP records with STO-3G H. The fixtures are neutral
singlet NaH/KH and their +1 doublet cations, nine AOs, two/one valence electrons,
direct RHF/UHF on CPU/CUDA. No production/compiler/native arithmetic changes,
new angular domain, DFT/ECP support, or performance promotion are claimed.

## Results

- CPU: 31/31 native tests; 10 new numerical tests passed (10 expected GPU skips);
  11 existing ECP preflight tests passed.
- CUDA: 3/3 native ECP tests; 10 new numerical tests passed (10 deselected CPU tests).
- Compute Sanitizer: both raw-component tests passed, 0 errors.
- Four complete CPU and four complete CUDA endpoint records bind external-oracle
  comparisons and budgeted execution to exact library/fixture/driver hashes.

| Maximum across eight endpoints | Absolute error |
|---|---:|
| Local matrix | 0.000e+00 Eh |
| Nonlocal matrix | 6.646e-13 Eh |
| Complete energy | 2.331e-15 Eh |
| Complete force | 1.213e-09 Eh/bohr |
| Net force | 2.012e-16 Eh/bohr |

The tests check two geometries, both quadrature levels, two-step all-center
Libcint finite differences, nonsymmetric weighted contractions, full RHF/UHF
forces, and exact-budget geometry replay against complete-energy differences.
The local residual and its derivatives are exactly zero while the nonlocal
projectors and the effective-charge Coulomb tail remain active.

## Source and environment

Base commit: `072f6def38802696aa86138c4c5cc8dc4305ac97`. The measured tree is its Git archive plus
`tests/python/test_ecp_stuttgart.py` and `tools/qualify_ecp_stuttgart.py`, whose
SHA256 identities are in `source-identity.json`. All 3883
baseline regular files were verified unchanged. The two overlay files were
verified against both local working files and staged Git blobs. Later docs and
result files do not alter measured source.

Release CPU/CUDA builds; RTX 4090, CUDA 12.9, architecture 89, AOT shells off,
one OpenMP/OpenBLAS/MKL thread per numerical process. Compute Sanitizer uses
CUDA 12.8. Exact compiler/GPU/driver/package details and native-library hashes
are retained in the JSON reports. Single fresh-call timings have no warmup or
statistical sampling and must not be treated as speedup measurements.

## Reproduction

Install the pinned `reference-test` extra. Use this revision's two qualification
files over the baseline above to reproduce the recorded source identity, then:

```sh
cmake -S . -B build-stuttgart-cpu -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=OFF
cmake --build build-stuttgart-cpu --parallel 4
cmake -S . -B build-stuttgart-cuda -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release -DVIBEQC_ENABLE_CUDA=ON -DVIBEQC_ENABLE_AOT_SHELLS=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-stuttgart-cuda --target vibeqc vibeqc_ecp_projector_tests vibeqc_ecp_capability_tests vibeqc_ecp_cuda_error_tests --parallel 4
export PYTHONPATH=python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VIBEQC_LIBRARY=$PWD/build-stuttgart-cpu/libvibeqc.so
ctest --test-dir build-stuttgart-cpu --output-on-failure --parallel 2
python -m pytest tests/python/test_ecp_stuttgart.py tests/python/test_ecp_validation.py -q
python tools/qualify_ecp_stuttgart.py --device cpu --output stuttgart-cpu.json
export VIBEQC_LIBRARY=$PWD/build-stuttgart-cuda/libvibeqc.so VIBEQC_ECP_CUDA_TEST=1
ctest --test-dir build-stuttgart-cuda -R vibeqc_ecp_ --output-on-failure
python -m pytest tests/python/test_ecp_stuttgart.py -q -k cuda
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/python/test_ecp_stuttgart.py -q -k 'cuda and components'
python tools/qualify_ecp_stuttgart.py --device cuda --output stuttgart-cuda.json
```

Select the architecture for the actual allocated GPU if it differs from sm_89.

## Retention and limits

`summary.json` holds acceptance receipts; `cpu-endpoints.json` and
`cuda-endpoints.json` hold compact numerical/resource records;
`source-identity.json` binds baseline/overlay bytes; `toolchain.json` identifies
the tools. `raw-evidence-manifest.json` hashes every raw build/test/configuration
record in the operator's retained `ISSUE-171-stuttgart-results.tar.gz` archive.
The verified archive SHA256 is `b98e482a4f99c55a45f69f8a44e508c6d6b6cb21316708cc59873cff420058f3`.
Routine logs, build products and the bulk baseline manifest are not committed.
The tests and compact reports provide the reviewable reproduction contract.

PySCF parameter tables are consumed from the test installation, not redistributed
here. Consult that installation's basis-file notices and original references.
This evidence does not establish other Stuttgart elements, RSC/MDF families,
arbitrary geometries, high-angular shells, spin-orbit physics, or other methods.

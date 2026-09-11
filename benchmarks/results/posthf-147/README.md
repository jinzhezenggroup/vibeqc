# CG10: owned RHF references and bounded MO providers

These records validate issue #147 at clean scientific revision
`bf85a4176b3c232c0f6cfd0715b324f07761e3e6`. They include the corrected source
metadata capacity model and the actual f-shell partial-tile/full DF MP2 tests.
`manifest.json` preserves the SHA-256 hashes of the raw evidence files.

## Validation

| Gate | Result |
| --- | --- |
| Complete Python regression suite (clean CI) | 762 passed, 113 skipped |
| Focused CPU tests | 41 passed |
| Native CPU suites | 9 passed |
| Real CUDA tests | 14 passed |
| Compute Sanitizer, conventional CUDA / generated DF | 10 / 4 tests passed; zero errors and zero leaked bytes |
| H2, H2O, LiH endpoint paths | 18 passed |
| Conventional block/amplitude/energy maximum absolute error | `7.771561172376096e-15` |
| Same-Hamiltonian DF block/amplitude maximum absolute error | `2.806088694740083e-14` |
| Native HF-to-MP2 maximum energy error | `9.76007469288831e-14` Eh |
| Element/amplitude gates | `atol=1e-11`, `rtol=1e-10` |
| MP2 correlation-energy gate | `1e-9` Eh |
| Raw conventional timing samples | 180: 30 cold and 150 reuse |

The six shared-schema conventional records in `evidence.json` compare CPU
and cuBLAS providers with identical pinned orbitals and Hamiltonians. The six
`*-df-*.json` files compare CPU and generated CUDA raw sources using the same
DF Hamiltonian. Conventional-versus-DF fitting differences appear separately.
The six `*-native-*.json` files exercise native CPU/CUDA HF, owned canonical
snapshot export, and the MP2 bridge. Independent references pin PySCF 2.14.0
and NumPy 2.4.6; a repeated generation produced identical array hashes.
Committed fixtures and their original exporter retain reference licensing and
BLAS provenance. PySCF is not imported by the production interface or CI tests.

GPU executions used finite Slurm allocations on `main` with
`--gres=gpu:5090:1`; final endpoint job **9001** ran on an NVIDIA GeForce RTX
5090, `sm_120`, 170 SMs. The records preserve native library/source hashes,
device identity, NVCC/PTXAS 12.9.86, CUDA runtime `12090`, driver `13000`, and
the actually loaded cuBLAS version `120901` (12.9.1). Other GPU architectures
have not been numerically validated by this run. The transform compilation
took **3.248553679 seconds**. Its small validation kernel used **26 registers**,
with no stack or spills; these resource counts do not describe cuBLAS internals.

## Costs and scope

Median complete provider/CPU-MP2 bridge times, in milliseconds:

| Case | CPU cold | CUDA cold | CPU reuse | CUDA reuse |
| --- | ---: | ---: | ---: | ---: |
| H2 | 14.276 | 27.986 | 0.0628 | 0.1052 |
| H2O | 2215.898 | 2525.785 | 0.0778 | 0.1213 |
| LiH | 1188.829 | 1348.835 | 0.0754 | 0.1191 |

CUDA is slower on all three tiny fixtures. **No performance replacement is
promoted.** These results establish interface and numerical validity. Raw
samples alternate CPU/CUDA trials; synchronized setup, source, transformation,
transfer, download and reuse costs are reported separately. HF export is a
separate endpoint and is not included in the provider timing table.

The conventional source evaluates values-only AO shell tiles on the CPU.
The CUDA provider stages those tiles and performs the four transformations
with cuBLAS, retaining requested MO outputs on the device. Generated DF runs
replace only the raw A/M source: whitening, MO transforms and the MP2 bridge
remain on the CPU, with explicit D2H staging. Ordinary streams are used.

| Largest observed planned capacity | Bytes |
| --- | ---: |
| Conventional CPU provider | 8,400,160 |
| Conventional CUDA provider, including three cached blocks | 322,981,152 |
| Individual native CUDA arena | 4,198,400 |
| Mixed DF source/provider | 8,414,103 |

Actual native arenas match their plans. Each cuBLAS handle charges a checked
96 MiB provider allowance in addition to explicit workspace and buffers;
measured provider deltas in this run were 8–12 MiB. The allowance remains
conservative across supported provider versions and is checked at preparation.
The source includes an 8 MiB through-f recurrence allowance. These bounds
cover numeric buffers, with the precise ownership and cache rules in
[`docs/posthf.md`](../../../docs/posthf.md). Object headers, allocator rounding,
BLAS host workspaces, CUDA context/modules/stacks, caller-retained detached
exports, and preceding HF setup are outside this scope. They are not total
process or total VRAM bounds. Simultaneous providers require summed budgets.

## Reproduction

Build the native CPU/CUDA libraries as described by the repository, using the
recorded toolchain. The endpoint run used Python 3.11.14 from
`/home/jzzeng/codes/qc/build/gpu4pyscf-venv/bin/python`. From a source checkout:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  env PYTHONPATH=.:python VIBEQC_LIBRARY=$PWD/build/cuda/libvibeqc.so \
  OMP_NUM_THREADS=1 python tools/validate_posthf.py --cuda --generated-df \
  --nvcc /group/software/cuda-12.9.1/bin/nvcc \
  --cache /tmp/posthf147-cuda-cache --output /tmp/posthf147-final-evidence

PYTHONPATH=.:python VIBEQC_LIBRARY=$PWD/build/cpu/libvibeqc.so \
  python -m pytest tests/python/test_posthf_reference.py \
  tests/python/test_posthf_providers.py -q

srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  env PYTHONPATH=.:python VIBEQC_LIBRARY=$PWD/build/cuda/libvibeqc.so \
  VIBEQC_POSTHF_CUDA_TEST=1 VIBEQC_NVCC=/group/software/cuda-12.9.1/bin/nvcc \
  OMP_NUM_THREADS=1 python -m pytest tests/python/test_posthf_cuda.py -q
```

For Compute Sanitizer, insert `compute-sanitizer --tool memcheck --leak-check
full --error-exitcode 99` before the GPU Python command. The archived sanitizer
runs split the conventional and generated DF tests; together they cover all
14 tests, including the actual f partial tile and full DF MP2 amplitudes.
`ci-python-summary.log` links the clean scientific-revision CI run. Native
adapter behavior is exercised through Python ctypes; those executions are
outside the separate C++ coverage instrumentation reported by Codecov.

Routine log/XML files named in this historical account are now represented in
[the retention audit](../retention-238/migration.json), with extracted measurements,
diagnostic conclusions, and exact original Git/checksum identities.

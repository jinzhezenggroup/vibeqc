# Scalar ECP baseline evidence

Measured 2026-09-12 on an allocated Inspire RTX 4090, CUDA 12.9, driver
570.195.03, Python 3.11.16, PySCF 2.14.0; CPU/OpenBLAS workers set to one.
`measurements.json` retains hardware, library hashes, source hashes, inputs,
three timing samples, numerical errors and conservative resource bounds.
This is small-fixture validation of the [documented domain](../../../docs/ecp.md),
not a clean-revision performance promotion or broad heavy-element claim.

| NaH/NaH+ check | CPU | CUDA |
| --- | ---: | ---: |
| Raw matrix maximum error vs libcint / Eh | 3.47e-12 | 3.47e-12 |
| RHF energy error / Eh | 2.44e-15 | 2.00e-15 |
| RHF force maximum error / Eh/bohr | 2.46e-12 | 5.54e-13 |
| UHF energy error / Eh | 4.44e-16 | 4.44e-16 |
| UHF force maximum error / Eh/bohr | 9.79e-16 | 8.40e-16 |
| Complete warm RHF API median / ms | 1665.4 | 1049.4 |
| Complete warm UHF API median / ms | 1774.9 | 1042.7 |

The local-matrix coarse/refined difference is 3.47e-12 Eh; the local-derivative
difference is 2.03e-11 Eh/bohr. Maximum CPU/GPU differences are below 1.4e-17
for these raw matrices and derivatives. Independent-center and d-shell tests
use separate fixtures and tolerances documented in the test source.

Component API medians (value/gradient, ms) are CPU local 129.0/167.7,
nonlocal 126.2/158.9; CUDA local 90.5/197.5, nonlocal 88.9/196.7.
These zero the other component's coefficients while retaining the full engine;
they include allocation, context, transfer and synchronization overhead. They
are not isolated kernel timings. GPU gradients are slower for this tiny raw
fixture; no specialized schedule or general speedup is promoted.

Validation: ECP/IR on CUDA 25 passed; CPU ECP/external-basis/resource suite
79 passed, 3 GPU-only skips; affected compiler/Fock/resource/ownership suites
281 passed, 2 environment skips. CPU CTest: 16/16. CUDA CTest: 18/19; the
unmodified `vibeqc_mixed_precision_tests` fails with `cached plan did not enter
isolated Fock mode: CUDA runtime error`. The same failure was reproduced on
the pre-ECP #170 build on this device. It remains a separate baseline issue.

Reproduce with the CPU/CUDA CMake flags and test commands in `docs/ecp.md`.
For timing, set `PYTHONPATH=python`, `VIBEQC_LIBRARY` to the exact library,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, then run:

```sh
python tools/benchmark_ecp.py --device cpu --output benchmark-cpu.json
python tools/benchmark_ecp.py --device cuda --output benchmark-cuda.json
```

Raw build/test logs remain in the authorized `evidence-171` workspace; only
selected numerical summaries and reproducible source are retained in Git.

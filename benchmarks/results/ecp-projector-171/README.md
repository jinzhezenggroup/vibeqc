# Generated scalar ECP radial/projector migration

The original RTX 4090 record below did not capture the measured candidate
revision or dirty state. It is retained as historical data, and is not the
qualification record for this merge. Fresh, source-bound RTX 5090 validation
is retained in [review-qualification](review-qualification/README.md); no
missing historical provenance has been inferred.

Measured 2026-09-15 against master `743414dcb44a4a45821b900753dc1d0c4c1cf5c9`
on an Inspire RTX 4090, driver 570.124.06, CUDA 12.9, architecture 89,
Release without AOT shells. Python/BLAS workers were limited to two.
`measurements.json` retains both binary identities, measured source hashes,
all timing samples, physical kernel resources, raw/reference errors and
resource budgets. The unchanged CPU ECP implementation is an independent
oracle/fallback. The [capability domain](../../../docs/ecp.md) is unchanged.

The compiler owns residual radial integrands (powers 0..4), spherical
projection reductions, local/nonlocal bilinear derivatives and ECP-center
translation recovery. Projected jets are returned as local value records:
this preserves the original per-thread accumulation layout without exposing
an aliased output pointer inside the angular reduction.

## Numerical and complete endpoint checks

| NaH / NaH+ check | Baseline | Generated |
| --- | ---: | ---: |
| Matrix max error vs Libcint / Eh | 3.47e-12 | 3.47e-12 |
| RHF energy error / Eh | 2.22e-15 | 2.22e-15 |
| RHF force max error / Eh/bohr | 5.77e-13 | 6.01e-13 |
| UHF energy error / Eh | 4.44e-16 | 4.44e-16 |
| UHF force max error / Eh/bohr | 8.40e-16 | 7.98e-16 |
| Complete warm RHF API median / ms | 1076.57 | 1052.15 |
| Complete warm UHF API median / ms | 1069.07 | 1045.92 |
| Local value API median / ms | 92.93 | 95.40 |
| Local gradient API median / ms | 199.61 | 188.68 |
| Nonlocal value API median / ms | 91.95 | 95.03 |
| Nonlocal gradient API median / ms | 199.48 | 188.53 |

These are three warm synchronous samples per endpoint, baseline followed by
candidate. Component measurements zero the other component's coefficients;
they retain the full engine, allocations and transfers. The small-fixture
comparison is a regression check, not a statistical speedup or schedule
promotion. Independent Libcint and finite-difference tests cover every radial
power and d projectors, distinct/coincident physical centers, nonsymmetric
weights, Cartesian/spherical d orbitals, and complete RHF/UHF force/replay.

## Code and resources

- Maintained CUDA adapter: 324 to 287 nonblank/noncomment lines, with no
  reclassification of the remaining mixed scientific/runtime adapter.
- Generated header: 345 to 492 lines; 9,179 to 13,625 bytes.
- ECP CUDA object: 159,336 to 164,736 bytes; full library grows by 3,992 bytes.
- `project`: 28 registers, 64-byte stack in both builds; `contract`: 72
  registers, 32-byte stack in both builds. All other ECP kernel register and
  stack counts also remain equal; reported local memory is zero.
- Planned device peaks remain 2,343,376 bytes for RHF and 2,378,528 for UHF;
  host/pageable peaks also match. No radial-shell staging allocation changes.

## Validation and reproduction

CPU native CTest: 29/29. ECP/IR/input tests: CPU 42 passed with 10 explicit
GPU skips; CUDA 52 passed. The independent generated-host C++ projector test
uses a double angular-node sum and the Legendre addition theorem, covering
all local/s/p/d channels, powers 0..4, mixed-exponent signed terms, radial
origin, center filtering and poisoned unused derivative slots.
Both CUDA ECP native tests pass, including allocation failure cleanup,
error classification and recovery. Compute Sanitizer 12.8 memcheck reports
zero errors for the CUDA error/recovery test running the CUDA 12.9 build.

Use the build flags in `docs/ecp.md`, then:

```sh
export PYTHONPATH=python
export VIBEQC_LIBRARY=/absolute/build/path/libvibeqc.so
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
ctest --test-dir /absolute/build/path -R 'vibeqc_ecp_' --output-on-failure
VIBEQC_ECP_CUDA_TEST=1 python -m pytest tests/python/test_ecp.py tests/python/test_ecp_ir.py tests/python/test_ecp_validation.py -q
python tools/benchmark_ecp.py --device cuda --repeats 3 --output comparison.json
compute-sanitizer --tool memcheck --error-exitcode 99 /absolute/build/path/vibeqc_ecp_cuda_error_tests
```

Raw build/test logs and the initial full versus incremental build resource
measurements remain under the authorized `evidence-171` directory; those
different build scopes are not compared as kernel compilation speedups.
DFT/ECP integration, higher angular domains and remaining native scientific
portions stay open under #171. This change is **Refs #171**, not full closure.

# Generated CPU integral backend

The generated CPU integral path keeps the scientific Gaussian-integral
recurrence in the existing compiler DAG. CPU specialization begins only after
that DAG has been built.

```text
IntegralIR / ShellClassComponentKernel
                |
        algebra / CSE policy
                |
           CpuScheduleIR
                |
       +--------+--------+
       |        |        |
    scalar     AVX2   AVX-512F
     1 lane   4 FP64    8 FP64
```

## Lane mapping

The first SIMD schedule packs independent primitive records into lanes. Runtime
input records remain the established AoS ABI. A bounded execution tile
transposes the active records to an internal SoA layout so each exponent or
Cartesian coordinate is loaded contiguously by the generated vector kernel.

Incomplete final tiles are not read past the caller's input. Inactive lanes are
filled with finite coordinates, positive unit exponents, and zero contraction
weight. They therefore exercise an ordinary well-defined recurrence without
contributing to the result.

## Numerical policy

The scalar, AVX2, and AVX-512 candidates are lowered from the same
`ShellClassComponentKernel`; there is no shell-name-specific AVX recurrence.
Ordinary recurrence arithmetic uses backend vector values. Numerically
sensitive Boys construction and general transcendental operations are evaluated
with the strict scalar C++ library function independently in each active SIMD
lane, then loaded back into vector values.

Compilation uses explicit target flags such as `-mavx2 -mfma` or
`-mavx512f -mfma`, never `-march=native`. The baseline has no SIMD ISA
requirement. `-ffp-contract=off` remains explicit, and no fast-math mode is
enabled. FMA is a schedule decision rather than a global numerical-policy
change. The initial target record accepts only its exact declared ISA flags;
arbitrary extra compiler options and non-integer lane widths are rejected.
See the [arithmetic admission decision](../.agents/notes/implemented/numerics/2026-09-19-cpu-target-arithmetic-admission.md).

## Validation and benchmark

`tests/python/test_cpu_lane.py` compiles generic, AVX2, and AVX-512 source
from one ERI DAG. On supporting hosts it executes the ISA-specific candidates,
including a partial SIMD tail, and compares raw values plus all twelve
four-center nuclear derivatives with the independent native integral oracle.

The retained node3 benchmark is
`benchmarks/results/issue469-cpu-simd.json`. Reproduce it with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python \
  python benchmarks/issue469_cpu_simd.py \
  --shell fdps --records 2048 --samples 80
```

For the measured FDPS component, the generic generated candidate took
2.464011 ms median and AVX2 took 1.2937415 ms, a 1.9046x speedup. Their final
outputs had zero observed absolute difference. The same host compiled the
AVX-512 candidate but did not advertise AVX-512F, so that binary was not
executed.

This qualification is a generated-kernel result, not a claim that the current
SCF CPU default has been replaced. Portable runtime ISA selection and
multi-variant dispatch are tracked separately by #470; CPU cost-model and
schedule autotuning are tracked by #471. The independent native oracle remains
outside this generated production-candidate path.

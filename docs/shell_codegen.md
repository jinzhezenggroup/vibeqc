# Shell-class CUDA code generation

## Objective

Move automatic differentiation out of hot GPU execution and into a compiler
pipeline:

```text
integral recurrence -> expression DAG -> symbolic derivatives -> CSE
                    -> shell-class CUDA -> AOT or cached NVRTC
```

The runtime kernel should contain ordinary scalar FP64 arithmetic, explicit
analytic derivatives, and no `Dual3` propagation. The existing hand-written
closed recurrences and finite-difference tests remain the correctness oracle
while generated coverage is introduced incrementally.

## Current pilot

`tools/qce_codegen` implements an interned scalar expression DAG, local
constant folding, symbolic differentiation, structural common-subexpression
elimination, deterministic CUDA emission, and a complete canonical
`(p s|s s)` primitive-gradient pilot. Boys values are external leaves with the
analytic rule `dF_n(T)/dT = -F_(n+1)(T)`, so the generator can share one Boys
sequence without embedding a numerical special-function implementation into
the algebra graph. Three centers are differentiated independently and the
fourth is restored by translational invariance.

The pilot is intentionally not dispatched by the production force kernel yet.
Its purpose is to validate the compiler stages before generating the
order-five shell classes that dominate the 96-AO profile. Generated source can
be inspected with:

```bash
python tools/generate_shell_kernels.py --shell-class psss --axis x
python tools/generate_shell_kernels.py --shell-class psss --axis x --format stats
cmake --build build --target qce_codegen_pilot
```

The optional CMake target writes deterministic AOT candidates under the build
tree and does not add a Python dependency to normal library compilation.

## AOT-first policy

AOT generation is the default for shell classes that account for material
time in the fixed 96-AO and 192-AO gates. Each generated function should be
checked into the source tree, compiled for the configured CUDA architectures,
and covered by:

1. generated expression versus the generic analytic/AD oracle;
2. finite differences for energy and all center derivatives;
3. translation and shell-permutation invariants;
4. `cuobjdump` register, stack, shared-memory, and code-size limits;
5. isolated shell-class timing before end-to-end gate timing.

The first production target should be the highest-work canonical order-five
class observed in a task histogram, not a generic total-angular-order kernel.
Its generated contraction should share primitive-pair quantities, Boys values,
and derivative states across all Cartesian components handled by one worker.

For the 96-AO WATER27 tetramer, the pre-screen direct topology contains three
order-five classes. Weighting unique Cartesian AO quartets by primitive
contractions gives `dppp` 54.9% of order-five primitive work, `dpds` 29.7%,
and `ddps` 15.5%. An opt-in final-density profiler now measures the exact tile
list retained after Schwarz and density screening without adding work to
normal timing runs. At the formal `1e-14` screening threshold it records:

| Gate point | `dppp` | `dpds` | `ddps` |
| --- | ---: | ---: | ---: |
| 96 AO, batch 1 | 58.85% | 26.83% | 14.32% |
| 96 AO, batch 4 | 58.97% | 26.73% | 14.30% |
| 192 AO, batch 1 | 60.13% | 26.60% | 13.27% |
| 192 AO, batch 4 | 60.12% | 26.61% | 13.27% |

These are fractions of active order-five primitive contractions, aggregated
across each batch after the final converged-density Fock compaction. They
confirm `dppp`, the canonical `(d p|p p)` class, as the first production
generation target at both realistic sizes. Reproduce the topology and active
reports with:

```bash
PYTHONPATH=python:benchmarks build/gpu4pyscf-venv/bin/python \
  benchmarks/shell_class_histogram.py --angular-order 5

env -u CUDA_VISIBLE_DEVICES PYTHONPATH=python:benchmarks \
  build/gpu4pyscf-venv/bin/python benchmarks/shell_class_histogram.py \
  --active --batch 1 --angular-order 5
```

Generated AOT coverage is deliberately sparse. Specialization increases NVCC
time, binary size, and instruction-cache pressure, and excessive unrolling can
also increase registers or local stack. QCE therefore keeps one compact generic
fallback/oracle and checks in only profile-backed classes that pass resource and
endpoint gates; it does not AOT-expand all 55 s/p/d/f classes.

## NVRTC fallback and cache

NVRTC is reserved for long-tail shell classes or experimental precision
policies. A cache entry must be content-addressed by all source and binary
compatibility inputs. `NvrtcCacheSpec` currently covers:

- generator ABI and generated-source digest;
- canonical shell class and differentiated-center mask;
- precision and screening policies;
- compute capability, NVRTC version, and driver version.

Compilation should occur outside timed execution. Cache publication must use a
temporary file followed by an atomic rename, with a per-key lock so concurrent
processes cannot publish partial artifacts. Failed compilation is never
cached. A loaded kernel must still pass launch/resource validation, and an AOT
kernel always takes precedence when both implementations exist.

## Next implementation slice

1. Extend the DAG builder with Hermite/Rys recurrence primitives and generate
   the dominant order-five Cartesian shell class.
2. Emit one cooperative `dppp` contraction kernel rather than one scalar
   function per AO quartet, so primitive bra-pair intermediates are reused
   across ket work.
3. Compare generated code against the existing order-five analytic path and
   retain it only if energy/force gates and resource/performance gates pass.
4. Generalize the successful template to the remaining high-frequency classes,
   then add NVRTC loading only for uncovered long-tail classes.

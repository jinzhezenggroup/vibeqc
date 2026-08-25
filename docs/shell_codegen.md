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
and `ddps` 15.5%. Screening can change these fractions, so `dppp` is the first
candidate rather than a performance conclusion. Reproduce the topology report
with:

```bash
PYTHONPATH=python:benchmarks build/gpu4pyscf-venv/bin/python \
  benchmarks/shell_class_histogram.py --angular-order 5
```

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

1. Add a shell-quartet task histogram to the 96-AO component harness.
2. Extend the DAG builder with Hermite/Rys recurrence primitives and generate
   the dominant order-five Cartesian shell class.
3. Emit one cooperative contraction kernel rather than one scalar function per
   AO quartet, so primitive bra-pair intermediates are reused across ket work.
4. Compare generated code against the existing order-five analytic path and
   retain it only if energy/force gates and resource/performance gates pass.
5. Generalize the successful template to the remaining high-frequency classes,
   then add NVRTC loading only for uncovered long-tail classes.

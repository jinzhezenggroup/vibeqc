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

## Current generators

`tools/qce_codegen` implements an interned scalar expression DAG, local
constant folding, symbolic differentiation, structural common-subexpression
elimination, deterministic CUDA emission, and a complete canonical
`(p s|s s)` primitive-gradient pilot. Boys values are external leaves with the
analytic rule `dF_n(T)/dT = -F_(n+1)(T)`, so the generator can share one Boys
sequence without embedding a numerical special-function implementation into
the algebra graph. Three centers are differentiated independently and the
fourth is restored by translational invariance.

The compact pilot is intentionally not dispatched by the production force
kernel yet. The same pipeline now also builds an exact canonical `dppp`
Cartesian component using the subset/Wick pair expansion and closed Coulomb
derivatives used by the order-five production oracle. It supports two
lowerings:

- `full` embeds primitive geometry, the Boys argument and the complete
  component derivative in one scalar function;
- `factored` consumes product-center geometry, pair shifts, Boys values, the
  primitive prefactor and decay derivatives computed once by a cooperative
  shell-quartet worker.

Generated source can be inspected with:

```bash
python tools/generate_shell_kernels.py --shell-class psss --axis x
python tools/generate_shell_kernels.py --shell-class psss --axis x --format stats
python tools/generate_shell_kernels.py --shell-class dppp \
  --d-component xy --p-components xyz --lowering factored --format stats
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

### First `dppp` lowering result

The mixed-axis `d=xy`, `p=xyz` component is the current resource probe because
it exercises all Cartesian axes without duplicating the full 162-component
shell contraction. On NVCC 12.9 targeting `sm_120`, symbolic factoring changes
the generated component as follows:

| Lowering | Reachable DAG nodes | CUDA source | Registers | Stack/spill | ptxas time |
| --- | ---: | ---: | ---: | ---: | ---: |
| full, dynamic primitive input | 2,249 | 2,246 lines / 81,073 B | 255 | 128 B / 128 B | 154 ms |
| factored, dynamic precomputed-geometry input | 959 | 942 lines / 35,695 B | 180 | 0 B / 0 B | 41 ms |

The factored lowering is therefore the only production candidate. Independently
emitting all 162 scalar components would still produce roughly 5.8 MB even at
the factored representative size and would multiply instruction-cache and NVCC
costs. Production integration must instead distribute Cartesian components
across cooperative lanes and hoist common pair/Coulomb states beyond what the
single-component probe can express. Until that kernel passes correctness,
resource and endpoint performance gates, generated `dppp` source remains under
the build tree and is not checked into the source tree.

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

1. Emit one cooperative `dppp` contraction kernel rather than one scalar
   function per AO quartet, so primitive bra-pair and Coulomb intermediates are
   reused across Cartesian components and live ranges remain lane-local.
2. Compare generated code against the existing order-five analytic path and
   retain it only if energy/force gates and resource/performance gates pass.
3. Check in the generated header only after those gates pass, then add a CI
   regeneration check that requires an empty diff.
4. Generalize the successful template to the remaining high-frequency classes,
   then add content-addressed NVRTC loading only for uncovered long-tail classes.

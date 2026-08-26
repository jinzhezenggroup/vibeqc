# Shell-class CUDA code generation

## Goal

Move integral differentiation and shell specialization out of GPU execution
and into a deterministic compiler pipeline:

```text
Integral IR -> Schedule IR -> CUDA -> correctness/resource/timing gates
```

Generated kernels contain ordinary FP64 arithmetic and no runtime automatic
differentiation objects. The maintained surface is the generator, the
backend-independent oracle, the bounded schedule search space, and the
architecture manifest; generated production CUDA remains a build artifact.

## Current pipeline

`tools/qce_codegen/ir.py` separates mathematical intent from execution policy:

- `IntegralIR` describes a canonical shell class and its consumers (`fock`,
  `force`). Force differentiates centers 0, 1, and 2 and restores center 3 by
  exact translation invariance.
- `ScheduleIR` describes task/component ownership, block size, component tile,
  Coulomb-state placement, pair orientation/storage, and loop unrolling.
- `KernelIR` combines the two and validates component coverage before CUDA is
  emitted.

The subset/Wick recurrence is shared by ERI values and analytic gradients.
One plan can therefore emit RHF/UHF Fock and force kernels from the same
mathematical definition. The Fock lowering deliberately emits a value-only
Coulomb table, while the force lowering adds the one derivative order needed
for analytic nuclear gradients. The production registry now exposes both
consumers; `dpps` is the first class selected for generated Fock and force.

The mathematical catalog contains all 55 canonical s/p/d/f quartet classes,
including zero-order `ss` pairs. Current CUDA lowering supports pair orders
zero through six and provides four schedule families:

| Schedule | Mapping | Current status |
| --- | --- | --- |
| `packed_tasks` | one independent small-shell task per lane | emitted and benchmarked |
| `shell_task` | one warp/block cooperates on one shell task | emitted and benchmarked |
| `component_lanes` | one Cartesian component per lane | production path |
| `tiled_components` | a bounded component slice per block | emitted and benchmarked |

Tiled lowering removes the former 1024-component limit. For example, `dddd`
has 1296 Cartesian quartets and defaults to a 64-component tile; `fddd` has
2160. State packing, Coulomb indices, Wick multiplicities, and f-component
axis tables widen automatically for these classes.

### Automation boundary

High-performance first nuclear gradients are now generator-owned for the full
canonical s/p/d/f catalog: shell algebra, Cartesian component decoding,
translation recovery of center 3, RHF/UHF density contraction, CUDA scheduling,
resource inspection, and production source generation do not require a new
handwritten kernel per class. Promotion is still measured rather than assumed;
the handwritten fallback remains when a generated candidate loses end to end.

This is not yet a general Hessian or arbitrary derivative-order compiler.
`KernelConsumer.FORCE` currently means one nuclear derivative, the output ABI
is a force vector, and the oracle exploits first-derivative translation
invariance. Second and higher nuclear derivatives would require an explicit
derivative-order IR, tensor symmetry/layout, higher-order translation rules,
new correctness oracles, and separate resource/timing gates.

## Correctness model

The host oracle evaluates the same factored recurrence independently of CUDA.
Tests cover:

1. symbolic full-integral derivatives versus factored lowering;
2. generated recurrence values and all four center gradients;
3. finite differences, translation, and shell-permutation invariants;
4. every `dpds`/`ddps` component and representative `dddd`, `ffps`, `fddd`,
   `psss`, and `ssss` components;
5. real CUDA 12.9 compilation for joint Fock/force, shared/recomputed Coulomb,
   tiled d/f shells, zero-order pairs, and packed tasks.

The committed handwritten kernels remain endpoint oracles and performance
goldens. In particular, the `psss` force kernel added by `032f497` combines
all three weighted Cartesian outputs in one primitive traversal. Generated
low-order code must match that arithmetic quality before replacing it.

## Architecture autotuning

`tools/qce_codegen/autotune.py` emits every CUDA-supported schedule variant
with unique symbols, compiles the translation units in parallel, links them
into one executable, and runs all variants in one GPU allocation. A candidate
is rejected for:

- CUDA compilation failure;
- spills, excessive registers, stack, or shared memory;
- Fock-value or force disagreement with the independent recompute oracle;
- failure to meet the configured timing threshold.

Passing variants are ranked by measured kernel time. The winner can be written
to a schema-v2, architecture-specific production manifest:

```bash
python -m tools.qce_codegen.autotune \
  --nvcc /group/software/cuda-12.9.1/bin/nvcc \
  --architecture sm_120 \
  --shell-class dpds \
  --partition main --gres gpu:5090:1 \
  --output build/dpds-autotune.json \
  --manifest-output build/production-shells-tuned.json
```

Use `--consumer fock` to tune the value-only SCF worker with the same resource,
correctness, and timing gates. A Fock winner upgrades the manifest row to the
joint `fock`/`force` consumer set because both kernels share the canonical task
ABI.

The manifest records every code-shape decision rather than relying on emitter
defaults. CMake selects a profile with `QCE_AOT_CODEGEN_ARCHITECTURE`.

Large-shell tuning uses a staged compiler pipeline. Equivalent schedules share
one separately compiled correctness oracle per component mapping; tiled oracle
recurrences are noinline so NVVM does not expand the reference into every
candidate. Timing translation units contain only the RHF kernel they execute.
After ranking, the tuner recompiles the fastest passing candidate with all four
RHF/UHF and persistent production wrappers and applies the resource gates again
before writing a manifest. `--compile-timeout` bounds every NVCC invocation and
terminates its entire process group, preventing timed-out `cicc` children from
surviving into later trials.

### Measured schedule evidence on `sm_120`

The largest recent end-to-end gain came from optimizing shared recurrence
structure rather than promoting one more exact class. For small Boys arguments,
the runtime and generated CUDA now evaluate only the highest requested order by
power series and recover lower orders by downward recurrence. The committed
192-AO batch-one measurement improved from about 4.49 s to 3.60 s (`1.25x`).
Keeping the Boys implementation in production and benchmark templates means
this algebraic optimization automatically reaches every generated class.

The complete Fock autotune uses 128 synthetic `dpps` tasks with two primitives
per shell. Searching both pair orientations expands the component schedule
space from four to eight variants. Canonical/shared/unrolled value lowering
remains the winner at 0.365 ms versus 0.485 ms for independent per-component
recomputation (`1.33x`), with maximum Fock disagreement `1.39e-17`, at most
138 registers, zero stack, and zero spills. Swapped/shared/unrolled is only
`0.5%` slower at 0.367 ms, while swapping improves the rolled shared variant
from 0.401 ms to 0.369 ms. Both recomputed-Coulomb orientations lose, and both
unrolled variants spill. The measured winner matches the current production
`dpps` schedule, but endpoint promotion remains governed by the separate
real-molecule gate.

The same eight-way search on the generated `dpps` first-gradient consumer
selects swapped/shared/unrolled at 0.604 ms, versus 0.630 ms for the previous
canonical orientation (`1.044x`). It is `2.89x` faster than the independent
per-component gradient oracle, with maximum force disagreement `8.67e-19`,
151 registers, a 40-byte stack, and zero spills. Pair orientation changes only
the contraction loop/materialized pair, not physical center routing, so this
gain required no shell-specific gradient algebra. Because the joint production
row also owns Fock, a real-molecule endpoint gate must decide whether to retain
one compromise schedule or justify consumer-specific schedules.

The first full tiled search covers 24 `dddd` variants: tile 64/128/256,
canonical/swapped contraction order, rolled/unrolled loops, and materialized
versus recomputed pair storage. Materializing a 16-entry order-four pair table
puts 720--1136 bytes in the per-thread stack and makes every original variant
fail; unrolled variants also spill. Recomputing pair terms removes the array
and lowers the accepted production wrappers to an 80-byte stack with zero
spills. Autotuning selects tile 128/shared/unrolled/swapped/recomputed at
5.302 ms, with maximum force disagreement `4.34e-19`, 168 registers, and all
four production wrappers passing. Tile 256 is arithmetically faster in several
variants but spills, demonstrating why tile timing alone is insufficient.

The same search succeeds for the 2160-component `fddd` class. The selected
tile 128/shared/rolled/canonical/recomputed schedule runs in 42.789 ms, has
maximum force disagreement `8.67e-19`, and uses at most 160 registers, an
80-byte stack, and zero spills across all production wrappers. The shared
noinline oracle is a correctness reference rather than a production-speed
baseline; candidate ranking uses fused kernel time, while real-molecule
promotion still requires the endpoint gate.

For 128 synthetic `dpds` tasks with two primitives per shell, the bounded
search produced:

| Schedule | Time | Registers | Stack | Shared | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| component/shared/unrolled | 1.880 ms | 156 | 40 B | 1880 B | winner |
| component/shared/rolled | 1.941 ms | 154 | 40 B | 1880 B | accepted |
| tiled 64/shared/unrolled | 3.503 ms | 162 | 40 B | 1688 B | accepted |
| component/recomputed/unrolled | 4.489 ms | 168 | 40 B | 1216 B | too slow |
| component/recomputed/rolled | 4.159 ms | 163 | 208 B | 1216 B | stack/slow |

The measured winner matches the current production `dpds` policy. Tiling is
correct and resource-safe, but is retained for larger component products where
the non-tiled mapping is impossible.

For 128 synthetic `ddps` tasks with two primitives per shell, the matching
component/shared/unrolled schedule ran in 2.022 ms versus 4.515 ms for the
independent recompute oracle (`2.23x`). With primitive-pair cache reuse it uses
at most 164 registers, a
64-byte stack, 1880 bytes of shared memory, and zero spills. The real-spherical
def2-SVP water-tetramer promotion gate then measured `1.0099x` speedup for
batch one and `1.0078x` for batch four, with maximum energy and force
differences of `1.48e-12 Eh` and `4.01e-13 Eh/bohr`. This positive endpoint
result promoted `ddps` to the `sm_120` production profile.

An active def2-TZVP water profile shows that classes containing at least one
f shell account for `46.6%` of screened primitive work. `fpps` is the largest
single f-shell class at `4.49%`. Its generated component/shared/unrolled
schedule ran in 1.989 ms versus 4.514 ms for the recompute oracle (`2.27x`),
using at most 157 registers, a 64-byte stack, and zero spills. A 15-sample
real-molecule gate measured `1.0022x`, `1.0054x`, and `1.0060x` end-to-end
speedups for batches one, four, and eight. The maximum force difference was
`7.17e-13 Eh/bohr`. `fpps` is therefore the first production f-shell gradient
class selected entirely by the generic IR, emitter, autotuner, and endpoint
gates.

Shell-wide density-weighted CSE now interns all three `psss` component DAGs in
one graph, shares primitive geometry, contracts the density weights before the
recurrence, emits only centers zero through two, and restores center three by
translation. The independent component graphs contain 375 nodes in total;
the weighted graph contains 318.

For 1024 synthetic `psss` tasks with two primitives per shell, ten iterations,
and seven timing samples, autotuning selected a 32-thread packed schedule at
0.539 ms versus 0.932 ms for the independent recompute oracle (`1.73x`). The
kernel uses 208 registers, a 96-byte stack, and zero spills on `sm_120`.

That kernel still does not beat the committed handwritten endpoint. On the
real-spherical def2-SVP water tetramer after one-pass bucketing, enabling
generated `psss` produced `0.9933x` speedup for batch one and `0.9952x` for
batch four. Maximum energy and force differences remained below `1.0e-12 Eh`
and `6.4e-13 Eh/bohr`, respectively. The candidate therefore remains outside
the production manifest: synthetic improvement against a recompute oracle is
not sufficient evidence for promotion over a tuned handwritten kernel.

## Production AOT policy

`production_shell_classes.json` is architecture-specific. The current
`sm_120` profile contains force kernels for:

```text
dppp dpds ddps fpps ppps dpps dsps dspp
```

The generated registry records class index, consumer mask, block size, and
component tile. CUDA shards are generated in the build directory and consumed
through a stable C ABI, which keeps generator changes isolated from
`src/scf/cuda_rhf.cu`.

The `dpps` production row enables both `fock` and `force`. Its coefficient-only
Fock worker measured `1.02249x` end-to-end speedup on the real-spherical
def2-SVP S4 water octamer, with maximum energy and force differences of
`1.36e-12 Eh` and `4.58e-12 Eh/bohr`. `ppps` and `dsps` Fock candidates were
rejected by the same endpoint gate, demonstrating that automatic emission does
not imply automatic promotion.

Adding a class to the mathematical catalog does not make it production AOT.
Production promotion requires:

1. oracle and CUDA compile gates;
2. zero spills and architecture resource limits;
3. isolated schedule timing;
4. real molecular energy/force gates;
5. end-to-end improvement after task classification/dispatch overhead.

Sparse, profile-backed AOT remains intentional. Generating all 55 classes is
now possible mathematically, but compiling and dispatching all of them by
default can increase NVCC time, binary size, instruction-cache pressure, and
queue-management cost.

## Task classification and merge discipline

The runtime now buckets all enabled generated classes together:

```text
classify/count once -> 55-entry device prefix -> materialize class slices
                    -> Fock/force registry dispatches (offset, count, head)
```

The classification byte for each active logical quartet survives the prefix
kernel, so materialization does not decode the exact shell class again. Counts,
offsets, write cursors, and persistent-worker heads stay on the device; no host
readback or synchronization is introduced. Adding another generated class no
longer adds another full active-tile scan.

Before the incremental `ddps` promotion, the six generated force classes made
the water-tetramer endpoint
`1.065x` faster than the all-handwritten fallback for batch one and `1.089x`
for batch four. Maximum energy and force differences are `8.0e-13 Eh` and
`4.8e-13 Eh/bohr`. The seventh class, `ddps`, independently adds the positive
`1.0099x`/`1.0078x` endpoint improvement reported above. The eighth class,
`fpps`, adds a smaller but repeatable `1.0022x` to `1.0060x` on the f-shell
def2-TZVP workload. These results include bucketing and registry dispatch.

`cuda_rhf.cu` is also under active development in `../qc`. The codegen branch
is synchronized only from committed upstream history; the latest integration
fast-forwarded through `84152e3`, including the generated `dpps` Fock route,
highest-order-only Boys-series evaluation, primitive-pair cache reuse in both
handwritten and generated Fock/force kernels, and resident-bra chunking for the
handwritten `psss` force fallback.
Uncommitted or untracked state in `../qc` is never copied, overwritten, or used
as a merge source. IR, emitters, autotuning, and manifests remain in independent
modules so future upstream merges touch the hot runtime file only when dispatch
behavior actually changes.

## Commands

Inspect deterministic source:

```bash
python tools/generate_shell_kernels.py \
  --shell-class dpds --lowering fused --consumer fock
python tools/generate_shell_kernels.py \
  --shell-class dppp --lowering fused --format stats
python tools/generate_shell_kernels.py \
  --shell-class fddd --lowering fused --format stats
cmake --build build --target qce_codegen_pilot
```

Candidate batch screening is intentionally sparse: pass either an explicit
`--shell-class` list or a real-profile `--profile` plus `--limit`. Omitting
both is rejected so the tool cannot accidentally compile every uncovered
class in one batch.

Run Python gates:

```bash
python -m pytest tests/python/test_codegen.py -q
python -m ruff check tools/qce_codegen tests/python/test_codegen.py
```

Run the explicit CUDA gate:

```bash
QCE_NVCC=/group/software/cuda-12.9.1/bin/nvcc \
QCE_CUDA_ARCH=sm_120 \
python -m pytest tests/python/test_codegen.py -q -s
```

## Remaining work

1. Reduce the remaining generated `psss` endpoint gap or retain the handwritten
   kernel; do not promote the current synthetic winner.
2. Continue profile-driven f-shell promotion with `fsps` and the first tiled
   class; `fddd` now passes synthetic correctness/resource/timing gates, but do
   not infer endpoint value from primitive-work fraction alone.
3. Benchmark tiled d/f candidates on real molecular profiles, not only
   synthetic shell tasks.
4. Profile additional value-only Fock classes through the shared device slices;
   retain endpoint rejection as a normal outcome.
5. Generalize `IntegralIR` from first forces to explicit second/higher nuclear
   derivative orders before claiming Hessian automation.
6. Consider per-consumer production schedules only after an endpoint workload
   shows that the measured `dpps` Fock/force orientation tradeoff is material.

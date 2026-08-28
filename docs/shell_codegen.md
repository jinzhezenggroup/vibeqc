# Shell-class CUDA code generation

## Goal

Move integral differentiation and shell specialization out of GPU execution
and into a deterministic compiler pipeline:

```text
Integral IR -> backend lowering -> CUDA target/schedule IR
            -> CUDA -> correctness/resource/timing gates
```

Generated kernels contain ordinary FP64 arithmetic and no runtime automatic
differentiation objects. The maintained surface is the generator, the
backend-independent oracle, the bounded schedule search space, and the
architecture manifest; generated production CUDA remains a build artifact.

## Current pipeline

`tools/vibeqc_codegen/ir.py` is now strictly mathematical, while
`cuda_target.py` and `cuda_schedule.py` own NVIDIA execution policy:

- `IntegralIR` describes a canonical shell class and its consumers (`fock`,
  `force`). Force differentiates centers 0, 1, and 2 and restores center 3 by
  exact translation invariance.
- `CudaScheduleIR` describes task/component ownership, block size, component tile,
  Coulomb-state placement, pair orientation/storage, and loop unrolling.
- `CudaKernelIR` combines the two with a `CudaTargetInfo` and validates target
  limits and component coverage before CUDA is
  emitted.

The backend contracts in `backend.py` cover source emission, compilation,
resource parsing, device probing, benchmark execution, and registry emission.
NVCC process-group handling and finite Slurm execution live in the CUDA adapter,
not in the mathematical IR. Production code imports the generic CUDA emitter
surface; the historical `dppp` pilot is isolated behind a compatibility
specialization module.

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

## Experimental Rys backend

The compiler also contains a backend-independent Rys/TRR/HRR state IR. It
constructs a topologically ordered, duplicate-free one-dimensional recurrence
program from any catalog shell specification while retaining the existing
component order and translation recovery for center D. For `dddd`, this model
requires five roots, 1296 Cartesian components, 162 requested axis states, and
216 recurrence instructions. This exposes the high-order state surface without
expanding it into the subset/Wick scalar expression DAG.

An experimental CUDA lowering tests the approach on force-only `ppps`. It uses
the same persistent exact-class queue ABI as production, assigns one complete
shell task to each lane, evaluates a three-root interpolation table, and
contracts the explicit recurrence states immediately into nine force
accumulators. The three-root numerical data and interpolation are attributed to
GPU4PySCF/PySCF under Apache-2.0 in the generated source. Independent tests
cover the first six Boys moments, every Cartesian component, all four force
centers, and translation recovery.

The `sm_120` experiment rejects this lowering for production. The RHF and UHF
wrappers compile at 255 registers per thread, a 56-byte stack, 64-byte spill
stores, 64-byte spill loads, and 8528 bytes of shared memory. The 1,110,608-byte
cubin took 4.67 seconds to compile and embeds a 27,024-byte root table. In two
paired RTX 5090 runs, the Rys kernel took 1.315 ms and 1.032 ms versus 0.455 ms
and 0.456 ms for the production component-lane recurrence: a 2.89x and 2.27x
slowdown, respectively. Maximum force disagreement was `1.19e-13`.

Consequently, the `ppps` thread-task production dispatch remains unchanged. No
force endpoint was run for that candidate and no five-root `dddd` CUDA emitter
was added: the simpler three-root case already fails the zero-spill/resource
gate and loses the isolated timing gate by more than twofold. Reaching
high-order production performance requires a materially different cooperative
primitive/root/component mapping, not direct scaling of this thread-task
prototype. The full measurements are in
`benchmarks/results/rtx5090-ac39177-issue-3-rys3-rejection.json`.

`dppp` now exercises that cooperative alternative. One 192-thread block owns
one canonical task, lane zero evaluates the four-root interpolation once per
primitive quartet, and the 162 active component lanes execute the same
runtime-indexed one-dimensional TRR/HRR program. Each axis uses an explicitly
addressed `5x4` table and returns only its base value and three independent
first derivatives; this avoids both a 162-way divergent switch and the
shell-wide scalar state graph. The fourth center is still recovered from
translation after the six-warp force reduction. The fixed four-root table is a
reproducible, Apache-attributed slice extracted from GPU4PySCF.

On CUDA 12.9 `sm_120`, all four RHF/UHF ordinary and persistent wrappers use
168 registers, a 160-byte stack, zero spills, and at most 960 bytes of shared
memory. The checked-in 8192-task, three-primitive isolated gate improved from
218.329 ms to 183.241 ms (`1.191x`) with `8.32e-12` maximum force disagreement.
In the fixed-`dm0`, one-iteration 384-AO endpoint, the `dppp` kernel fell from
167.420 ms to 139.197 ms and the VibeQC median moved from 3.327 s to 3.298 s.
The smaller endpoint gain is expected: current profiling places the remaining
VibeQC/GPU4PySCF force-kernel gap across many exact shell classes rather than
inside `dppp` alone. The raw evidence is the
[isolated gate](../benchmarks/results/rtx5090-26ef747-dppp-cooperative-rys4-isolated.json),
[384-AO endpoint](../benchmarks/results/rtx5090-26ef747-384ao-dppp-cooperative-rys4.json),
and [kernel summary](../benchmarks/results/rtx5090-26ef747-384ao-dppp-cooperative-rys4-kernel-summary.csv).

The fixed-root component-lane emitter is now generalized across the measured
`dpps`, `dsps`, and `pppp` force classes, while their Fock consumers remain on
the validated value recurrence. Isolated 8192-task gates improved by `1.239x`,
`1.342x`, and `1.040x`, with maximum force disagreements of `6.51e-12`,
`1.13e-11`, and `6.12e-12`, respectively. All generated RHF/UHF ordinary and
persistent kernels remain spill-free on CUDA 12.9 `sm_120`.

The production decision uses the real fixed-`dm0` endpoint rather than the
isolated gate. In the three-class candidate profile, `dpps` fell from 148.233
to 124.349 ms and `dsps` from 139.192 to 129.825 ms, but `pppp` regressed from
99.936 to 102.041 ms. Production therefore promotes only `dpps` and `dsps` and
keeps `pppp` on subset/Wick. The accepted 384-AO median falls from 3.298 to
3.269 s; all three repeats use one SCF iteration on both engines, with maximum
energy and force disagreements of `1.68e-11 Eh` and `3.15e-8 Eh/bohr`. The
remaining endpoint is `0.655x` GPU4PySCF, so this is an incremental force
improvement rather than a claim that the large-system gap is closed. Raw
evidence is retained in the [accepted endpoint](../benchmarks/results/rtx5090-259c256-384ao-dpps-dsps-rys3.json)
and [candidate kernel summary](../benchmarks/results/rtx5090-259c256-384ao-cooperative-rys3-kernel-summary_cuda_gpu_kern_sum.csv).

The force-only density contraction now algebraically collapses the generic
eight-permutation orbit before entering each generated recurrence. Direct SCF
symmetrizes every accepted external or internally generated density, so the
orbit is exactly one Coulomb product and two exchange products, multiplied by
the diagonal and pair-equality orbit factors. This preserves the former
unique-permutation semantics while removing its nested comparison loop from
every Cartesian component. Exhaustive AO-index equality patterns agree with
the old RHF/UHF expressions to roundoff, and the allocated CUDA suites cover
the resulting production kernels. In the 384-AO fixed-`dm0` profile, `ppps`,
`psps`, `dppp`, `dpps`, and `dsps` move from
182.462/141.303/139.335/124.349/129.825 ms to
178.344/140.030/137.922/122.614/127.782 ms. The matching three-repeat endpoint
median is 3.258 s versus the preceding 3.269 s observation, with the same
one-iteration branch and existing energy/force limits. This is a small
instruction-path improvement; it does not close the remaining direct-force
architecture gap.

## Architecture autotuning

`tools/vibeqc_codegen/autotune.py` emits every CUDA-supported schedule variant
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
python -m tools.vibeqc_codegen.autotune \
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
defaults. The autotune driver queries the allocated device before invoking any
trial and exits on an architecture mismatch. Its artifact records real device
limits, driver/runtime versions, NVCC/PTXAS versions, generator ABI, and the
target-derived resource gates.

CMake selects profiles with `VIBEQC_AOT_PROFILE` or `VIBEQC_AOT_PROFILES`.
`auto` resolves exact measured profile, explicitly compatible profile, empty
`portable_cuda`, then generic fallback. `VIBEQC_ENABLE_AOT_SHELLS=OFF` omits all
generated shell objects.

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

`production_shell_classes.json` carries explicit tuned and portable profiles.
The current `sm_120` profile is measured; `portable_cuda` is intentionally
empty so unsupported targets retain generic correctness. The `sm_120` force
profile contains:

```text
dppp dpdp dddp dpss dsds ddss ddpp ddds dpds ddps fpps
ppps dpps dsps dspp pppp psps ppss dsss
```

The generated registry records profile identity, target compute capability,
class index, consumer mask, block size, and component tile. Every profile uses
architecture-suffixed C entry points and scoped device/type identifiers. Each
object target is compiled only for its intended SM; one runtime registry picks
the active `KernelSet` once per device. Architecture list order cannot change
the generated sources or dispatch behavior.

Compile-only builds are supported for `sm_80`, `sm_86`, `sm_89`, `sm_90`, and
`sm_120` with the project's CUDA 12.9 toolkit requirement. Only GPU-backed
performance claims are profile-specific; portable builds never apply RTX 5090
resource goldens.

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
cmake --build build --target vibeqc_codegen_pilot
```

Candidate batch screening is intentionally sparse: pass either an explicit
`--shell-class` list or a real-profile `--profile` plus `--limit`. Omitting
both is rejected so the tool cannot accidentally compile every uncovered
class in one batch.

Run Python gates:

```bash
python -m pytest tests/python/test_codegen.py -q
python -m ruff check tools/vibeqc_codegen tests/python/test_codegen.py
```

Run the explicit CUDA gate:

```bash
VIBEQC_NVCC=/group/software/cuda-12.9.1/bin/nvcc \
VIBEQC_CUDA_ARCH=sm_120 \
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

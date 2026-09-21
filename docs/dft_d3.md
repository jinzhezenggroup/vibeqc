# D3(BJ) production correction runtime

VibeQC represents additive geometry-only dispersion with
`DispersionCorrectionPrimitive` and an immutable `D3Spec`. The native library
now exposes a standalone production D3(BJ) correction owner for CPU and CUDA,
including ragged batches and fixed-topology changed-geometry replay. `Calculator`
now composes that owner automatically when its resolved `MethodIR` contains a
production D3(BJ) correction: the electronic subgraph remains the native KS model,
and the correction is added exactly once at the prepared execution boundary.
Issue #492 remains open for the still-unpromoted D3 variants and generated-runtime
performance/retirement work.

## Supported model and MethodIR composition

The production model is nonperiodic, real FP64, two-body D3(BJ), with `s9=0`.
`D3Spec` records explicit `s6/s8/a1/a2`, source-data SHA-256 identities,
coordination and pair cutoffs, and the pair-switch width. ATM and original
zero damping now have standalone CPU/CUDA qualification primitives, but neither
widens the public production owner: unsupported variants, invalid coefficients
and mismatched table identities are still rejected rather than silently
approximated.

The audited method catalog includes `PBE-D3(BJ)` and `PBE0-D3(BJ)`. Their
`MethodIR` graphs contain the normal semilocal/exact-exchange primitives followed
by one `DispersionCorrectionPrimitive`. The correction identity is independent
of a descriptive manifest name and is checked against the compiled table hashes
when a production owner is prepared.

Energy is in Hartree; coordinates are in bohr. The correction returns
**gradient = dE/dR**, including explicit pair-distance and coordination-number
response. Forces therefore have the opposite sign. The GFN1 halogen correction,
Hamiltonian, SCC state and D4 terms are not part of this endpoint.

## Calculator composition

The preferred composite entry point is a spin-explicit `MethodIR`. The method name
does not select D3 execution after resolution:

```python
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc_compiler.method import resolve_method

method = resolve_method("PBE-D3(BJ)", spin="unpolarized")
calc = Calculator(
    method=method,
    basis="sto-3g",
    device="cpu",
    ks_options=KsOptions(
        grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
    ),
)
result = calc.singlepoint(atoms, properties=("energy",))
print(result.energy, result.dispersion.energy)
```

A caller that already owns a native KS selector may equivalently provide the full
graph through `KsOptions(composition=...)`, for example `method="pbe-rks"` plus
the same PBE-D3(BJ) graph. `Calculator.method_ir` reports the full graph.
`Calculator.ks_options.method_ir` reports the electronic graph that is actually
lowered into the SCF owner. Direct native KS resolution still rejects correction
nodes, so no lower-level path can accept a D3 node and silently omit it.

At execution, the prepared electronic and D3 owners receive the same fixed atom
ordering and accepted geometry updates. For every successful item, composition is

```text
E_total = E_KS + E_D3
F_total = F_KS - dE_D3/dR
```

The subtraction is required because the D3 owner publishes a gradient, while the
public electronic endpoint publishes forces. `Result.dispersion` and
`BatchItemResult.dispersion` retain the correction component and backend evidence.
`PreparedBatch.dispersion_diagnostic` exposes the retained D3 plan identity and
resource inventory. A D3 per-item failure converts only that item to failure; a
malformed coordinate update already rejected by the electronic owner is not
reinterpreted by D3. This preserves the existing ragged per-item failure boundary.

The first automatic execution family is PBE-based MethodIR on RKS/UKS, including
PBE0 compositions for energy. Analytic total forces are exposed only when the
underlying electronic method/basis/backend already advertises an analytic force
endpoint; D3 composition never widens that electronic capability. In particular,
PBE0-D3(BJ) does not acquire public forces merely because the D3 gradient exists.

The D3 owner remains independently bounded by
`dispersion_memory_budget_bytes` (256 MiB by default). Whole-calculation
`ResourceBudget` / `estimate_resources()` does not yet aggregate the separate D3
owner and therefore fails closed for a composite calculation rather than reporting
an electronic-only budget as complete.

The architecture rationale and sign/failure invariants are recorded in the
[Calculator D3 composition decision](../.agents/notes/implemented/architecture/2026-09-21-d3-calculator-composition.md).

## Production API and ownership

```python
from vibeqc import D3CorrectionBatch, evaluate_d3_correction

one = evaluate_d3_correction("PBE-D3(BJ)", atomic_numbers, coordinates)
with D3CorrectionBatch(
    "PBE-D3(BJ)", systems, device="cuda", maximum_bytes=256 * 1024 * 1024
) as batch:
    cold = batch.execute()
    moved = batch.execute([new_coordinates, None])
    diagnostic = batch.diagnostic()
```

Preparation fixes each system's atom count and atomic numbers. Replay may replace
coordinates independently for every item while preserving ragged offsets.
`maximum_bytes` is an explicit plan bound. Diagnostics report plan/execution host
bytes, persistent device bytes, table/workspace bytes, system and atom counts,
and the MethodIR/correction/table identities.

The native owner copies all preparation inputs. CPU execution uses O(N) scratch
and direct pair loops. CUDA keeps offsets, atomic numbers, compact tables,
coordinates, masks, results and scratch resident behind one nonblocking stream;
only changed coordinates/masks and requested results cross the device boundary
per replay. The initial CUDA scheduling baseline uses one serial worker per
molecule while independent molecules run as separate blocks. It is a bounded
production ownership baseline, not a claim of pair-parallel performance.

## Data provenance and validation

No xTBloom, simple-dftd3 or dftd4 runtime dependency is added. The editable
source contract is the pinned snapshots and source manifest under
`tools/parameters/`, together with `method_parameter_overrides.json` for local
choices. Run `python tools/sync_dispersion_parameters.py` to regenerate the
committed intermediate `python/vibeqc_compiler/method/method_parameters.json`;
do not edit that intermediate by hand. Then run
`python tools/generate_method_parameters.py --python-output python/vibeqc_compiler/method/_generated_parameters.py`.
CMake uses the same intermediate and typed generator for native/CUDA `constexpr`
accessors, so calculation paths parse no configuration or upstream table.
The [source-ownership decision](../.agents/notes/implemented/architecture/2026-09-20-pinned-dispersion-catalog-sources.md)
records the input/update and regeneration contract. Build-time generation also
verifies the pinned xTBloom-derived D3 table and covalent-radius SHA-256 values
and emits only the compact production data needed by the native evaluator. The
runtime rejects a MethodIR whose recorded data identity differs from those
compiled tables.

`tests/data/d3_bj_reference.json` contains nine tiny independent fixtures from
simple-dftd3 1.4.0 with ATM explicitly disabled. The production tests use the
PBE/PBE0 fixtures for energy and complete analytic-gradient gates; the separate
migration suite also covers finite differences, covariance, cutoffs, parameter
scaling and malformed inputs.

```sh
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/python/test_d3_reference.py \
  tests/python/test_d3_production.py tests/python/test_dft_method_ir.py -q
python tools/check_compiler_structure.py
```

## Compiler GeometryIR/PairIR ownership

The two-body D3(BJ) scientific equation now also has a compiler-owned
GeometryIR/PairIR/TensorIR representation. CN response, C6 interpolation, BJ pair
energy and smooth switching lower through the shared TensorIR, and Cartesian
`dE/dR` is generated from that energy graph with the shared VJP. The compiler
representation uses an explicit fixed pair/switch-state identity.

Use the checked NumPy compiler-reference execution boundary for coordinate replay:

```python
from vibeqc_compiler.geometry import compile_d3_bj, execute_d3_bj

compiled = compile_d3_bj(spec, atomic_numbers, coordinates)
result = execute_d3_bj(compiled, new_coordinates, gradient=True)
energy = result["energy"]
gradient = result["gradient"]
```

`execute_d3_bj` copies and validates the coordinate snapshot before evaluating
primal outputs or the generated gradient. It rejects CN, pair-cutoff and
switch-region state changes; rebuild the compiled pair state for such geometry
changes. The generic TensorIR `execute(compiled.program, ...)` API and
`compiled.coordinate_vjp().program` are **unchecked low-level lowering interfaces**,
not executable D3 replay guards. Callers using them directly, including future
CUDA executors, must perform equivalent `compiled.validate_coordinates(...)`
preflight on the coordinates they actually execute. The checked reference API
does not select CUDA or substitute for the native production executor.

This is not yet the public ragged production execution path: `d3_bj.hpp` remains the
qualified native CPU/CUDA runtime and oracle until dynamic/ragged PairIR execution can
preserve the public batch/replay contract. The exact conventions, provenance,
underflow boundary, rejected alternatives and retirement condition are recorded in the
[D3 GeometryIR/PairIR decision](../.agents/notes/implemented/numerics/2026-09-20-d3-geometry-pair-ir.md).

## Generated ragged CUDA retirement candidate

The compiler now also has a non-public ragged CUDA execution candidate for the
same two-body D3(BJ) equation. `compile_d3_bj_batch` flattens a heterogeneous
molecular batch into one GeometryIR, preserves explicit system atom offsets, and
builds only within-system CN/energy pairs. Pair energies are reduced to a vector
of per-system energies with TensorIR `scatter_add`; the complete Cartesian
gradient is generated from that vector energy through one TensorIR VJP.

`PreparedD3CudaBatch` lowers the combined energy + generated-gradient graph
through the shared TensorIR CUDA planner/compiler/runtime. Replays that preserve
the CN/pair/switch state reuse the prepared artifact. A replay that crosses a
recorded topology or switch boundary prepares a replacement generated program
before the old prepared owner is released, so a failed rebuild cannot corrupt the
previous executable state.

This is deliberately a **retirement candidate, not the public production owner**.
The native `D3CorrectionBatch` remains authoritative until the generated route
has independent real-device qualification for numerical parity, resource bounds,
changed-topology replay, per-system failure isolation, energy-only execution and
endpoint performance. In particular, the current candidate evaluates the generated
gradient graph even when a caller only needs energy, and TensorIR's one-call batch
failure boundary is not yet equivalent to the public native per-item status ABI.

See the [generated ragged execution decision](../.agents/notes/implemented/numerics/2026-09-20-d3-generated-ragged-cuda.md).

## Remaining boundary

The public correction owner remains deliberately separate from the electronic DFT
SCF/Fock equation, but `Calculator` now owns their exact-once energy/force
composition at the prepared execution boundary. Remaining #492 work is the
pair-parallel/generated CUDA production promotion and retirement evidence, plus
production admission of the separately validated ATM and zero-damping variants.
The standalone ATM and D3(0) references do not grant nonzero-`s9` or zero-damping
production capability. Native DFT paths must continue to reject a correction node unless the Calculator
composition owner has explicitly split and retained it.

See [data provenance](../external/xtbloom-d3/README.md), the
[baseline migration decision](../.agents/notes/implemented/architecture/2026-09-19-d3-xtbloom-baseline.md),
and the
[production runtime decision](../.agents/notes/implemented/architecture/2026-09-19-d3-production-runtime.md).

## Ragged workspace and output isolation

The 4096-atom cap applies to each system, not the fleet total. Aggregate CUDA
workspace uses checked total-atom byte extents; skipped, failed and energy-only
gradient slots are zeroed before publication with successful peers.
See the [workspace decision](../.agents/notes/implemented/architecture/2026-09-19-d3-ragged-workspace.md).

## NVIDIA ALCHEMI performance reference

`tools/benchmark_d3_alchemi.py` compares the production D3 owner with NVIDIA
ALCHEMI Toolkit-Ops on exactly the same generated nonperiodic geometries, D3(BJ)
damping parameters, and hard pair/CN cutoff. It is an optional benchmark tool;
ALCHEMI, PyTorch, and its parameter cache are not VibeQC runtime dependencies.

The comparison deliberately reports three ALCHEMI timings separately:

- neighbor-list construction;
- D3 with the neighbor list already built, matching NVIDIA's canonical D3
  benchmark convention;
- neighbor-list plus D3 pipeline latency.

VibeQC currently performs pair traversal inside its D3 owner and therefore has no
separate neighbor-list stage to subtract. Its warm synchronous `execute()` latency
includes coordinate upload, the D3 kernel, and requested result download. The
benchmark records candidate-pair counts, ALCHEMI neighbor-edge counts, throughput,
resource diagnostics, package/CUDA metadata, raw timing samples, and the numerical
delta after converting ALCHEMI forces back to `dE/dR`.

A precision caveat is mandatory when interpreting performance: VibeQC production
D3 is FP64, while ALCHEMI Toolkit-Ops 0.4.x uses FP32 reference tables and FP32
energy/force/CN outputs even when positions are FP64. The benchmark records this
explicitly and does not declare an equal-precision winner.

Example:

```sh
PYTHONPATH=python:. VIBEQC_LIBRARY=/path/to/cuda/libvibeqc.so \
  python tools/benchmark_d3_alchemi.py \
  --device cuda --method 'PBE-D3(BJ)' --cutoff-angstrom 15 \
  --workload 32x1 --workload 128x1 --workload 32x64 \
  --alchemi-params ~/.cache/nvalchemiops/dftd3_parameters.pt \
  --output benchmark-results/d3-alchemi.json
```

For retained performance evidence, pin the exact `nvalchemi-toolkit-ops` wheel,
PyTorch/CUDA versions, GPU, VibeQC commit/library, cutoff, workload, and timing
samples. Do not compare published H100 numbers directly with a local RTX 5090 run;
run both implementations on the same allocated device.

## Parameter catalog availability

The generated parameter catalog retains all 157 pinned upstream D3(BJ) records,
projected explicitly to the implemented two-body `s9=0` model. Parameter
availability is separate from executable capability: `B97M-D3(BJ)` has negative
`a1`, and `SSB-D3(BJ)` has negative `s8`. The current `D3Spec` sign constraints
still reject these two records. Do not clip or take absolute values to bypass
that boundary; extending signed damping requires separate numerical qualification.

Quoted upstream TOML keys are decoded as names: for example,
`SKALA-1.0-D3(BJ)` and `SKALA-1.1-D3(BJ)` contain no literal quote characters.
The development-time synchronizer checks pinned source hashes; production uses
generated constants without opening an upstream table or importing its package.
The same pipeline preserves 118 D4 parameter records, without advertising new
public DFT+D4 endpoints solely because those records are present.

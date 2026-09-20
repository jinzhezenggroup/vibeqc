# D3(BJ) production correction runtime

VibeQC represents additive geometry-only dispersion with
`DispersionCorrectionPrimitive` and an immutable `D3Spec`. The native library
now exposes a standalone production D3(BJ) correction owner for CPU and CUDA,
including ragged batches and fixed-topology changed-geometry replay. This is an
additive correction endpoint: it does not yet make `Calculator` automatically
sum electronic DFT and D3 energies or forces, so issue #492 remains open.

## Supported model and MethodIR composition

The production model is nonperiodic, real FP64, two-body D3(BJ), with `s9=0`.
`D3Spec` records explicit `s6/s8/a1/a2`, source-data SHA-256 identities,
coordination and pair cutoffs, and the pair-switch width. ATM, zero damping,
unsupported versions, invalid coefficients and mismatched table identities are
rejected rather than silently approximated.

The audited method catalog includes `PBE-D3(BJ)` and `PBE0-D3(BJ)`. Their
`MethodIR` graphs contain the normal semilocal/exact-exchange primitives followed
by one `DispersionCorrectionPrimitive`. The correction identity is independent
of a descriptive manifest name and is checked against the compiled table hashes
when a production owner is prepared.

Energy is in Hartree; coordinates are in bohr. The correction returns
**gradient = dE/dR**, including explicit pair-distance and coordination-number
response. Forces therefore have the opposite sign. The GFN1 halogen correction,
Hamiltonian, SCC state and D4 terms are not part of this endpoint.

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

No xTBloom or simple-dftd3 runtime dependency is added. Method-level D3/D4/gCP
coefficients have one editable source in
`python/vibeqc_compiler/method/method_parameters.json`; codegen emits the
Python MethodIR constants and native/CUDA `constexpr` accessors, so calculation
paths do not parse configuration files at runtime. Build-time generation also
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

## Remaining boundary

The public correction owner is deliberately separate from the electronic DFT
SCF/Fock equation. Automatic `Calculator` composition of DFT + D3, the complete
combined electronic-plus-dispersion force endpoint, pair-parallel CUDA lowering,
ATM, and zero-damping variants remain separate work. Native DFT paths must not
accept a correction node and then omit it silently.

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

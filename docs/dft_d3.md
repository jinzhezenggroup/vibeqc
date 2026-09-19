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

No xTBloom or simple-dftd3 runtime dependency is added. Build-time generation
verifies the pinned xTBloom-derived table and covalent-radius SHA-256 values and
emits only the compact production data needed by the native evaluator. The
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

# Portable HF checkpoint/restart

`PreparedBatch.save_checkpoint` persists converged RHF/UHF density seeds and
complete scientific identity. `load_checkpoint` validates a file against a
separately requested target and imports compatible seeds. A one-item batch is
the single-system interface. Direct and density-fitted HF use the same contract;
CPU and CUDA can exchange checkpoints without exchanging runtime objects.

```python
from vibeqc import Calculator

systems = [[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]]
with Calculator(method="rhf", device="cuda").prepare_batch(systems) as batch:
    batch.execute(strict=True)
    batch.save_checkpoint("hf.vqcp")

# This may run in another process, on another machine/backend.
with Calculator(method="rhf", device="cpu").prepare_batch(systems) as batch:
    report = batch.load_checkpoint("hf.vqcp")
    result = batch.execute(strict=True)
    print(result.items[0].restart_origin)  # persistent_restart
```

Loading never changes the requested method, basis, charge/spin, fitting model,
precision policy or convergence target. It does not execute SCF or claim target
convergence. `execute` builds the target Fock and checks the normal energy and
density convergence criteria, with the established analytic-force path.
A saved convergence flag and energy cannot bypass this work.

## State inventory and identity

The required manifest records ordered nuclei, charge, spin populations, geometry
and actual basis hashes, AO conventions, all-electron Hamiltonian, direct or RI
provider identity, auxiliary basis and metric threshold, solver controls,
precision policy, the resource planner's complete runtime-control inventory,
source backend and basis provenance. Mathematical identity
reuses `ResolvedModel` from the accuracy and SCF-state layer. Numerical controls
are separate from that identity; backend and runtime scheduling are provenance.

The reusable optional state is one spin-summed RHF density or ordered alpha and
beta UHF densities, together with the geometry where that density was retained.
Source energy, iterations, energy change and density RMS are diagnostics only.
The source geometry and controls follow the seed even when warm updates are
frozen, a changed-geometry solve fails, or a restored seed is re-exported.
Each input slot also records its latest execution status, separately from the
converged source seed; failed items can retain an older valid seed.

Contexts, CUDA streams/events, cuBLAS/cuSOLVER handles/workspaces, graph
executables, allocators, raw pointers, Fock/DIIS histories, integral tensors,
geometry caches and resource plans are reconstructible and never serialized.
Native export copies already retained host densities. Native restore rebuilds
only the source AO overlap on the CPU for validation; this host work is explicit
also for a CUDA target. Normal preparation/execution creates target resources.
An existing prepared object's resident/frozen CUDA warm seed and energy cache
are invalidated before imported seeds can execute.

## Compatibility and validation

| Level | Meaning | Restore policy |
| --- | --- | --- |
| `exact_restart` | Same scientific model, ordered nuclei, geometry and numerical controls | Accepted by default; target SCF still runs |
| `warm_start_compatible` | Same scientific model and AO topology, changed geometry or numerical controls | Requires `allow_warm=True` |
| `transport_required` | Different orbital basis or AO representation | Rejected; orbital/basis transport belongs to #189/#190 |
| `incompatible` | Different nuclei/order, method, core treatment, electron/spin counts, auxiliary model/provider, or absent state | Rejected |

Basis compatibility uses actual normalized mathematical identities, not aliases
or matrix dimensions. Unknown schemas/providers are file errors. No CC, triples,
DIIS, orbital/virtual, or response/Krylov arrays are accepted by schema 1. These
need separately versioned equation/reference/operator and inner-product
identities before persistence can be enabled; shape matching is insufficient.

Native validation uses the same ensemble-density guard as SCF proposals:
finite data, Hermiticity, source-metric electron/spin trace, occupation bounds
and nonsingular overlap (`1e-10` minimum eigenvalue; `1e-7` density guard tolerance).
It does not change an invalid source density to make it pass. For a changed
target geometry, the existing fleet policy restores Hermiticity and rescales
each spin block to the target metric electron trace; an empty spin block remains
zero. This is a warm density guess, not orbital transport or an assertion of
target metric idempotency. Numerical stagnation/failure follows the existing
per-item cold retry. Runtime/CUDA failures retain the existing error policy.

All supplied compatible seeds pass native validation before any is installed.
A corrupt file cannot partially mutate a live batch. By default `strict=True`
rejects a batch containing any incompatible or no-state item. With
`strict=False`, compatible slots are restored and incompatible slots retain
their existing state. File corruption remains an error for the whole operation.
Input order is explicit and item counts must match; no implicit permutation is
inferred, even for equal-sized molecules.

`checkpoint_diagnostics` records bytes, schema, write/read time, manifest hash
on write, per-item compatibility/reasons and restored/rejected fields on load.
It initially labels target verification `pending_execute`; after execution it
contains target statuses, convergence diagnostics and origins. Result origins
are `cold`, `in_process_warm`, `persistent_restart`, or `cold_fallback`.
`clear_warm_starts` restores cold execution. An imported frozen seed continues
to report `persistent_restart` until successfully replaced or cleared.

## File format and failure behavior

The byte layout is deliberately simple and uncompressed:

1. Eight magic bytes `VQHFCP01`.
2. Unsigned 64-bit **little-endian** JSON-manifest length.
3. Thirty-two SHA-256 bytes for that manifest.
4. The exact UTF-8 JSON manifest (`schema="vibeqc.hf_checkpoint"`, version 1).
5. Contiguous blobs at manifest-relative payload offsets. Every element is
   IEEE-754 binary64, **little-endian**, C/row-major order. Density shape is
   `(spin_block, nao, nao)`; geometry shape is `(natom, 3)` in Bohr.

Each blob declares its dimensions, byte count, offset and SHA-256. There are no
external filenames, archive members, compression, pickles, executable objects
or native struct dumps. Names are exactly `density_INDEX`/`coordinates_INDEX`.
The reader rejects unknown/missing required fields, duplicate keys/blobs,
unknown dtype/endian/schema, impossible populations, dimension/byte mismatches,
truncation, checksum failures, nonfinite arrays and trailing data. It checks
size products before allocation; the C buffer API additionally checks dimensions
against the trusted prepared topology before any pointer arithmetic/copy.
Checksums detect corruption; they are not an authenticity/signature mechanism.

Writes use a sibling temporary file, flush and `fsync`, independently verify
metadata/blobs, then publish with `os.replace`. POSIX platforms also `fsync` the
parent directory. A failure before publication preserves the previous file;
a process killed during writing may leave an unpublished temporary artifact.
A failure after publication leaves the complete new file. Atomic rename/durability
are subject to the filesystem's own guarantees. Concurrent mutation/execution
of one prepared object remains unsupported as in the existing fleet contract.

`inspect_checkpoint(path)` validates the format/checksums without loading a
native library or retaining complete numeric blobs. Both APIs default to a
256 MiB serialized-file limit (`max_bytes`), with a 4 MiB manifest and 10,000
item limit. Loading applies a conservative numeric-buffer staging/validation
bound above the **current** `ResourcePlan`'s host peak before allocating arrays;
insufficient headroom fails explicitly. Python object/runtime overhead remains
outside the resource planner's numeric-buffer scope. Loading does not replace
the current resource plan, device budget, or execution schedule. Correlated and
response streaming formats are future extensions, not silently supported fields.

The public C buffer functions `vibeqc_batch_get_hf_warm_state` and
`vibeqc_batch_restore_hf_warm_states` expose scientific state without a file
format. C callers must verify complete model/basis/provider identity themselves;
the Python checkpoint layer performs that verification. The C descriptor is a
live ABI structure and must never be written as an on-disk checkpoint.

## Reproducible validation

```bash
cmake -S . -B build -G Ninja -DVIBEQC_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/python/test_checkpoint.py -q
```

For real CUDA coverage, build CUDA normally and run through the scheduler:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc 'PYTHONPATH=python:. OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 VIBEQC_LIBRARY=$PWD/build-cuda/libvibeqc.so \
  VIBEQC_CHECKPOINT_DEVICE=cuda .venv/bin/python -m pytest \
  tests/python/test_checkpoint.py -q'
```

The same tests cover new-process direct/DF RHF and UHF restart, independent
PySCF direct energies/forces, both CPU/CUDA transfer directions, geometry changes,
custom Cartesian/spherical d/f bases, frozen provenance, invalid source densities,
provider/model/spin/schema rejection, overflow/corruption, interrupted publication,
per-slot isolation and resource headroom. Tests deliberately falsify source
convergence diagnostics while preserving a physically valid nonstationary seed
to verify that only target execution can establish convergence.

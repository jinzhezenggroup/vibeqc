# #502 GeometryIR / PairIR

Agent: ChatGPT
Model: GPT-5.6 Sol

## Decision

Implement a thin fixed-topology `GeometryIR` / `PairIR` layer that lowers entirely through the existing TensorIR. Pair enumeration, ownership, element metadata, parameter identity, and cutoff/switch contracts are compiler data; runtime neighbor-list rebuild policy remains outside the IR.

The canonical pair list is encoded in TensorIR gather positions, so topology changes alter the mathematical equation. The outer `PairProgram.identity` additionally includes geometry metadata, cutoff/switch metadata, parameter identity, pair kind, and lowering version. Prepared pair consumers must validate this identity, so a topology/cutoff/parameter change cannot silently reuse stale pair execution state even when a lower-level TensorIR artifact is mathematically reusable.

Coordinate JVP/VJP reuse TensorIR `linearize` / `transpose_program`. In reverse mode, existing gather adjoints lower to incidence-matrix scatter-add; no method-specific force kernel or CUDA generator was added.

## Rejected alternatives

- A separate pairwise CUDA generator: duplicates TensorIR lowering and AD.
- A dynamic neighbor-list primitive in TensorIR: mixes runtime rebuild policy into mathematical IR.
- GFN-specific charge/SCC state in PairIR: would prevent D3/D4/gCP reuse.
- Encoding cutoff/parameter identity as zero-valued arithmetic: would contaminate equation semantics only to affect caches.

## Validation

- `python tools/check_compiler_structure.py`: 203 modules, 0 dependency errors.
- TensorIR/AD/transcendental/GeometryIR focused CPU suite: 224 passed.
- RTX 5090, CUDA 12.9, `sm_120`: PairIR energy + generated Cartesian VJP + CUDA finite difference: 1 passed.
- CPU qualification also covers deterministic topology/ownership, translation/permutation invariance, coordinate JVP/VJP finite differences, pair-to-atom/system reductions, and stale-identity rejection.

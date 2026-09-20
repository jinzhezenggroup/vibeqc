# Decision: stage generated ragged D3(BJ) CUDA execution before native retirement

Status: implemented candidate; production cutover not qualified
Date: 2026-09-20

## Problem

#627 moved the two-body D3(BJ) scientific equation and Cartesian derivative to
GeometryIR/PairIR/TensorIR, but the generated representation was fixed to one
molecular pair state. The public owner introduced by #549 supports heterogeneous
ragged batches and changed-geometry replay, so deleting `d3_bj.hpp` immediately
would have reduced public execution semantics even though the mathematical graph
was already compiler-owned.

The missing boundary is execution, not another D3 formula.

## Decision

Add a generated ragged retirement candidate with these ownership rules:

- one flattened `GeometryIR` contains every atom in a heterogeneous molecular
  batch;
- strictly increasing system offsets partition that atom space;
- pair construction enumerates only within each system range, rejecting any
  cross-system pair;
- the existing D3 CN/C6/BJ/switch TensorIR graph is reused unchanged;
- pair energy is reduced onto an explicit system axis with `scatter_add`;
- one generated VJP of the vector system energy produces the complete flattened
  Cartesian gradient;
- `PreparedD3CudaBatch` sends the combined energy/gradient graph through the
  shared TensorIR CUDA planner, compiler cache and prepared executor. It adds no
  D3-specific CUDA scientific kernel.

CN membership and pair-cutoff/switch regions remain explicit compiler state.
A topology-preserving replay reuses the prepared executable. A changed geometry
that crosses one of those state boundaries compiles/prepares a replacement
generated program first and only then releases the old prepared owner. Preparation
failure therefore leaves the previous owner intact rather than mutating a live
scientific plan.

## Why one ragged graph

A per-molecule collection of TensorIR executors would preserve correctness but
would turn batch execution into a host launch loop and weaken the existing public
ragged contract. TensorIR already supports indexed gather/scatter and ragged
segment primitives, and GFN2 already demonstrates the same flattened
heterogeneous-batch pattern. D3 uses the common machinery instead of adding a
dispersion-specific scheduling language.

## Production boundary

This candidate does **not** replace `D3CorrectionBatch` yet.

Before production cutover, the generated route must independently demonstrate:

1. real-device energy and complete CN-response gradient parity against the pinned
   simple-dftd3 fixtures and the native oracle;
2. exact and one-byte-below resource-budget behavior;
3. unchanged-topology and changed-topology replay;
4. per-system failure isolation equivalent to the public native result/status ABI;
5. a true energy-only plan that does not execute/download the generated gradient;
6. endpoint performance and resource behavior against the native D3 owner and the
   retained matched ALCHEMI harness.

The current TensorIR prepared call has one batch-level error boundary. That is not
silently presented as equivalent to native per-item status isolation. Likewise,
the initial candidate always prepares the combined energy/gradient graph; returning
no gradient to a caller is not an energy-only performance implementation.

## Scientific invariants

No D3 scientific formula changes in this slice. The compiler remains restricted to
the qualified nonperiodic FP64 two-body D3(BJ) model with `s9=0`, pinned table and
radii identities, elements 1--86, and the existing reference-weight underflow
failure boundary. Energy is Hartree and generated coordinates derivatives are
`dE/dR`, not force.

A changed topology is a new compiler identity. Performance tuning may change the
CUDA schedule but must not hide cutoff membership, alter the scientific graph or
reintroduce a handwritten D3 kernel.

## Evidence

- `tests/python/test_d3_generated_ragged.py` checks heterogeneous flattening,
  independent per-system simple-dftd3 energies/gradients, absence of cross-system
  pairs, explicit switch/topology rebuild state, and one-program CUDA lowering.
- The test is part of the explicit `compiler-heavy` CI shard.
- Real-device execution is opt-in through
  `VIBEQC_D3_GENERATED_CUDA_TEST=1`; absence of that run is not reported as a
  GPU numerical or performance qualification.

## References

- #492
- #549
- #605
- #627
- `docs/dft_d3.md`
- `docs/geometry_pair_ir.md`
- `python/vibeqc_compiler/geometry/d3.py`
- `python/vibeqc_compiler/geometry/d3_cuda.py`

Agent: ChatGPT
Model: GPT-5.6 Sol

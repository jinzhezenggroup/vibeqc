# Decision: bounded production D3(BJ) correction owner

Status: implemented
Date: 2026-09-19

## Problem

The xTBloom-derived D3(BJ) migration established shared CPU/CUDA arithmetic and
independent numerical qualification, but it remained a diagnostic baseline. A
production correction needs MethodIR identity, compact generated data, explicit
resource ownership, ragged batching, changed-geometry replay and a public ABI
without introducing an external D3 runtime dependency.

## Decision

D3 remains an additive geometry-only primitive, separate from SCF/Fock equations.
`PBE-D3(BJ)` and `PBE0-D3(BJ)` are audited MethodIR manifests. Preparation lowers
the pinned data into compact generated native tables and verifies their hashes
against the requested `D3Spec`.

A persistent `D3Plan` owns fixed topology and resource bounds. CPU execution uses
direct pair loops plus O(N) scratch. CUDA owns ragged offsets, atomic numbers,
compact tables, coordinates, masks, outputs and workspace on one nonblocking
stream. Replays may replace coordinates without rebuilding topology. The first
CUDA schedule assigns one serial worker to each molecule and independent blocks
to independent systems.

## Rejected alternatives

- Linking xTBloom or simple-dftd3 at runtime would add a second execution owner
  and weaken reproducibility of the public binary.
- Materializing O(N^2) pair/reference records would preserve the diagnostic
  implementation shape but violate the intended bounded production memory model.
- Hiding D3 inside semilocal XC or exact exchange would make correction omission
  and provenance difficult to detect.
- Treating the initial one-worker CUDA kernel as a performance endpoint would
  confuse ownership qualification with pair-parallel optimization.

## Invariants

- The supported production model is two-body D3(BJ), `s9=0`; unsupported ATM and
  damping variants fail explicitly.
- Coordinates are bohr, energy is Hartree, and the derivative is `dE/dR`.
- Complete coordination-number response is part of the analytic derivative.
- MethodIR/table identities must match the generated native data.
- The plan owns copied topology data; callers need not preserve input buffers.
- Memory admission happens before execution through an explicit byte bound.
- Native DFT must never execute only the electronic part of a graph containing an
  unhandled dispersion node.

## Evidence

The production CPU endpoint reproduces all six independent PBE/PBE0
simple-dftd3 1.4.0 fixtures at effectively roundoff-level energy/gradient error.
A CUDA 12.9.86 `sm_120` build on an RTX 5090 completed the final shared library.
The public Python CUDA endpoint reproduced all six PBE/PBE0 independent fixtures
with worst energy error 2.17e-19 Hartree and worst gradient error 1.33e-18
Hartree/bohr; the production runtime suite passed 9/9. Ragged replay, changed
geometry, energy-only execution, lifetime rejection and resource-budget rejection
are covered by production tests. Compiler structure checks keep generation
independent of the public runtime.

## Consequences

This establishes production ownership and a bounded CUDA baseline without yet
promoting a combined DFT+D3 `Calculator` endpoint. The serial-per-system CUDA
schedule is intentionally simple; pair-parallel lowering can replace it without
changing MethodIR or public correction semantics.

## Revisit when

Revisit scheduling when realistic molecule-size benchmarks justify pair-parallel
or tiled work distribution. Extend the model only with separately versioned and
qualified ATM or alternate damping semantics. Integrate with `Calculator` when
the electronic energy/force path can compose the correction atomically.

## References

- Issue #492
- PR #497 (qualification baseline)
- `docs/dft_d3.md`
- `tests/data/d3_bj_reference.json`

Agent: ChatGPT
Model: GPT-5.6 Sol

# GeometryIR / PairIR

Issue #502 adds a thin backend-neutral geometry layer above the shared TensorIR.
It does not own neighbor-list rebuild policy or a method-specific CUDA generator.

## Contract

`GeometryIR` stores immutable element metadata, a parameter-table identity, and
one differentiable Cartesian coordinate input. `PairTopology` stores the exact
canonical undirected pair list, deterministic lower-atom ownership, and optional
cutoff/switch metadata. Pairs are required to be unique, lexicographically
ordered, and written as `i < j`.

Changing the pair list changes the lowered TensorIR equation because gather
positions are equation data. Changing cutoff or parameter metadata changes the
`PairProgram.identity` even when the emitted mathematical equation is reusable.
Prepared pair consumers must retain and validate that identity; a mismatch is a
stale execution state.

Runtime neighbor-list construction/rebuild remains outside the IR. A runtime
that rebuilds a list must construct a new `PairTopology` contract.

## Lowering

The geometry layer lowers only through existing TensorIR primitives:

```text
coordinates[atom,cartesian]
  -> gather(left/right atom indices)
  -> reshape[pair,cartesian]
  -> displacement
  -> squared distance
  -> distance
  -> pair scalar equation
  -> pair-to-system reduction
```

Pair-to-atom accumulation is expressed as an incidence-matrix einsum. Reverse
coordinate differentiation uses TensorIR's generated VJP; the gather adjoint is
therefore the existing scatter-add incidence lowering rather than a hand-derived
force kernel.

The qualification potential is

```text
E = sum_p c_p r_p^n
```

and is deliberately method-neutral. It exercises pair geometry, exact
parameters, transcendental TensorIR, system reduction, generated coordinate
adjoints, CPU reference execution, and CUDA lowering. D4/gCP/xTB consumers can
reuse the geometry contract without importing GFN charge or SCC semantics.

## D3(BJ) consumer

The two-body D3(BJ) compiler consumer uses this fixed-topology contract for the union
of CN-response and pair-energy pairs. It records CN membership and hard/smooth
pair-cutoff regions in the execution-state identity, reuses the pinned D3 tables, and
derives Cartesian `dE/dR` from the same TensorIR energy graph with the generated VJP.
Crossing a CN/pair/switch boundary requires rebuilding the D3 pair state.

This establishes compiler ownership of the D3 scientific equation; the existing
ragged native D3 runtime remains the production executor/oracle until shared PairIR
supports the same dynamic/ragged replay contract. See the
[D3 GeometryIR/PairIR decision](../.agents/notes/implemented/numerics/2026-09-20-d3-geometry-pair-ir.md)
for exact cutoff conventions, provenance, normalization fallback boundary, evidence,
and retirement criteria.

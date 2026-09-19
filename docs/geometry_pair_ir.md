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

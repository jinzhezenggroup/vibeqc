# Serial ProgramIR and fixed-density XC tile lifetimes

`vibeqc_compiler.common.program` describes a **finite, synchronous sequence of
opaque provider calls**. It derives dependency and last-use information without
replacing IntegralIR, TensorIR, MethodIR, XC mathematics or native runtime owners.
This is a deliberately narrow compiler prototype, not a whole-program optimizer.

## Contracts

`ProgramBuffer` describes a disjoint ownership group and its numeric capacity.
It may also bind the shared `DenseLayout` plus item size when a producer/consumer
boundary has a qualified physical layout; the byte capacity must match exactly.
`PlanCall` binds a named provider identity to explicit read/write buffers.
`ProgramIR` validates the selected order and derives inclusive resource intervals.
A value must be an input or have exactly one earlier producer. Duplicate owners,
missing/forward/cyclic dependencies and in-place writes fail before analysis.
The chosen call order is part of the identity; no opaque call is deleted or moved.

Inputs are borrowed and retained for the complete region. Outputs survive through
publication. Other values become releasable **after** their last consuming call
finishes. Read and newly written buffers overlap during a call. There is no
input donation, alias analysis, view ownership inference or asynchronous release.
Providers must not retain undeclared aliases or launch unfinished work. An async
or aliasing provider needs a different explicit contract, not this serial model.

Serialization contains data only. `ProgramIR.from_payload()` rejects unknown or
missing fields and reconstructs validation rather than trusting serialized
lifetime claims. Identity includes provider bindings, buffer capacities/spaces,
qualified dense layouts/item sizes, requested outputs and selected order. It is
not an executable cache key or a
certificate that a native library or numerical state is current.

`resource_request()` feeds the existing `common.resources` planner. Its accounting
is **boundary-only**: opaque-provider scratch/copies, library workspaces, Python
objects, allocator overhead and caller-retained result history are excluded.
The diagnostic `retain_temporaries=True` alternative is a retain-all model, not a
measurement of an existing runtime or an end-to-end performance baseline.

## First native consumer

For dense synchronous CPU potential requests, `PreparedXCContractions.tile_program`
is an immutable description of the existing path:

```text
borrowed basis + complete quadrature owner
                  |
                  v
          NativeAO.evaluate
                  | jets
                  v
    NativeContractionProgram.evaluate  <-- spin density + quadrature
                  |
                  v
       E_xc / V_xc / electron contribution
```

The XC provider already owns density features, scalar XC and potential assembly.
The graph deliberately does not invent separate native kernels for those stages.
The complete quadrature owner is charged once: point/weight tile views are not
misrepresented as independent allocations. Final partial tiles use the same
maximum capacity contract.

For this graph, `release_after("collocation")` is empty: Vxc still needs the AO
jets. `release_after("xc")` contains `jets`. The existing CPU execution template
recognizes that release contract and removes the consumer reference after use.
The suspended collocation generator drops its reference before allocating the
next tile. The caller also releases the consumed contribution after accumulation.
No second allocator, generic interpreter or new scientific arithmetic is added.

`tile_program` is `None` for spatial/CUDA and non-potential consumers. Those routes
retain their existing independent qualification boundaries. The complete native
resource budget remains unchanged and conservative. **Do not add the boundary
request to `prepared.resource_plan`: it would double-count existing buffers.**

Execution statistics expose `tile_program_identity` and `tile_boundary_releases`
only for the qualified dense CPU potential template. Graph construction and release
analysis happen during preparation, not per tile. Native state/provider checks,
numerical domains, quadrature, precision, reduction order and failure propagation
remain with the established owners.

## Second consumer: DFT feature layout into native XC

The initial lifetime slice kept the complete XC contraction opaque. The second
qualified slice is narrower scientifically but more explicit operationally: for
dense synchronous **polarized** CPU LDA/PBE potential tiles, ProgramIR records

```text
NativeAO.evaluate
      | [jet, point, AO]
      v
dft.density_feature_block
      | [feature, point] + optional [spin, point, xyz]
      v
xc.NativeContractionProgram.scalar_values_packed
      | generated scalar rows
      v
xc.NativeContractionProgram.potential_from_rows
```

The DFT feature producer owns one C-contiguous FP64 feature-major buffer. The
generated XC scalar consumer validates that exact owner without making the former
validation copy, active-point gather, variable `np.stack`, or output scatter.
Potential assembly still uses the existing coefficient and AO contraction code;
no XC formula, quadrature, spin convention or reduction order is duplicated.

The shared `DenseLayout` is the same physical-layout descriptor used by TensorIR,
but ProgramIR imports only the backend-neutral contract. It does not import the
TensorIR planner, choose arbitrary affine layouts, alias buffers, donate inputs,
or infer asynchronous lifetimes. Unpolarized, spatial, CUDA, response and geometry
routes retain their previous paths until independently qualified.

Execution statistics additionally expose `tile_layouts` for these real boundary
owners. Tests require the complete prepared endpoint to run with the legacy
`scalar_values` repacking entry point disabled and verify that the native scalar
input shares storage with the DFT-owned feature buffer.

## Logical SPMD lowering contract

`vibeqc_compiler.common.spmd` adds a backend-neutral lowering plan around the
immutable ProgramIR. The scientific ProgramIR remains schema v2 and does not
contain physical GPU ordinals, product names, links or topology assumptions.
A `DeviceMesh` names only logical axes. `BufferPlacement` describes whether a
boundary buffer is replicated or deterministically sharded along one dense tensor
axis, and `CollectiveSpec` makes all-reduce, reduce-scatter and all-gather
synchronization explicit. Scatter/gather tensor axes are explicit, and their
result placement is validated against the buffer contract. All-to-all is
intentionally absent until a real consumer requires it.

The same ProgramIR identity can therefore be lowered to a one-device mesh or a
larger mesh. Size-one collectives canonicalize away, so the one-device lowering is
the fallback rather than a separate scientific equation. Mesh shape, placement,
collective order and logical shard coordinates participate in the SPMD plan and
provenance identities. Replaying serialized SPMD metadata revalidates the bound
ProgramIR identity and all placement/collective invariants.

The v1 accounting contract charges one communication scratch allocation per
logical rank and records collective source-plus-result traffic as the resource
candidate's relative-cost work count. This is deliberately not a hardware
latency/bandwidth prediction:
collective-library internals and physical interconnect properties remain explicit
scope exclusions until a target/backend supplies measured profitability evidence.
Backends must declare support for every required collective before execution.

A result publication boundary must call `require_complete_shards()`; missing,
duplicate or unexpected logical ranks fail instead of publishing a partial
scientific result. `reference_collective()` supplies pure CPU semantics for the
three v1 collectives so backend implementations can be qualified independently.

This contract does **not** yet make any existing XC/SCF path multi-GPU. The first
real bounded consumer, runtime collective binding and measured 1/2/4+ GPU endpoint
scaling remain follow-up work under #834. That qualification must reuse this
logical contract rather than adding method- or GPU-name-specific scientific IR.

Rationale:
[ProgramIR SPMD contract decision](../.agents/notes/implemented/architecture/2026-09-21-programir-spmd-contract.md).

## Validation and reproduction

```bash
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_program_ir.py tests/python/test_program_spmd.py \
  tests/python/test_program_ir_xc.py \
  tests/python/test_xc_contractions_native.py
PYTHONPATH=python:. python tools/check_compiler_structure.py
```

Native tests require an available CPU library and C++ compiler. They use independent
LDA/PBE energy/potential fixtures, repeated and changed densities, partial tiles,
and weak references checked at the **next actual native AO allocation**.

The benchmark below separates instrumentation from ordinary complete fixed-density
E/V timings. Use the same explicit native library and thread settings for two
checkouts, pointing `PYTHONPATH` at the checkout being tested and running the same
benchmark script. Source/library identities and raw timing samples are reported.

```bash
export VIBEQC_LIBRARY=/path/to/qualified/cpu/libvibeqc.so
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python:. \
  python benchmarks/programir_xc_lifetimes.py --case f_spherical --repeats 11
```

The reported live boundary payload is **not** a process RSS or device peak.
Freeing a boundary earlier does not prove a smaller complete-endpoint peak, since
provider-internal scratch may dominate. Fixed-density XC is not a full SCF,
force, geometry optimization or GPU benchmark. Extend the prototype only when a
separate consumer and measured benefit justify more machinery.

Rationale and initial evidence:
[serial XC lifetime decision](../.agents/notes/implemented/performance/2026-09-19-programir-xc-lifetimes.md).

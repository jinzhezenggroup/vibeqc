# Serial ProgramIR and fixed-density XC tile lifetimes

`vibeqc_compiler.common.program` describes a **finite, synchronous sequence of
opaque provider calls**. It derives dependency and last-use information without
replacing IntegralIR, TensorIR, MethodIR, XC mathematics or native runtime owners.
This is a deliberately narrow compiler prototype, not a whole-program optimizer.

## Contracts

`ProgramBuffer` describes a disjoint ownership group and its numeric capacity.
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
requested outputs and selected order. It is not an executable cache key or a
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

## Validation and reproduction

```bash
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_program_ir.py tests/python/test_program_ir_xc.py \
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

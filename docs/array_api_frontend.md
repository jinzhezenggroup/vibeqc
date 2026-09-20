# Array API frontend for TensorIR

Issue #633 introduces a bounded, compiler-internal symbolic array frontend.  Its
purpose is developer ergonomics and interoperability: normal array expressions
are captured once and lowered to the existing TensorIR.  TensorIR remains the
scientific IR and keeps the stronger quantum-chemistry type system.

The frontend does **not** currently claim Array API conformance. It provides an
Array-API-shaped internal preview and fails closed outside the declared subset.
In particular, `VibeArray` deliberately does not implement
`__array_namespace__` yet: the Array API standard uses that method as a
compliance discovery signal and requires the returned namespace to provide the
standard's top-level API. Advertising the protocol for this bounded subset
would therefore misidentify a preview object as conforming.

## Layering

```text
symbolic array expression
        |
        v
VibeArray / namespace
        |
        v
existing TensorIR nodes
        |
   +----+----+
   |         |
   AD      CPU/CUDA lowering
```

MethodIR, IntegralIR, ProgramIR, stationary/implicit solves and integral
providers retain their existing ownership.  The frontend does not turn ERI,
J/K, XC, SCF iteration or eigensolvers into generic array primitives.

## Preserved scientific semantics

A `VibeArray` wraps an ordinary TensorIR node.  Therefore capture preserves:

- AO / occupied / virtual / auxiliary / batch / spin index-space identity;
- selected index ranges and ordered gathers;
- orbital representation and declared symmetry;
- exact rational compile-time coefficients;
- input/parameter role and differentiability;
- TensorIR logical identity, serialization, optimization and JVP/VJP behavior.

Equal numerical shapes do not make different scientific domains compatible.
Elementwise operations currently require identical TensorIR domains.

## Initial capability subset

| Surface | Initial contract |
| --- | --- |
| `+ - * /` | Equal-domain arrays; `* /` also accept exact scalar scaling |
| unary `-` | Exact coefficient lowering |
| `pow/exp/log/sqrt` | Existing TensorIR real-valued contracts |
| `sum` | Explicit reduction, `keepdims=False`, no dtype conversion |
| `permute_dims` | Full TensorIR axis permutation |
| `reshape` | Requires explicit target TensorIR `Index` metadata |
| `broadcast_to` | Requires explicit target indices and source-axis map |
| `slice` | Static unit-step half-open ranges; populations are retained |
| `take` | Static integer gather along one axis; source domain is retained |
| `matmul` | Rank-2 only |
| `einsum` | VibeQC extension lowered to existing TensorIR einsum |
| implicit broadcasting | Not yet supported |
| dtype promotion/casts | Not yet supported |
| implicit reshape/domain inference | Not supported; target indices must be explicit |
| dynamic shapes/control flow | Not supported |

Exact scalar spelling accepts `int`, `Fraction`, or a rational string.
Python floating-point spellings such as `0.5` are deliberately rejected so a
frontend convenience cannot weaken TensorIR scientific identity.

Shape-changing operations follow the same rule: integer extents alone never
define AO/occupied/virtual/auxiliary meaning. `reshape` therefore requires
explicit target `Index` objects, while `broadcast_to` additionally requires
an explicit source-to-target axis map. TensorIR then validates preserved
domains on every mapped axis; equal extents cannot authorize relabeling.

Compiler code imports `vibeqc_compiler.array_api.namespace` explicitly.
`__array_namespace__` and a versioned Array API declaration will only be added
after a dedicated conformance matrix proves that the advertised namespace meets
the corresponding standard version.

## Example

```python
from vibeqc_compiler.array_api import namespace as xp
from vibeqc_compiler.array_api import trace

program = trace(
    lambda coefficients, occupations: {
        "density": xp.einsum(
            "bspi,bsi,bsqi->bspq",
            coefficients,
            occupations,
            coefficients,
        )
    },
    {
        "coefficients": coefficients_spec,
        "occupations": occupations_spec,
    },
)
```

The resulting object is an ordinary `vibeqc_compiler.tensor.Program`; no
frontend-only node survives lowering and no Python callback is needed for
prepared native execution.

## Native SCF adoption

The CPU SCF density and energy-weighted-density production helpers are generated
at build time from the validated `array_api.scf -> TensorIR` topology. The
dynamic native lowering preserves the existing AO/orbital storage contract and
the legacy FP64 product/accumulation order, including the exact density witness
used by occupied density-fitting exchange. Python tracing is therefore absent
from the SCF iteration hot path.

This cutover covers CPU density construction and its compact occupied-factor
consumer. Resident CUDA SCF density kernels retain their existing device
ownership for now; moving those kernels requires separate stream/layout and
performance qualification. This is still an internal preview and does not add
an Array API conformance claim.

## Ownership

`vibeqc_compiler.array_api` is a separate compiler owner above
`vibeqc_compiler.tensor`. Its dependency direction is deliberately one-way:
the frontend may import TensorIR, while TensorIR cannot import the frontend.
This keeps TensorIR usable by hand-built/generated equations and avoids making
array syntax part of mathematical IR identity.

The design rationale, rejected alternatives and invariants are retained in the
[Array API frontend architecture note](../.agents/notes/implemented/architecture/2026-09-20-array-api-tensorir-frontend.md).

The native cutover is a checked specialization, not a general C++ graph emitter.
Its topology, FP64 order and source-identity contract are recorded in the
[native specialization note](../.agents/notes/implemented/architecture/2026-09-20-native-array-scf-specialization.md).

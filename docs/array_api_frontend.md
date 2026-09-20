# Array API frontend for TensorIR

Issue #633 introduces a bounded, compiler-internal symbolic array frontend.  Its
purpose is developer ergonomics and interoperability: normal array expressions
are captured once and lowered to the existing TensorIR.  TensorIR remains the
scientific IR and keeps the stronger quantum-chemistry type system.

The frontend does **not** currently claim full Array API conformance.  It fails
closed outside the declared subset.

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
| `matmul` | Rank-2 only |
| `einsum` | VibeQC extension lowered to existing TensorIR einsum |
| implicit broadcasting | Not yet supported |
| dtype promotion/casts | Not yet supported |
| reshape/new-domain creation | Not yet supported |
| dynamic shapes/control flow | Not supported |

Exact scalar spelling accepts `int`, `Fraction`, or a rational string.
Python floating-point spellings such as `0.5` are deliberately rejected so a
frontend convenience cannot weaken TensorIR scientific identity.

`VibeArray.__array_namespace__()` returns the internal namespace, but a
versioned `api_version` request currently fails explicitly.  Versioned
conformance will only be advertised after a dedicated conformance matrix exists.

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

# Programmable extension API

`vibeqc.extensions` is the public, versioned construction surface for advanced
users who need to describe supported quantum-chemistry computations without
writing backend-specific CUDA or C++.

The architectural rule is that built-in and user-defined methods share the same
scientific specs and canonical IR. Built-ins may be AOT-compiled for release;
advanced extensions may later opt into JIT, but AOT/JIT must not change the
method's scientific identity.

VibeQC uses a **single-wheel** model. The compiler may ship with the normal
package so advanced capabilities are available without a second distribution.
The important boundary is activation: ordinary built-in calculations use
AOT/native artifacts and must not invoke code emission, compiler subprocesses,
or require a local C++/CUDA compilation toolchain. JIT and autotuning are
explicit on-demand capabilities and may require such a toolchain when requested.

## XC and method composition

```python
from vibeqc.extensions import method, xc

my_xc = xc.compose(
    "my-pbe0",
    {"GGA_X_PBE": "3/4", "GGA_C_PBE": "1"},
    exact_exchange="1/4",
)
my_method = method.compose("my-pbe0", xc=my_xc)
ir = method.resolve(my_method)
```

Coefficients accept integers, rational strings, or `Fraction`; floats are
rejected so scientific identity never depends on approximate spelling.
Equivalent custom and built-in compositions canonicalize to the same
`MethodIR.identity`, while manifest identities retain descriptive names.

`method.verify` checks a resolved method against an explicit backend capability.
Representability and execution support remain separate: constructing an IR does
not imply that energy, gradients, Hessians, or a particular backend are
available.

## TensorIR

`vibeqc.extensions.tensor` exposes a curated backend-neutral TensorIR subset:
typed indices/tensors, immutable programs, serialization, optimization, AD and
the interpreter. CUDA scheduling, source emission, compiler processes and
artifact loading remain compiler APIs until a separately versioned JIT contract
is defined.

## Current boundary

This first API increment supports composition of already-audited primitives.
It intentionally does not expose arbitrary Python callbacks, an XC expression
decorator, a third-party plugin loader, or a public JIT compiler ABI. Those
features require separate versioned contracts rather than leaking current
compiler internals.

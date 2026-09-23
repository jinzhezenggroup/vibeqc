# Programmable extension API

`vibeqc.extensions` is the public, versioned construction surface for advanced
users who need to describe supported quantum-chemistry computations without
writing backend-specific CUDA or C++.

The architectural rule is that built-in and user-defined methods share the same
scientific specs and canonical IR. Built-ins may be AOT-compiled for release;
advanced extensions may explicitly opt into JIT, but AOT/JIT must not change the
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
the interpreter. Advanced users may explicitly request the first public native
compilation slice with `tensor.compile(program, target="cpu", mode="jit")`.
This path is CPU-only today and activates the host compiler only after the
request has passed TensorIR and resource admission.

Use `tensor.compile_capabilities(...)` to inspect the same lowering without
probing or invoking a local compiler. A successful report distinguishes
representation/lowering from independent scientific validation and production
promotion, returns bounded resource requirements, and exposes the exact
compiler-owned source/program `identity` that an equivalent explicit CPU JIT
request will use. Unsupported target/mode/IR requests return no compile identity.
The compiled artifact then adds toolchain-specific cache and binary provenance.

## Current boundary

The public API supports composition of already-audited XC/method primitives and
a documented TensorIR subset with explicit CPU JIT. It intentionally does not
expose arbitrary Python callbacks, an XC expression decorator, a third-party
plugin loader, CUDA JIT, or a custom-primitive device ABI. Those features require
separate versioned contracts rather than leaking current compiler internals.

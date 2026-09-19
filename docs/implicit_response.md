# Generated first-order implicit response

`vibeqc_compiler.method.ImplicitSolveSpec` describes a solved state through one
TensorIR residual `R(x, q) = 0`, in explicitly independent real FP64 coordinates.
`compile()` produces an `ImplicitVJPPlan`: ordinary TensorIR programs for the
primal residual, Jacobian action, transpose action, adjoint RHS, physical
multiplier and parameter-source weights. It adds neither another tensor algebra
nor a solver-iteration tape. The schema and generated plans support data-only
replay; replay regenerates the derivative rule and rejects substituted graphs.

## Mathematical and layout contract

For positive diagonal state/residual metrics `Wx` and `Wr`, the state cotangent
`gx` is defined by `dF = gx.T @ Wx @ dx`. The parameter cotangents are Euclidean.
The negative-adjoint convention is

```text
Wx^-1 Jx.T Wr lambda = -gx
bar_q = direct_bar_q + Rq.T Wr lambda
```

The existing Euclidean solver receives `y = sqrt(Wr) lambda` and

```text
A = invsqrt(Wx) Jx.T sqrt(Wr)
b = -sqrt(Wx) gx
A y = b
```

Every scaling, transpose, RHS sign and source contraction above is generated.
Empty metric tuples select unit weights. When an upstream derivative supplies
an **Euclidean** state cotangent `bar_x`, convert it to `gx = Wx^-1 bar_x` before
using a non-unit state metric. A weighted packed cotangent already defined in
the same metric needs no such conversion. Direct parameter contributions must
be passed separately; omitting them differentiates only the implicit dependence.

Packed callers first express the residual in their existing independent
coordinates, then supply the packing-map identity and orbit multiplicities.
A redundant dense symmetric state is rejected rather than silently flattened.
State and residual shapes may differ, but their independent dimensions must
match. Layout, gauge, provider/operator identity, metrics, equation hash,
first-order derivative rule and opaque solver contract are identity-bearing.
Solver implementation/options and the current reference/data identity are
additionally bound by the execution identity.

The rule assumes a differentiable branch with an invertible Jacobian in the
declared independent gauge. A small primal residual or a zero cotangent is not
a general invertibility/conditioning certificate. Production providers must
supply their method-specific branch and stability qualification. No higher-order
implicit rule is registered: differentiating a fixed-state action graph does
not establish a Hessian or second derivative of the solved state.

## Explicit execution adapters

`tools.vibeqc_response.implicit.BoundImplicitState` reuses the existing #179
GMRES implementation through `ResponseGMRES`, or an explicit callback with the
same solution/status/residual/workspace contract. The callback receives an
already-transposed, matrix-free Euclidean operator; it must not transpose it
again. It publishes neither a partial adjoint nor source weights when convergence,
true-residual, state or resource checks fail.

By default the transpose action is the generated TensorIR program. A caller may
instead pass an existing qualified `RHFResponseOperator` through `response_operator=`. Other
`ResponseProblem` operators are admitted only after every additional kernel/provider
publishes the same explicit logical resource contract. `ResponseTransposeBinding` then checks
the exact operator identity, response layout, gauge, reference identity and the
operator's declared logical host/device resources against the #465 plan before
the first solve. A live `current_reference` callback is mandatory for this
provider-bound path. The returned solution is independently rechecked by a fresh
application of that same bound physical transpose action before generated source
weights are evaluated. The generated residual/source graph remains authoritative;
delegating the linear action does not introduce a handwritten parameter VJP.

Inputs and published arrays are immutable snapshots. `reference_identity` is
required both at binding and at VJP execution. An optional `current_reference`
callback checks the live owner's identity before, during and after execution.
Without that callback, the object is only a tooling snapshot, **not proof of a
current native molecular state**. A changed geometry/reference requires rebinding.

Example from a source checkout (`PYTHONPATH=python:.`):

```python
import numpy as np
from vibeqc_compiler.method import ImplicitSolveSpec
from vibeqc_compiler.tensor import Program, TensorSpec, add, input_tensor, multiply
from tools.vibeqc_response.implicit import BoundImplicitState

parameter = TensorSpec((), role="parameter", differentiable=True)
x = input_tensor("x", parameter)
q = input_tensor("q", parameter)
equation = Program({"residual": add(multiply(x, x), q, coefficients=(1, -1))})
plan = ImplicitSolveSpec(equation, "x", ("q",), "positive-root").compile()
state = BoundImplicitState(
    plan, {"x": np.array(2.0), "q": np.array(4.0)},
    reference_identity="root-at-q4",
)
result = state.vjp(np.array(1.0), reference_identity="root-at-q4")
assert result.parameter_cotangents["q"] == 0.25
```

`PreparedImplicitCuda` in `tools.vibeqc_response.implicit_cuda` explicitly
compiles/prepares the same generated stages through the existing TensorIR CUDA
backend. All six providers share one resource admission, including normal
library allowances; host capacity is reserved for the retained state, generated
work and the #179 solver before native preparation. There is no silent CPU
scientific fallback. `bind()` reuses the admitted programs for a fresh state.

**The #179 Krylov controller remains Python/host-controlled.** CUDA contractions
are real native execution, but each action uses the existing host/device staging
boundary. Neither this adapter nor a successful CUDA test is a device-resident
Krylov loop, a public force endpoint, or complete #193 production integration.

## Resource scope and validation

The reference adapter admits simultaneous logical arrays plus the declared
solver/executor reservations before copying state or evaluating the primal.
Its logical-array bound excludes opaque NumPy/BLAS scratch and Python objects;
it is not a process RSS guarantee. CUDA admission includes the selected native
arenas, workspaces, provider allowances and host staging. Driver/context/module
memory and allocator rounding retain the existing resource-scope exclusions.
The adapter reports the actual tensor and solver backends separately.

Fast tests cover independent analytic/directional differences, nonsymmetric and
weighted transpose identities, existing packed orbit weights, objective-VJP
composition, stale state, false solver success, nonconvergence, resource rejection,
data-only replay and nontrivial water MP2 Z-vector parity with the unchanged
#293 oracle. The molecular parity test uses the RHS **after same-space canonical
orbital relaxation**, not the earlier raw `energy_gradient`.

```sh
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/python/test_implicit_vjp.py tests/python/test_implicit_response.py
```

The real-device tier is opt-in: run `tests/python/test_implicit_cuda.py` inside an
allocated Slurm GPU job with `VIBEQC_IMPLICIT_CUDA_TEST=1`, `VIBEQC_NVCC` pointing
to the supported compiler and optional `VIBEQC_IMPLICIT_CACHE` for local artifacts.
It disables the CPU interpreter, checks a nonsymmetric matrix-free weighted
problem, changed-state execution, numerical failure/recovery and retained-provider
allocation accounting. No new large molecular finite-difference campaign is
added to ordinary CI.

Method-level implicit-rule dispatch is available through StationaryProblem v2:
a state that declares `implicit_operator_identity` is compiled automatically
to an identity-bearing `ImplicitVJPPlan`; coupled state rows are rejected until
they are represented as one explicit independent block. Native response-provider
binding is now qualified by the response-operator path above, including strict
stale-state and simultaneous logical resource checks.

Installed public runtime integration, production-scale force evidence and the
complete #193 C2 assembly (including #466 metric response and upstream provider
pullbacks) remain downstream work. The #293 gradient oracle and public MP2
energy-only capability are unchanged. See the
[decision note](../.agents/notes/implemented/numerics/2026-09-19-implicit-response-primitive.md)
for the ownership and sign/layout rationale.

# StationaryProblem compiler boundary

`vibeqc_compiler.method.stationary` composes an explicit scalar objective,
physical residual equations, independent state coordinates and a parameter-source
inventory. It implements the first compiler slice of #181, not a new solver or a
complete molecular-gradient endpoint.

## Mathematical contract

For independent state blocks `x` and source fields `q`, declare

```text
R(x, q) = 0
L(x, q, lambda) = E(x, q) + lambda^T R(x, q)
R_x^T lambda = -E_x
source weights = E_q + R_q^T lambda
```

`StationaryState` binds one input to one residual output, with explicit coordinate,
gauge and inner-product identities. Constraint multipliers are ordinary state
blocks with `kind="constraint"`, bound to their constraint equations. The other
residual blocks must include their actual KKT stationarity equations. Neither
constraints nor orthogonality/overlap terms can be inferred from an energy alone.

`StationaryProblem.compile()` builds the Lagrangian from the declared residuals
and uses the existing #151 TensorIR reverse generator. It returns:

- `lagrangian`: the explicit scalar L;
- `rhs`: `-E_x`, keyed by state name;
- `partials`: `E_x + R_x^T lambda` and source-field weights, with explicit
  `stationarity_outputs`, `weight_outputs` and `multiplier_inputs` maps.

The adjoint solver must supply qualified multipliers. A small state partial checks
only the adjoint equation at the supplied point; it does not establish primal
convergence, conditioning or freshness. The plan does not run an implicit solve,
record iteration histories, or construct a dense state Jacobian. Tiny dense
linear solves in the tests are independent analytic oracles, not production code.

## Minimal example

```python
import numpy as np
from vibeqc_compiler.method.stationary import (
    ParameterSource, StationaryProblem, StationaryState,
)
from vibeqc_compiler.tensor import (
    Program, TensorSpec, add, execute, input_tensor, multiply,
)

spec = TensorSpec(role="parameter", differentiable=True)
x, q = input_tensor("x", spec), input_tensor("q", spec)
problem = StationaryProblem(
    equations=Program({
        "energy": add(multiply(multiply(x, x), x), q, coefficients=(1, 2)),
        "residual": add(multiply(x, x), q, coefficients=(1, -1)),
    }),
    objective="energy",
    states=(StationaryState("x", "residual", "scalar-x-v1", "positive-root-v1"),),
    sources=(ParameterSource("q", "external-q-field-v1"),),
    model_identity="scalar-nonvariational-v1",
    solver_contract="analytic-toy-transpose-solve-v1",
)
plan = problem.compile()
q_value = 2.3
x_value = np.sqrt(q_value)
# Analytic toy adjoint, not a general solver: (2*x)*lambda = -3*x*x.
multiplier = -1.5 * x_value
result = execute(plan.partials, {
    "x": np.asarray(x_value),
    "q": np.asarray(q_value),
    plan.multiplier_inputs["x"]: np.asarray(multiplier),
}).outputs
assert abs(result[plan.stationarity_outputs["x"]]) < 1e-12
assert abs(result[plan.weight_outputs["q"]] - (2 + 1.5*x_value)) < 1e-12
```

The example's fully re-solved objective is `q**1.5 + 2*q`, providing a separate
finite-difference check rather than only generator/interpreter agreement.

## Provider DAG and derivative cut

Each `ParameterSource` identifies a physical field, not merely a provider object.
For example, overlap and one-electron Hamiltonian fields can share one geometry
ancestor but must have distinct field identities. Two aliases of the same field
are rejected; reuse one TensorIR input name to accumulate all its appearances.

A derived source declares its upstream `dependencies` and `pullback_identity`.
Ancestors need not occur as TensorIR inputs. The graph validates complete source
coverage, missing dependencies, provider cycles, unused declarations and duplicate
field identities. Equation dependencies are derived from actual live TensorIR SSA
edges, not a second manually maintained equation list. The state/residual graph is
an explicitly cut implicit region; provider cycles and provider edges back to a
state are rejected rather than treated as an ordinary acyclic pullback.

**Provider declarations remain an inventory, while declared implicit state rules
are executable compiler dispatch.** A state with `implicit_operator_identity`
causes `compile()` to build an identity-bearing #465 `ImplicitVJPPlan` directly
from that residual graph. The residual must be an independent state block; a
coupled KKT/CC row is rejected until the caller exposes the coupled unknowns as
one explicit independent state. This prevents silently solving only part of a
stationary system.

The generated source plan still stops at live independent TensorIR source inputs.
Its `provider_pullbacks` property lists outstanding upstream provider contracts
in reverse topological order. A named matrix-function/overlap provider rule is
not run merely because it appears in the source DAG. Where both an ancestor and
descendant are direct inputs, the provider adapter must add direct and propagated
cotangents exactly once before pulling back the ancestor.

This boundary preserves missing overlap/Fock/metric branches when they are in the
manifest. No compiler can detect a scientific term omitted from both the declared
manifest and the equations; independent method-level fixtures remain necessary.

## Identity, replay and resource boundaries

The mathematical identity includes objective/residual equations, state layouts,
gauges, equation kinds, source field and pullback identities, model identity and
solver contract. Order of source/state declarations is canonicalized. Runtime
pointers, iteration histories and timestamps are not part of this schema.

Problem and derivative-plan JSON are data-only and reject unknown/missing fields,
version changes, duplicate JSON keys and identity mismatches. Derivative-plan
replay regenerates the fragments from the problem and compares the supplied
artifacts; trusting a claimed derivative hash alone is insufficient.

The first composition boundary accepts real FP64 independent state coordinates
with the explicitly declared Euclidean inner product. Redundant symmetric or
packed state coordinates are rejected, not flattened with guessed multiplicities.
Dense symmetric **source** fields retain the shared TensorIR projected adjoint.
Packed coordinates and non-Euclidean state metrics require an explicit compatible
#465 adapter. A coordinate/gauge label is not a numerical certificate.

Generated fragments can be independently passed to the existing TensorIR CPU
interpreter or CUDA planner. `max_elements` preserves #151's generation limit for
projection/incidence constants. It is not an overall live host/device byte limit.
Separate fragment plans do not establish the simultaneous primal/adjoint/provider
resource budget required by a molecular runtime. CUDA planning is not real-device
execution evidence. No public force capability or native execution path changes.

## Ownership and next integration

MethodIR (#396) remains the method/component front end. The existing semilocal
`StationaryGradientPlan` (#163) remains a specialized fixed-stationary-density
source-contraction consumer and is not replaced by this schema.

The implicit-solve primitive (#465) owns response execution, true residuals,
solver failure propagation and current-state binding. StationaryProblem v2 now
dispatches explicitly declared independent state blocks to that primitive; the
runtime may bind the resulting plan to an existing qualified ResponseProblem
operator without replacing generated source VJPs. The symmetric matrix rule
(#466) owns its qualified spectral mathematics and remains separate.

Subsequent #181/#193 slices must compose upstream provider pullbacks, #466 metric
response and complete nuclear derivatives into supported public force endpoints.
#460's optional execution/lifetime graph is not required here. Higher-order
derivatives and Hessian composition remain separately registered work; a
first-order implicit VJP does not grant second-order support.

Run the independent scalar, nonsymmetric and constrained tests with:

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/python/test_stationary_problem.py
python tools/check_compiler_structure.py
```

See the [composition-boundary decision](../.agents/notes/implemented/architecture/2026-09-19-stationary-problem-composition.md)
for the rationale and deliberately rejected alternatives.

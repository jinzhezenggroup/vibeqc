# Generated RCCSD Lambda equation actions

`tools.vibeqc_cc.build_lambda_programs(nocc, nvir)` generates the fixed-amplitude
energy gradient, residual Jacobian-vector product and transpose action for the
same conventional real RCCSD equations as `build_ccsd_program`. It is an internal
mathematical frontend, **not a Lambda solver or nuclear-gradient capability**.

## Equations and coordinates

For the unpreconditioned physical residual `R(t; q)` and correlation energy,

```text
L(t, lambda; q) = E_corr(t; q) + <lambda, R(t; q)>
J* lambda = -grad_t E_corr
```

`q` denotes fixed Fock/MO-integral blocks. All amplitude/response inputs use
FP64 dense spatial arrays: `t1[i,a]` and simultaneous-pair-symmetric
`t2[i,j,a,b] = t2[j,i,b,a]`. No separate occupied or virtual antisymmetry is
imposed. The residual normalization remains the audited opposite-spin doubles
projection from the primal frontend.

The inner product is the full dense Frobenius product. The doubles reverse
result is projected as `(bar + bar.transpose(1,0,3,2))/2`. This is not an optional
postprocessing convention: it is the adjoint in the constrained amplitude
space. For independent packed coordinates with orbit metric `W`, the equivalent
transpose is `W^-1 J_p^T W`. Ordinary unweighted packed transposition is wrong.
A Euclidean Krylov solver can instead use square-root-weighted coordinates.

These multipliers are derivatives of the stated Lagrangian. They must not be
labeled PySCF Lambda arrays or physical density matrices without an explicit
normalization/convention conversion and validation.

## Generated programs

The result contains `primal` and three derivative programs:

| Attribute | Additional input seeds | Returned derivatives |
| --- | --- | --- |
| `energy_vjp` | `bar_correlation_energy` scalar | `bar_t1`, `bar_t2` |
| `residual_vjp` | `bar_singles_residual`, `bar_doubles_residual` | `bar_t1`, `bar_t2` |
| `residual_jvp` | `d_t1`, `d_t2` | `d_singles_residual`, `d_doubles_residual` |

For example, given the same mathematical input `feeds` used by the primal:

```python
import numpy as np
from tools.vibeqc_cc import build_lambda_programs
from vibeqc_compiler.tensor import execute

programs = build_lambda_programs(nocc, nvir)
rhs = execute(
    programs.energy_vjp.program,
    {**feeds, "bar_correlation_energy": np.asarray(-1.0)},
).outputs
```

A seed of **-1** supplies the RHS of the stated Lambda equation. A seed of +1
supplies the positive energy gradient. `provenance()` records all primal and
derivative hashes and the sign/inner-product convention. Each `Program` retains
normal TensorIR data-only serialization and replay.

No CC solver iterations, level shifts, DIIS operations or denominator
preconditioners enter these derivative programs. Arbitrary fixed amplitudes
are permitted for differentiation tests; construction does not assert that
the amplitudes describe a converged CC state.

## Symmetry, resources and backends

The shared TensorIR generator supports dense symmetry-constrained inputs.
Forward seeds retain their input symmetry checks. Reverse seeds are projected
onto the full signed permutation group, not sequentially onto potentially
noncommuting generators. The projector uses ordinary transpose/add nodes and
exact rational coefficients; no coordinate-incidence matrix is introduced.
Symbolic group expansion rejects more than 4096 signed permutations. Existing
explicit `packed=` expansion and its weighted-metric rules are unchanged.

The CPU interpreter's existing logical-buffer budget applies independently to
each execution; it is not a process-RSS cap. The generated RCCSD action DAGs
contain no amplitude-by-amplitude Jacobian, dense packing-incidence matrix or
cross-iteration tape. This does not yet establish a composed native
primal/adjoint/Krylov peak-memory bound.

CPU numerical tests and CUDA **planning** are covered. The CUDA planner retains
its normal provider/workspace reservations. Actual CUDA compilation/execution,
resident Lambda state, molecular Lambda convergence and native/public response
APIs are not validated by this slice.

## Validation and remaining consumers

`test_cc_lambda.py` checks independent determinant-space directional differences
at three step sizes, a tiny explicit numerical Jacobian, weighted transpose
identities, symmetry factors, all three primal equation forms, replay, rejection
of invalid seeds and CUDA planning. A fixed-amplitude interoperability test uses
the existing shared GMRES, not a new CC-specific solver. Its dense numerical
Jacobian is a test-only oracle.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/python/test_tensor_ad_symmetry.py \
  tests/python/test_cc_lambda.py
```

The solved-state consumer still needs validated SCF/CC identity binding,
independent physical CC/Lambda residuals, shared implicit-solve integration and
complete state/workspace admission. Fixed-orbital parameter/RDM weights and
orbital/Z-vector/nuclear response remain separate work. In particular, this
frontend must not enable `compute_forces` or close the complete #152/#153 tasks.

See the [dense-symmetry adjoint decision](../.agents/notes/implemented/numerics/2026-09-19-dense-symmetry-cc-adjoints.md)
for the representation choice and rejected alternatives.

# RCCSD Lambda and fixed-orbital CPU input response

`tools.vibeqc_cc.build_lambda_programs(nocc, nvir)` generates the fixed-amplitude
energy gradient, residual Jacobian-vector product and transpose action for the
same conventional real RCCSD equations as `build_ccsd_program`. It is an internal
mathematical frontend. `BoundCCSDLambda` additionally solves the amplitude-response
equation from a converged internal CPU CCSD result. Neither interface enables
a native/public response method or a nuclear-gradient capability.

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
resident Lambda state and native/public response APIs remain unqualified.
The separate bound CPU consumer below establishes small molecular Lambda solves;
it does not promote those same actions to a GPU solver.

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

## Bound converged-state CPU consumer

```python
from tools.vibeqc_cc import BoundCCSDLambda, LambdaOptions, SolverOptions, solve

# snapshot is the validated RHF ReferenceSnapshot used by this provider.
cc = solve(snapshot, provider, options=SolverOptions(
    residual_tolerance=1e-11, energy_tolerance=1e-13,
))
bound = BoundCCSDLambda(snapshot, cc, options=LambdaOptions())
lambda_result = bound.solve(reference_identity=snapshot.identity)
```

`BoundCCSDLambda` accepts the internal `CCSDResult` from `solve`, or `.state`
from the energy facade, not an arbitrary amplitude tuple. It verifies the
RHF/reference and conventional-unscreened Hamiltonian identities, both CC
equation identities, the replayed reference and Fock/integral hashes, and the
actual finite FP64 amplitude shapes and simultaneous T2 symmetry. Replayed
Fock blocks must also match `C.T @ F @ C` of the supplied reference. It then
recomputes energy and physical R1/R2 with the expanded primal equations;
claimed convergence or a small historical residual is not sufficient.

The bound numerical feeds are copied into immutable bytes-backed arrays.
Subsequent changes to the caller's replay dictionaries cannot alter a solve.
Without `current_reference`, this is explicitly a **detached immutable tooling
snapshot**. A runtime owner can pass `current_reference=callback`, where the
callback checks provider lifetime and returns the current reference/generation
identity. It is checked before and after actions and publication. Every solve
also requires the expected reference identity explicitly. A detached snapshot
is not a claim that its originating native provider remains current or open.

The existing generated RHS and transpose act in dense Frobenius coordinates.
The thin block adapter uses the existing `PackedLayout` maps and passes
`sqrt(W) * lambda_p` to the shared solver, preserving all doubles orbit weights.
`checked_transpose_solve` and `ResponseGMRES` are shared with the generic
implicit-VJP adapter; the #179 GMRES implementation is unchanged. The CC
consumer does **not** flatten redundant T2 into an `ImplicitSolveSpec`, build a
dense mapping matrix, differentiate iterations, or implement another solver.

Acceptance is independent at each boundary:

- Fresh physical CC singles/doubles maxima must satisfy `cc_tolerance <= 1e-9`.
- The generic checked solver reevaluates the true transpose residual, validates
  returned solution/status/resource diagnostics, and checks solver identity.
- The consumer reevaluates Lambda stationarity using the expanded derivative
  programs. Its physical maximum and Euclidean-weighted norm, and the original
  true-residual norm, must satisfy `lambda_tolerance <= 1e-9` regardless of a
  looser injected solver tolerance.

A rejected primal, stale state, nonfinite value, failed/stagnant adjoint or
insufficient workspace raises before a successful Lambda result is published.
`CCSDLambdaResult` separately reports SCF residual, CC singles/doubles maxima,
CC weighted norm, both Lambda residual norms, Lambda physical maximum,
iterations/actions, exact state/equation identities and execution backends.
`lambda1` and `lambda2` retain the stated Lagrangian normalization; they are not
unconverted external Lambda arrays or physical RDMs.

## Resource and capability boundary

`LambdaOptions.max_bytes` admits the simultaneous logical numeric storage for
bound inputs/reference, amplitude-coordinate and publication scratch, the
largest generated interpreter program, and the opaque solver's declared
workspace **before replay conversion or interpreter execution**. The solver
also enforces its own workspace limit. This is a conservative tooling numeric
reservation, not measured native allocation or process RSS: caller-owned replay
lists, Python IR/layout objects, and opaque NumPy/BLAS workspaces are excluded.
No complete molecular host/device peak-memory or performance claim is made.
The existing packing-map reference implementation retains its element limit.

Only CPU interpreter actions and the host-controlled shared GMRES are enabled;
`backend="cuda"` is rejected explicitly. Fresh small-system native HF inputs use
the existing `export_rhf` bridge, including its disclosed host canonicalization.
Neither fixture input nor native export is misrepresented as a resident GPU
reference provider. CPU/native endpoint tests require no PySCF solver.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VIBEQC_LIBRARY=/path/to/current/cpu/libvibeqc.so python -m pytest -q \
  tests/python/test_cc_lambda_solver.py tests/python/test_cc_lambda.py \
  tests/python/test_implicit_vjp.py tests/python/test_implicit_response.py
```

The new tests compare converged molecular Lambda against tiny numerical
Jacobian solves and multi-step Lagrangian stationarity differences, include
native HF -> CC -> Lambda endpoints with changed geometry and a determinant
reference, and exercise stale/corrupt states, immutable outputs, false solver
success, true nonconvergence and resource rejection. Dense Jacobians are
strictly test-only. GPU execution/residency and native response-provider
integration remain part of #152 B. The CPU correlation-only input weights below implement the block-streamed
portion of #152 C; physical RDM conventions and orbital/Z-vector/nuclear
response remain separate work. No force capability or higher derivative is enabled here.

See the [bound-state decision](../.agents/notes/implemented/numerics/2026-09-19-bound-ccsd-lambda.md)
for why this consumer shares the generic checked-solve boundary rather than
building a dense packed-coordinate incidence matrix.

See the [dense-symmetry adjoint decision](../.agents/notes/implemented/numerics/2026-09-19-dense-symmetry-cc-adjoints.md)
for the representation choice and rejected alternatives.


## Fixed-orbital correlation input weights

`BoundCCSDResponse` takes the existing `BoundCCSDLambda` and its solved
`CCSDLambdaResult`. Before retaining multipliers, it checks the exact reference,
CC-state, equation, sign and execution conventions, copies the amplitudes into
immutable arrays and **freshly re-evaluates both physical Lambda equations**.
A successful status or fabricated zero-residual diagnostic is not sufficient.
Weight generation neither calls a CC/adjoint solver nor differentiates its
iterations. It uses the same shared and expanded primal DAGs with #151 VJPs.

```python
from tools.vibeqc_cc import BoundCCSDResponse

response = BoundCCSDResponse(bound, lambda_result, max_bytes=256 << 20)
for weight in response.iter_weights(reference_identity=snapshot.identity):
    # dq must be this field's fixed-orbital, symmetry-compatible direction,
    # associated with the exact same snapshot and mathematical input convention.
    contribution = weight.contract(dq[weight.parameter])
    # Contract/release each block rather than retaining the entire sequence.
```

The mathematical boundary is

```text
q = (foo, fov, fvv, ovov, ovvo, oovv, ovvv, ovoo, oooo, vvvv)
L = E_corr(T; q) + <lambda1, R1(T; q)> + <lambda2, R2(T; q)>
bar_q = partial E_corr / partial q + (partial R / partial q)*lambda
```

`build_parameter_vjp(primal, parameter)` seeds the energy with +1 and the two
physical residuals with their Lambda arrays. Its sole output is
`bar_<parameter>`. The shared AD machinery projects the dense Frobenius weight
onto the parameter's declared symmetry; no independent-coordinate incidence
matrix is constructed. `CCSDParameterWeight.contract` rejects incompatible
shape/dtype, nonfinite values or a direction that violates those symmetries,
rather than silently projecting a different perturbation.

**These are amplitude-relaxed, orbital-unrelaxed derivatives of correlation
energy with respect to independent mathematical input blocks.** In particular:

- F is held independent of the chemists'-notation g blocks. The upstream
  normal-ordering `g -> F` and `h -> F` chain is NOT included. Perturbing g while
  fixing F is not the same experiment as perturbing g while fixing raw h.
- The HF reference energy, orbital/overlap response, nuclear terms and physical
  1-/2-RDM reconstruction are NOT included. Unconverted weights are not RDMs.
- Diagonal F contributions appear through the physical residual equations.
  Solver denominators, level shifts and DIIS updates are preconditioners, not
  extra physical energy dependencies to differentiate.

Several input blocks overlap or are permutations of the same physical ERI
field. For a physical full-g perturbation, map its direction into ALL affected
blocks and sum their dense contractions once. Do not treat the blocks as a
completed raw-g pullback or add ad hoc spin/symmetry factors. The tiny
independent determinant oracle reconstructs h from the supplied F and g,
matching this fixed-F convention before comparing correlation energies.

Every requested block is checked against the expanded-equation VJP using
absolute tolerance 1e-12 and relative tolerance 1e-10. Each form's output is
frozen before the other runs, so executor-buffer reuse cannot make the two
results alias and pass accidentally. The reference/lifetime contract is checked
before/after every execution and immediately before returning a weight. State,
Lambda and generated-program identities accompany each result.

## Streaming and validation boundary

Only one parameter output is generated per request. Consumers can stream blocks
without assembling a complete NMO^4 2-RDM. **The selected block itself remains
dense**, including vvvv; intra-block tiling, resident GPU weights and a full
native provider-resource contract remain unqualified. The response object does
not cache all generated blocks. `iter_weights` is a sequence of individually
qualified results, not an atomic full-gradient transaction. A late stale-state
or numerical failure raises instead of returning that block; previously yielded
immutable blocks retain their original response identity.

The response budget includes the bound state's existing simultaneous logical
reservation, owned Lambda values and scratch, the larger live generated graph,
and independent/output-copy storage for the current block. Admission precedes
numeric execution. As for the Lambda consumer, Python IR objects and opaque
NumPy/BLAS allocations are excluded; earlier blocks deliberately retained by a
caller are also outside the per-request reservation. No process-RSS, production
peak-memory or endpoint speedup claim follows from this estimate.

`tests/python/test_cc_lambda_response.py` validates all ten water input blocks at
three finite-difference steps by re-solving physical CC equations after each q
perturbation. It separately checks fixed-T energy partials and requires cases
where omitting Lambda changes the answer. The fixed-q test seam changes only
prepared equation feeds; it does not export a noncanonical displaced F as a
converged RHF snapshot or weaken production reference validation. Additional
native HF -> CC -> Lambda -> weight endpoints use H2, displaced H2 and H4, and
re-solve an independent determinant-space CC problem for each perturbation.
These native tests invoke no PySCF solver. Tests also cover input symmetries,
uniform diagonal-F shifts, graph replay, stale/corrupt multipliers, output-buffer
aliasing, false generated weights and state/block memory admission.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VIBEQC_LIBRARY=/path/to/current/cpu/libvibeqc.so python -m pytest -q \
  tests/python/test_cc_lambda_response.py tests/python/test_cc_lambda_solver.py \
  tests/python/test_cc_lambda.py tests/python/test_implicit_vjp.py
```

The method-level `StationaryProblem` still requires independently parameterized
state blocks; redundant dense T2 is not silently admitted into it. This consumer
reuses the existing symmetry-qualified equations and #151 derivative machinery,
not a second scientific algebra. See the
[input-weight boundary decision](../.agents/notes/implemented/numerics/2026-09-19-ccsd-fixed-orbital-weights.md)
for the ownership choice and conditions for migration to generic composition.

# Symmetric matrix-function custom rule

`vibeqc_compiler.method.SymmetricMatrixFunctionSpec` defines a backend-neutral
matrix-function contract and CPU reference implementations of symmetric inverse
square root and pseudoinverse first-order JVP/VJP rules. It is a reusable
mathematical primitive, not a DFT energy component or an MP2-specific response
implementation.

## Supported boundary

The current implementation supports real FP64 symmetric matrices, the full
Frobenius inner product, and first derivatives of `inverse_sqrt` and
`pseudoinverse`. A missing relative threshold selects the full SPD branch. A
positive relative threshold selects a truncated locally stable rank branch.
Complex values, packed metrics, other matrix functions and second
matrix-function derivatives are not supported.

CPU eigendecomposition prepares the reference spectral state. The derivative
contractions are emitted as an ordinary, serializable TensorIR `Program`,
interpretable on CPU and accepted by the existing CUDA planner. For runtime-sized
native consumers, `matrix_function_cuda.py` emits one shared CUDA custom-rule
lowering with inverse-square-root and pseudoinverse entry points. Native code
still owns factorization, eigensystem/rank validation, scratch, streams and
method integration; the generated custom rule owns only the spectral response
arithmetic.

```python
import numpy as np
from vibeqc_compiler.method import SymmetricMatrixFunctionSpec

spec = SymmetricMatrixFunctionSpec(2, "example-metric", relative_threshold=0.1)
state = spec.prepare(np.diag([0.0, 4.0]))
value = state.value  # diag(0, 0.5)
seed = np.array([[0.0, 1.0], [1.0, 0.0]])
tangent = state.jvp(seed)  # off-diagonal entries 1/8, NOT zero
cotangent = state.vjp(seed)
program = spec.response_program()
```

The emitted program takes `vectors`, `divided`, and `seed` feeds and produces
`response`. These spectral feeds are fixed coefficients **at the current primal
matrix**, not independently differentiated parameters. TensorIR differentiation
of this linear map does not supply a second derivative of the matrix function.
An upstream generated objective VJP can supply its `bar_X` as the seed; full
method-level automatic custom-rule composition is not implied by this example.

## Spectral and branch contract

For `M = U diag(lambda) U.T`, define `f_i` as either
`1/sqrt(lambda_i)` (`inverse_sqrt`) or `1/lambda_i` (`pseudoinverse`) on
retained modes and zero on discarded modes. The derivative is

```text
L_M(E) = U [K * (U.T sym(E) U)] U.T
K_ij = (f_i - f_j) / (lambda_i - lambda_j)
```

For `inverse_sqrt`, retained/retained entries use the cancellation-free
`K_ij = -1 / (sqrt(lambda_i) * sqrt(lambda_j) *
(sqrt(lambda_i) + sqrt(lambda_j)))`. For `pseudoinverse`, they use
`K_ij = -1/(lambda_i*lambda_j)`. Both include the repeated-eigenvalue limit
without differentiating arbitrary eigenvector orientations. Discarded/discarded
entries are zero. **Retained/discarded entries are not zero**: a fixed rank does
not freeze the spectral projector. Degeneracy within one side of the cutoff is
legal; an unresolved cutoff or a rank change on explicit rebinding is not.

`branch_guard` is relative to the largest eigenvalue. It guards the distance to
the truncation cutoff (or to zero in full-rank mode), independently of matrix
units. Truncated mode admits PSD nullspaces and small negative discarded modes
within the stated relative guard, but rejects materially indefinite matrices.
No eigenvalues are silently clipped. Full-rank mode requires an SPD matrix.

`state.rebind(new_matrix)` recomputes the spectral projector and rejects a
changed rank. A fresh `spec.prepare` can deliberately select another branch;
there is no hidden global rank cache. Rebinding checks endpoint rank and local
spectral separation, not every point along an arbitrary finite interpolation.
JVP inputs must be symmetric. VJP seeds may be nonsymmetric and are projected
onto the symmetric domain with the full-Frobenius transpose convention.

## Identity, ownership and resources

The spec has strict `to_payload`/`from_payload` round trips and deterministic
identity, including matrix-provider identity, function, threshold, branch guard,
dtype, derivative order, inner product and rule version. The state manifest
adds the input-byte hash, retained mask/rank, relative gap and response-program
hash. Changing geometry/matrix data requires a fresh state. Spectral arrays and
values are detached immutable snapshots; responses are detached outputs.

`logical_workspace_bytes` is a conservative CPU logical-array admission estimate.
The preflight occurs before factorization; it covers spectral state and generated
interpreter arrays. It is **not** a measured allocation peak or a bound on NumPy,
BLAS/LAPACK internal scratch, Python/JSON storage, process RSS, or caller-owned
arrays. Production native/device resource accounting is still required.

## Validation

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/python/test_matrix_function.py -q
python tools/check_compiler_structure.py
```

The tests use tiny matrices: closed-form scalar/diagonal derivatives, an
independent Kronecker/Sylvester solve, pseudoinverse closed forms and finite
differences, full-Frobenius dot tests, repeated-eigenspace rotations,
retained/discarded response, scale/rotation covariance, state/rank failures,
and the independent #293 metric-response oracle. The production generated CUDA
header carries both versioned `inverse-sqrt-frechet-v1` and
`pseudoinverse-frechet-v1` rule identities; the explicit Slurm CUDA tier checks
both entry points over exact FP64 scalar fixtures.

See the [decision note](../.agents/notes/implemented/numerics/2026-09-19-symmetric-matrix-function-rule.md)
for the ownership rationale and rejected alternatives.

Native CPU and generated CUDA spectral coefficients use ordered divisions and
overflow-safe symmetrization for representable FP64 responses. See the
[spectral-range decision](../.agents/notes/implemented/numerics/2026-09-19-native-spectral-range.md).

# Symmetric matrix-function custom rule

`vibeqc_compiler.method.SymmetricMatrixFunctionSpec` defines a backend-neutral
matrix-function contract and a CPU reference implementation of the symmetric
inverse square root and its first-order JVP/VJP. It is a reusable mathematical
primitive, not a DFT energy component or an MP2-specific response implementation.

## Supported boundary

The current implementation supports real FP64 symmetric matrices, the full
Frobenius inner product, and first derivatives of `inverse_sqrt`. A missing
relative threshold selects the full SPD inverse square root. A positive relative
threshold selects a truncated symmetric inverse square root on a locally stable
rank branch. Complex values, packed metrics, other matrix functions and second
matrix-function derivatives are not supported.

CPU eigendecomposition prepares the spectral state. The derivative contractions
are emitted as an ordinary, serializable TensorIR `Program`, interpretable on
CPU and accepted by the existing CUDA planner. **CUDA planning is not device
execution**: native factorization, device-state ownership, automatic dispatch of
this custom rule from a complete method graph, and public molecular force
integration remain separate work. No public method capability changes here.

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

For `M = U diag(lambda) U.T`, define `f_i = 1/sqrt(lambda_i)` on retained modes
and zero on discarded modes. The derivative is

```text
L_M(E) = U [K * (U.T sym(E) U)] U.T
K_ij = (f_i - f_j) / (lambda_i - lambda_j)
```

Within retained modes the cancellation-free formula is
`K_ij = -1 / (sqrt(lambda_i) * sqrt(lambda_j) *
(sqrt(lambda_i) + sqrt(lambda_j)))`. It includes the repeated-eigenvalue limit
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
independent Kronecker/Sylvester solve, multistep finite differences, full-Frobenius
dot tests, repeated-eigenspace rotations, retained/discarded response, scale and
rotation covariance, state/rank failures, and the existing #293 metric-response
oracle. The latter remains unchanged and is never a compiler dependency.

See the [decision note](../.agents/notes/implemented/numerics/2026-09-19-symmetric-matrix-function-rule.md)
for the ownership rationale and rejected alternatives.

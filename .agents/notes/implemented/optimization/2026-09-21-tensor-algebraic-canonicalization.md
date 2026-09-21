# Decision: extend TensorIR canonicalization without floating-point reassociation

Status: implemented
Date: 2026-09-21

## Decision

Continue #682 Slice B after the lossless view pass with a separate `algebraic_canonicalization` stage. The first algebraic rules remove only elementwise multiplication or division by an exact literal tensor of ones when the surviving value has the same TensorIR semantics apart from SSA role.

Do not simplify `x + 0`, a single-term `1*x` encoded as ordered `add`, `x * 0`, or reassociate nested sums/products. TensorIR's ordered add starts from positive zero, so apparently neutral rewrites can change the sign bit of `-0.0`. Absorbing-element rewrites can likewise change observable IEEE behavior. No fast-math assumption is introduced.

Generalize the existing constant-folding pass from rank-zero literals to constant tensors. Candidate literals are formed with exact rational arithmetic and promoted only when interpreter execution of the original DAG and candidate is bitwise identical. Division-by-zero and non-finite diagnostics remain on the original path.

Extend lossless view/index cleanup to compose nested static slices and nested same-axis gathers. Reconstructed nodes pass ordinary TensorIR validation, so index-space and selection semantics remain fail-closed.

The TensorIR optimizer pipeline/version and affected pass versions change so artifact/provenance identity reflects the new transformations.

## Evidence

- Full `tests/python/test_tensor_*.py`: 619 passed, 117 skipped.
- Focused canonicalization/pass-manager regression: 14 passed.
- Compiler dependency audit: 259 modules, 0 dependency errors.
- Ruff check and format checks pass.
- Signed-zero tests prove ordered add is intentionally retained.
- Exact-one multiply/divide tests compare output bytes, including `-0.0`.
- Constant tensor folds retain the existing bitwise promotion gate.

## Follow-up

Continue #682 Slice B/C with more proven canonical forms and then GVN/CSE. Do not add commutative reordering, reassociation, absorbing-zero rules, or profitability-dependent transforms until their numerical and resource contracts are explicit.

---
Agent: ChatGPT
Model: GPT-5.6 Sol

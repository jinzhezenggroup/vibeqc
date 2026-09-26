# Decision: TensorIR transcendental primitives and derivative error boundaries

Status: implemented
Date: 2026-09-19
Related: #500 (under #499)

## Problem

Semiempirical and geometry expressions need nonlinear scalar functions in the
same TensorIR graph as contractions and generated derivatives. Adding only
NumPy evaluation or standalone CUDA calls would leave replay, generated AD,
resident execution, and domain diagnostics incomplete.

## Decision

Add first-class unary `exp`, `log`, `sqrt`, and `power` nodes. `power` has a
static, normalized rational exponent (integer, Fraction, or rational string),
not a second differentiable operand. Exponents must remain finite and nonzero
when a nonzero value is rounded to the operand dtype. Runtime parameter tables
and dynamic exponents are not part of this slice.

The real-domain contract is explicit:

- `exp`: finite inputs and finite results.
- `log`: strictly positive inputs.
- `sqrt`: nonnegative inputs for values; zero has no finite first derivative.
- `power`: strictly positive bases, including for zero and integer exponents.

Inputs, results, and required derivative intermediates retain the existing
finite-arithmetic checks. Ordinary floating-point underflow is not replaced by
fast math or implicit higher precision. There is no claim of correctly rounded
`exp`/`log`/`power`, bitwise CPU/GPU equality, or globally range-extended AD for
all finite mathematical seeded derivatives. CPU reference execution supports
FP32/FP64; the existing CUDA baseline remains explicitly FP64.

Nonlinear functions retain positive permutation symmetries, but do not inherit
antisymmetry. Arity, attributes, exponent canonicalization and metadata are
validated at node construction and on deserialization.

## Derivatives

Both reference JVP/VJP and generated derivative programs are implemented.
Logarithm differentiates as `seed/x`, not `(1/x)*seed`, so a subnormal input
need not overflow an otherwise representable seeded derivative. Square root
uses `seed/(2*sqrt(x))`, with division rejecting the singular point at zero.
Power differentiates through `x**(p-1)`, with explicit p=0 and p=1 cases; it does
not divide a possibly underflowed primal power by x.

Generated logarithm/power derivatives keep zero-weighted dependencies on the
original primal node. Optimization, replay, and CUDA fusion must preserve
these error boundaries: a finite slope does not authorize an invalid or
nonfinite primal calculation. The same principle applies to an exactly zero
seed. Higher derivatives remain ordinary TensorIR DAGs.

## CUDA and compatibility

CUDA uses the standard double-precision device math functions in the shared
scalar emitter. Ordinary and resident executors consume exactly the same
mathematics and domain checks. Domain failures have a disjoint negative error
code range beyond the existing division codes; the planner bounds that range.
The resident path still transfers only the error status during evaluation and
rejects output leases after a failed or subsequent run.

No global mathematical or AD schema version is changed: existing operations
and their derivative rules are untouched. New operation names/attributes enter
logical identities normally. Compiler source inventories still invalidate
compiled artifacts after source changes. Twelve existing example/schedule
combinations were checked against base `32b0095bf5f0aed3b92190a8fd9be7979cd42443`:
logical hashes, plan identities, ordinary CUDA source and resident CUDA source
remain unchanged.

## Evidence

`tests/python/test_tensor_transcendentals.py` checks 80-digit Decimal references,
multistep finite differences, both reference/generated AD directions, second
derivatives, scalar/noncontiguous/empty arrays, rational exponent identities,
packed-AD rebuilding, symmetry metadata, floating-point boundaries, and error
preservation after optimization/replay. Existing TensorIR/compiler regression:
326 passed, 47 conditionally skipped on the initial complete CPU run.

`tests/python/test_tensor_transcendentals_cuda.py` is opt-in within an allocated
GPU job. It covers ordinary/resident primal/JVP/VJP, fusion/recomputation,
forbids CPU interpreter arithmetic during device execution, and tests failure,
recovery and stale leases. Raw logs/cache are ignored under `.artifacts/issue500/`.
Actual Slurm-allocated RTX 5090 execution with CUDA 12.9.86 passed all 10
new device tests in 103.78 seconds. CPU and device runs use the same worktree
and equations. Compiler structure (187 modules), CUDA ownership (183 files),
SCF structure (217 modules), evidence-retention, Ruff and diff checks passed.

## Rejected alternatives and revisit conditions

Do not lower power through `exp(p*log(x))`, erase primal checks from derivative
programs, infer odd-function symmetry, or introduce a second scalar IR/host
callback. Dynamic exponents, broader real-power domains, range-extended weighted
power derivatives, new special functions and CUDA FP32 need explicit numerical
contracts and independent tests. SCC and method runtime policy remain outside
this compiler change.

References: #500; [NumPy power](https://numpy.org/doc/stable/reference/generated/numpy.power.html);
[NVIDIA CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html).
Validation compiler: CUDA 12.9.86.

Agent: ChatGPT
Model: GPT-6 Astra Pro

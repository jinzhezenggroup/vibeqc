# Decision: scale the complete TensorIR division derivative

Status: implemented
Date: 2026-09-19
Agent: ChatGPT
Model: GPT-6 Astra Pro

## Problem

Issue #477 reproduces spurious overflow/underflow in division JVP/VJP on
`x=y=1e200` and `x=y=1e-200`. Both the reference rules and generated programs
form `y*y`. The JVP's numerator products can independently overflow even when
the final tangent is finite through cancellation.

Reassociating the VJP as `-(w/y)*(x/y)` is not a repair: with
`x=1e300, y=1e100, w=1e-300`, it loses the representable `bar_y=-1e-200`.

## Decision

Add the generic elementwise TensorIR primitive
`scaled_bilinear(a,b,c,d,e,f) = (a*b-c*d)/(e*f)`. It is a single arithmetic
boundary, not a method-specific gradient or an alternative IR. Each input is
split into a binary mantissa and integer exponent. Only normalized mantissas
are multiplied; their exact product residuals are retained before aligned
subtraction. Restore the exponent only after division by the normalized
product of the denominator mantissas.

The NumPy implementation uses a Dekker split for FP32/FP64 product residuals.
The generated FP64 CUDA implementation uses explicit round-to-nearest
multiply/add/subtract/divide and FMA intrinsics. The FMA here deliberately
computes the exact product residual; it is not an implicit reassociation.
The compiler owns the emitted mathematical helper; native resource templates
and runtime ownership are unchanged. No PySCF, runtime-oracle or host fallback
is introduced into generation or device arithmetic.

Products more than `2*p+4` binary exponents apart cannot cancel; the smaller
one is explicitly discarded before alignment rather than underflowing a
temporary. Zero products do not choose the alignment exponent. The denominator
mantissa product and final division are rounded normally: this is bounded-error
floating-point arithmetic, not a correctly-rounded arbitrary-precision ratio.

Division JVP uses the complete fused difference. Its numerator-only generated
path remains `dx/y`. VJP uses `w/y` and a fused denominator partial. Reference
VJP additionally tracks reachability from requested inputs and skips inactive
quotient partials before arithmetic, so an unrequested overflowing `bar_x`
cannot poison a finite requested `bar_y`.

AD rule and derivative-generation versions both change from 1 to 2. The new
primitive serializes, optimizes, rebuilds packed graphs, has JVP/VJP rules, and
uses the existing CUDA elementwise planning/fusion machinery.

## Scope and invariants

- Zero denominator factors always fail, including with a zero numerator.
- Existing nonfinite input, intermediate-node and final-result checks remain.
  Only the normalized internals of this explicit primitive share one boundary.
- Final subnormals and unavoidable underflow use ordinary dtype rounding.
- CPU reference and generated-program interpretation support FP32 and FP64.
  CUDA remains FP64-only and still explicitly rejects FP32 plans.
- The acceptance gate is 8 machine epsilons relative plus one minimum subnormal
  absolute, with explicit zero/nonzero checks in the regression corpus. No
  correctly-rounded promise is made at overflow/underflow rounding boundaries.
- Original division first derivatives are the range-hardening target. Higher
  AD remains expressible and ordinary-scale second derivatives are tested;
  arbitrary multi-node/higher-order AD is not a globally compensated reduction.
  In particular, higher-order denominator partials use the rounded primal and
  cannot recover information already lost outside this primitive boundary.

## Evidence

Baseline reproduced at `23f84797f42171c7ce4851e8396197a8ed33c3e7`.
`tests/python/test_tensor_scaled_division.py` uses exact binary rationals
(`fractions.Fraction`) as an independent test-only oracle. It covers both
precisions, sign changes, nonunit seeds, numerator-/denominator-only selection,
normal/subnormal endpoints, the reassociation sentinel, exact and near
cancellation, a seeded exponent sweep, replay/optimization, shape/error gates,
and an ordinary-scale second derivative.

The opt-in CUDA tests batch the analytic cases and exercise JVP, VJP, and
selected denominator VJP with fusion both enabled and disabled. They also
check zero-denominator/nonfinite-result diagnostics and handle reuse after
errors. Real-device validation uses an allocated Slurm job on an RTX 5090,
NVCC 12.9.86, `sm_120`, with existing CUDA 12.4 cuBLAS headers/libraries.

Reproduce CPU tests with `PYTHONPATH=python python -m pytest -q
 tests/python/test_tensor_*.py`. CUDA tests require an allocated device,
`VIBEQC_TENSOR_CUDA_TEST=1`, and an explicit working `VIBEQC_NVCC` toolchain;
run the new test module with `-k allocated_cuda`.

## Consequences and revisit conditions

The scaled primitive performs more scalar arithmetic than an unguarded formula;
this change makes no performance claim. A future bounded fast path must retain
the exact-rational tests and both the overflow and silent-underflow sentinels.
Do not replace it with a fixed multiplication/division ordering. Broader
range-safe higher-order differentiation requires a separate numerical contract.

References: #477; parent #137; TensorIR AD foundations #151, #213, #216.

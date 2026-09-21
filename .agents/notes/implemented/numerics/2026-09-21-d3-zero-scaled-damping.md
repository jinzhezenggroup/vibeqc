# Decision: scale complete D3(0) terms when the damping denominator overflows

Status: implemented qualification repair
Date: 2026-09-21

## Problem

The admitted positive damping exponent can make `6*(rs*R0/r)^alpha`
overflow although the final damped inverse power and its radial derivative
remain representable. Replacing the damping factor by zero prematurely loses
both results. No invalid input, colliding center, or density cutoff is needed.

## Decision

Keep the original arithmetic for finite denominators. Only when that denominator
overflows, combine the inverse-power and damping exponents in log2 coordinates
before producing the value. Scale the radial derivative independently, since
its extra inverse-square distance can survive energy underflow. In this branch
the discarded `+1` is far below FP64 resolution. Requested nonfinite final
outputs still fail the existing admission checks.

## Evidence and boundary

Two exact binary-power reference cases (powers six and eight) fail before the
repair and pass afterward. The expected results are derived directly from the
complete damped expression, not the implementation's logarithmic evaluation.
The existing independent simple-dftd3 and finite-difference checks remain
unchanged. This is standalone D3(0) qualification, not production promotion;
no cutoff, parameter domain, or existing tolerance is relaxed.

Agent: ChatGPT
Model: GPT-6 Astra Pro

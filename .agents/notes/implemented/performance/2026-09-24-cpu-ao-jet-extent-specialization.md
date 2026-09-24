# Decision: specialize the shared CPU AO traversal by legal jet extent

Status: implemented in PR #1178; final integration checks pending
Date: 2026-09-24

## Problem

The radial-reuse traversal initially retained a 20-value accumulator and runtime
jet/derivative dispatch even for order zero, where there is no exponential work
to reuse. An isolated exact-function A/B experiment reproduced a value-only
regression on the review host, contrary to the PR's non-regression condition.

The retrieved complete source files were verified by their Git blob hashes:

- original: `8bc04e4f1ef45f48358f4be748fa1fda92e6fa93`;
- initial radial reuse: `6037f7f873ff7c8c856cda0ddfd1d22e9c36006e`.

## Decision

Keep one written primitive/component traversal. Instantiate its legal accumulator
sizes 1, 4, 10 and 20 with a C++20 templated lambda. In the one-jet instantiation,
the derivative supplied to the existing polynomial helper is the known zero.
All other derivative indices still come from the established CCA enumeration.
There is no second AO formula or precision policy.

## Rejected alternatives

Compile-time accumulator sizes alone reduced but did not eliminate the local
order-zero regression (candidate/base ratios about 1.045--1.067). Making its
already-known derivative zero explicit removed the remaining dispatch cost.
A separately handwritten value-only formula would duplicate science; loosening
the non-regression condition or changing basis/grid inputs would conceal the
problem. Neither is used.

## Invariants

Primitive and Cartesian-expansion term order within every jet, exponential
underflow handling, finite/input checks, selected-AO behavior and jet-major output
layout are unchanged. The reduction still uses FP64 and the same polynomial
helper. The compile-time extents are exactly the already-admitted orders 0--3.

## Evidence and limits

AMD EPYC 9V74 review container, GCC 14.2, `-O3 -ffp-contract=off`.
The unchanged evaluator/helper bodies were compiled in one identical standalone
ABI harness per version; it supplies packed records instead of constructing the
full native library. Eight through-f atom fixtures, three signed primitives per
shell, 256 deterministic Gaussian-distributed points, 160 Cartesian or 128
real-spherical AOs; seven alternating-order rounds, three calls per sample.
Medians below are ratios to the original evaluator, not DFT endpoint speedups:

| Representation | Order | Initial radial reuse / original | Repaired / original |
| --- | ---: | ---: | ---: |
| Cartesian | 0 | 1.155 | 0.815 |
| Cartesian | 1 | 0.823 | 0.717 |
| Cartesian | 2 | 0.902 | 0.790 |
| Cartesian | 3 | 0.965 | 0.847 |
| Real spherical | 0 | 1.187 | 0.817 |
| Real spherical | 1 | 0.934 | 0.796 |
| Real spherical | 2 | 0.980 | 0.852 |
| Real spherical | 3 | 0.989 | 0.879 |

These are synthetic packed-fixture, single-host measurements. The two campaigns
were separate, each with interleaved original/candidate calls. They do not prove
profitability for every CPU, basis, installed build or complete SCF endpoint.

Sixteen representation/order/selection combinations match the original evaluator
bitwise and an independent closed-form Leibniz reference within 1.34e-15 absolute.
`test_cpu_ao_jet_numerics.py` separately compiles the actual evaluator and helper,
checks all through-f Cartesian monomials and signed d/f combinations against a
long-double Leibniz oracle, and exercises output canaries, selected AOs, far-field
underflow, empty points and malformed inputs: 20 cases passed locally. It does
not validate basis construction or the native library ABI. Existing full native
and DFT tests remain required and no numerical tolerance has been weakened.

## Revisit when

Matched full-library/end-to-end measurements reveal a shape or target regression,
or a compiler-owned CPU lowering can subsume this bounded native traversal while
preserving independent value/derivative and complete-consumer gates.

References: #1178, #682.

Agent: ChatGPT — Even-PR Review
Model: GPT-6 Astra Pro

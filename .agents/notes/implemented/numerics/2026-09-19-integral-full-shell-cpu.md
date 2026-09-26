# Decision: bound complete CPU integral shells over shared compiler DAGs

Status: implemented
Date: 2026-09-19
Issue: #351

## Problem

PR #468 proved one selected full-range ERI Cartesian component could be lowered
to native CPU C++ from the compiler-owned DAG also available to CUDA. The
remaining CPU integration still hand-partitioned component subsets in callers,
and there was no aggregate artifact identity or bounded complete-shell result.

The production CPU recurrence in `src/integrals/s_integrals.cpp` must not be
retired merely because one generated component agrees with it. That file is
also the structurally independent dynamic-`Jet` oracle.

## Decision

Add a complete-shell execution layer on top of the existing first-component
lowering. `first_derivative_component_tiles` deterministically partitions the
canonical Cartesian component domain into at most 64 components per generated
program. It creates no second recurrence or scientific IR.

`CompiledFirstDerivativeShell` binds the ordered tile set, tile size, integral
IR and an aggregate content identity. Every constituent artifact is revalidated
before use. Changing tile order, tile size, integral semantics or native binary
identity invalidates the aggregate artifact.
`FirstDerivativeShellEvaluator` executes one bounded tile at a time and only
returns after all tiles succeed. Its numeric budget accounts for the complete
returned shell plus the peak storage of one live tile. The default 4 MiB budget
covers the compiler-supported s/p/d/f four-center Cartesian domain, including
10,000-component ffff shells, without constructing an unbounded symbolic DAG.

Normalization, primitive contraction, spherical transformation and physical
atom scatter remain outside the recurrence. Backend-specific special-function
policy is unchanged. The independent CPU oracle is not generated and remains a
separate implementation.

## Correctness evidence

`tests/python/test_first_derivatives_native.py` validates a complete d-p-s-s
four-center shell split 5/5/5/3. It compares all 18 values and all twelve
shell-center first derivatives against `NativeSource`, which reaches the
independent dynamic-`Jet` Hermite/Coulomb implementation.

The matrix includes contracted primitives, tight/diffuse exponents, nearly
coincident centers, a partial final tile, translation invariance, malformed
aggregate identity and explicit numeric-budget rejection. The supported ffff
domain is also checked structurally as 157 bounded tiles with a 16-component
tail.

Local qualification on node3 with the current CPU native library:
`tests/python/test_first_derivatives_native.py` = 12 passed.

## Compile/resource sample

A fresh d-p-s-s cache with tile size five produced four native programs:
136,828 generated-source bytes and 110,336 bytes across the four shared
libraries. Cold compile wall time was 1.771 s and the evaluator-reported numeric
peak was 4,328 bytes. The aggregate program identity was
`5b1a82409b6035daff61f2718bd2894cb992e2c9ba134e5783d139b5721a3843`.

For the contracted d-p-s-s fixture on one CPU thread, 50 warm samples gave a
0.262 ms median for the generated complete value plus twelve derivatives. The
independent raw source took 0.106 ms median for the same shell values only.
Those timings are deliberately not presented as an endpoint speedup: the
oracle sample omits derivatives and neither path is a matched SCF consumer.
Reproduce the measurement with one CPU thread, a built `VIBEQC_LIBRARY`, and
`python benchmarks/issue351_full_shell_cpu.py --samples 50`.

## Consequences

This completes the bounded full-shell/runtime-consumer prerequisite recorded
after #468, but it does not satisfy #351's production-retirement acceptance by
itself. No default CPU SCF path changes and no handwritten oracle recurrence is
deleted in this slice. Promotion still requires matched endpoint evidence and
the backend specialization work tracked by #469/#470; #471 owns later CPU
schedule autotuning.

The full-shell artifact boundary is intentionally suitable for those follow-ups:
SIMD width, ISA target identity and CPU schedule policy can vary per tile
without forking the mathematical recurrence specification.

Agent: ChatGPT
Model: GPT-5.6 Sol

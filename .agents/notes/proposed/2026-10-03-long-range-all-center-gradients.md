# Proposal: share high-order LR moments across center derivatives

Status: experimental; host numerical gates pass, GPU and endpoints pending
Date: 2026-10-03

## Problem and decision

The existing compiler-emitted all-center full-range gradient covers total
angular orders 4, 5 and 6. The bounded LR consumer still repeats its separate
Dual3 contraction for every unique atom. Extend the same all-center primitive
and contraction with a default-false LR template parameter. Select it only for
LR orders 4--6 in the bounded range force consumer. One primitive/AO quartet
produces one LR moment ladder and all center derivatives before atom scatter.

At fixed primitive exponents and omega, LR moments satisfy
`dM_n/dT = -M_(n+1)`. The established shared range-moment owner therefore replaces
only the Boys ladder. The compiler keeps ownership of Wick pair coefficients,
Coulomb recurrence, canonicalization, primitive contraction and center mapping.
No independent recurrence or production CPU oracle is introduced.

## Invariants and fallback

Full-range is the default specialization and retains its existing arithmetic.
Low-order LR roots from #1756, short range, fused RSH, orders above six,
AO normalization, screening and source coefficients keep their existing paths.
This adds no retained allocation or data transfer; per-thread local gradients
remain bounded. The invalid-input branch publishes a compile-time NaN marker
for every coordinate, preserving the downstream numerical failure contract and
NVCC/CuMetal portability. Repeated shell centers accumulate into their shared
atom before the final unique atom is recovered by translation invariance.

This changes LR accumulation order. Scientific acceptance requires every
complete energy/force observation at 1e-8 Eh / 1e-7 Eh/Bohr and independent
CPU displaced-ERI tests. An all-center primitive may have more register/local
storage than a single-center Dual3 evaluator; fewer moment ladders alone do
not prove a faster complete endpoint.

## Evidence and retained identities

The isolated experiment starts from #1756 production commit
`ac8e14728e389d4a8791da9a603260455feccb81`, ultimately based on master
`d35ae539f645cb5e8b9c0c2fb7a426c5e2ad08e8`. That qualified baseline library is
`e51239008c378cf61e4703abd5e287825beb131bb192f1aea7a444a7a449f5bd`.
It does not include the source-screening or force-storage proposals. Existing
live benchmarks continue using frozen source/binary copies in their own roots.
Later master integration must retain its own source and binary provenance.

A host harness checks 19,440 center/axis derivatives across all ten pair-order
partitions of total orders 4--6, nine Cartesian orientations, three exponent
scales and omega 0, 1e-8, 0.3, 2 and 1e4. Its five-point displaced-value Hermite
contraction differs from the Wick gradient recurrence; maximum observed LR
absolute error is 2.01e-10. It also checks full-range derivatives, translation
balance and nonfinite propagation for invalid omega. This shared-moment
control is not a replacement for the independent native CPU oracle.

The native four-center gate is extended with d/p/p/s, d/d/p/s and d/d/p/p
fixtures, including repeated-center bindings and both spin contractions.
Those independent CPU displaced-ERI gates and complete molecular tests are
pending the candidate CUDA build. Artifact paths are ignored under
`.artifacts/range-all-center/`; no GPU speedup has yet been measured.

## Promotion

Require finite Slurm numerical/sanitizer qualification of the exact binary,
controlled complete cold/priming/three-warm endpoints, moved geometry and a
larger system. Retain and reject this direction if local storage or schedule
cost eliminates the anticipated reuse benefit. Do not infer actual executed
quartet counts from logical dense capacity.

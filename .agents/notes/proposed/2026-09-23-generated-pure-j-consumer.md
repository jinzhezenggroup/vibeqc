# Candidate: reuse the compiler's shell source for pure Coulomb

Status: proposed (implementation present; GPU qualification pending)
Date: 2026-09-23

## Source of the slowdown

Native semilocal CUDA KS used the generic public-AO independent J/K kernel.
At 48 AOs its J work consumed 5.34 of 5.77 seconds in a two-Fock warm endpoint,
even though generated shell sources already accelerate direct HF. Optimizing
XC cannot remove this dominant recurrence cost.

## Candidate implementation

The canonical compiler scatter now has a typed Coulomb consumer. It retains
the same ERI permutations and J equation and suppresses only K scatter. The
native provider prepares bounded shell-pair topology, cached primitive pairs,
and Cartesian/public transforms, then borrows existing AOT class schedules.
It never constructs an all-quartet table or subtracts HF matrices to recover J.

The generated value task ABI grows from 192 to 200 bytes. Every producer sets
the consumer explicitly; value-initialized HF stream aggregates select zero
(HartreeFock). Shape-sensitive page capacity tests use the charged byte cap and
the current ABI size. Qualify the HF performance consequence too.

Generated classes cover spd except dddd; dddd keeps the established exact
native integral consumer with the same compiler-emitted scatter. Unavailable
classes, explicit small budgets and optional allocation failure retain the
generic provider. Derivative and mixed-precision paths are unchanged. The
pure-J stream keeps geometry-only Schwarz screening; HF's density tails are
bypassed by consumer identity, not fabricated density metadata.

Runtime owns resources and stream lifetime; compiler owns integral equations,
scatter, ABI emission and existing schedules. KS resource queries charge a
conservative shape-only generated capacity plus the generic fallback. The
owner records actual device/preparation bytes and releases borrowers before
their stream or geometry. The public input may be nonsymmetric: its
antisymmetric component cancels under ERI pair symmetry.

## Acceptance

Completed: compiler/SCF ownership checks, host C++ syntax checks, and 220
independent dense contractions covering J/HF, RKS/UKS and all coincident index
patterns with nonsymmetric densities. CUDA tests add through-d generated
admission, through-f fallback, exact-budget fallback, spin sums and independent
full-ERI matrices. Device execution and complete KS/HF endpoint qualification
remain required before promoting this candidate. See #1077.

## Allocated GPU update

The through-d J matrix, nonsymmetric spin-density, through-f fallback and exact
budget fallback tests passed on RTX 5090. Integration library
`b3e08e5a556b1f8abbd6c72e7109ca92a135f9626ac121bc81c687bb3390916b`
also includes the existing #1073 and #1076 implementations. PBE on the unchanged
explicit grid passed complete cold/priming/two-repeat GPU4PySCF comparisons:
3 atoms reached 0.264–0.274 s warm, and 6 atoms reached 0.630–0.631 s warm,
both with two native iterations. Maximum energy errors across all four pairs
were 1.33e-12 and 2.36e-12 Hartree, respectively.

The 24-atom cold endpoint reached its 60-second deadline and was stopped.
A bounded profile exposed a separate ordinary-KS eigensolver problem (#1090,
PR #1091): one completed graph-native Jacobi call took 9.2 seconds. This is
not evidence that 96-atom endpoint qualification is complete. Keep the larger
gates and HF ABI performance check pending.

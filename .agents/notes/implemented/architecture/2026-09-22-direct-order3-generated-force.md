# Decision: retire handwritten Direct-HF order-three force mathematics

Status: implemented

## Context

Direct-HF total-angular-order-three force evaluation retained two independent
handwritten scientific owners: the complete contracted coordinate derivative in
`direct_native_order3_gradient.cuh` and the subset/Wick pair-gradient helper in
`direct_native_pair_order3_gradient.cuh`. The compiler already owns the
four-center ERI graph, first nuclear derivatives, force-only output pruning and
pressure-aware algebra ordering through Weighted IntegralIR.

Keeping the native formulas duplicated the same PPPS/DSPS/DPSS/FSSS
mathematics across generated and handwritten paths. It also left bounded
streaming with a hidden order-three instantiation even after the fixed queue was
migrated.

## Decision

- Emit force-only Weighted IntegralIR helpers for the four canonical
  total-order-three classes: PPPS, DSPS, DPSS and FSSS.
- Keep queue ownership, Schwarz screening, density folding, primitive-pair
  storage and atom-force scatter in the native Direct runtime.
- Precontract the complete shell-task component weights before invoking the
  generated mathematical helper. This preserves the existing exact compact
  queue while avoiding a second ERI/gradient recurrence.
- Use the same generated-math adapter for fixed and bounded/paged Direct force
  execution. Bounded streaming no longer instantiates the generic handwritten
  order-three quartet derivative.
- Preserve standalone generated-AOT ownership as a disjoint mask: a class
  already owned by the selected AOT force registry is skipped by the native
  generated-math fallback.
- Recover the fourth-center derivative from translation invariance, as in the
  existing generated force-only contract.
- Delete both handwritten order-three scientific formula owners.

The compiler is the single mathematical source of truth for order-three force
science; native code remains an execution/scheduling adapter rather than a
second scientific implementation.

## Validation

- Weighted IntegralIR parity includes PPPS, DSPS, DPSS and FSSS against the
  independent raw component-derivative construction.
- Focused weighted/codegen tests: 31 passed.
- SCF structure check: 217 modules, 0 dependency errors.
- CUDA 12.9.86 / sm_120 compilation covers both
  `direct_angular_force.cu` and `direct_bounded_fallback.cu`; the Direct
  native static library device-links successfully.
- Full-library CUDA link and explicit fixed/bounded GPU endpoint qualification
  are recorded in the corresponding PR once complete.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Final qualification addendum

Post-rebase focused generated-math/codegen tests: 32 passed. SCF structure tests: 106 passed; the structure inventory reports 217 modules and 0 dependency errors. The complete CUDA 12.9.86 / sm_120 vibeqc target links successfully on the rebased branch, including direct_angular_force.cu and direct_bounded_fallback.cu through the generated weighted-ERI header.

RTX 5090 endpoint qualification explicitly narrowed force AOT ownership to ssss, forcing every total-order-three class through this fallback. RHF/UHF spd exact-versus-bounded moved-geometry replay plus an FSSS fixed/bounded CPU-oracle replay all passed: 3 tests in 38.09 s.

This establishes correctness and route coverage. A separate matched Release endpoint/resource timing gate remains required before any performance claim or non-draft promotion.

Agent: ChatGPT
Model: GPT-5.6 Sol

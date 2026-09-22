# Decision: qualify only the one-root 000 Rys lowering

Status: implemented
Date: 2026-09-16

## Problem

#394 proposes seven low-angular derivative classes, but their aggregate coverage
does not establish a useful endpoint saving. The user requires a complete 000
comparison before considering 100/001, then 110/101, then 200/002.

## Decision

Generate only SSS first derivatives from the existing one-axis Gaussian-moment
IR. Degree-one moments require one quadrature node, `F1(T)/F0(T)`, with weight
`F0(T)`. An independently owned analytic evaluator uses fixed Taylor polynomials
at small T, erf/exp in the middle, and analytic asymptotics for T >= 40.
The omitted relative F1 tail above that boundary is below 4e-17, beneath
FP64 epsilon; the independent grid covers both neighboring boundary values.
Six raised orbital-center moments contract directly with the existing response weights; the existing finish function
recovers auxiliary-center translation. No runtime differentiation is introduced.

Keep polynomial geometry emission byte-identical and reuse the existing shell
scheduler, signature packets, normalization, symmetry folding, and atom output.
The only scientific change is the primitive lowering selected by
`VIBEQC_DF_SHELL_MATH_000=polynomial|rys|auto`. Automatic selection remains
polynomial. The qualified comparison is the explicit 000 path on the retained
384/768 AO warm endpoints; broader default-policy qualification is separate.
The higher-class prototype in a separate experiment is not part of this change.

## Invariants

- FP64, screening, metric, convergence and force tolerances stay fixed.
- Primitive products, active Cartesian components and response panels must match.
- Generated IR owns equations; native templates own dispatch and scheduling.
- Root tests use an independent high-precision incomplete-gamma reference.
- Primitive contractions additionally use independent libcint derivatives.
- Clean endpoint repeats and intrusive work/resource diagnostics remain separate.
- Do not broaden the angular domain as a response to a weak 000 endpoint result.

## Evidence

Five interleaved complete energy-and-force warm samples per arm, with identical
frozen input D and exactly three SCF updates, give:

| AOs | Polynomial (s) | 000 Rys (s) | Reduction |
| ---: | ---: | ---: | ---: |
| 384 | 0.922749193 | 0.907209149 | 1.68% |
| 768 | 4.059822947 | 3.984740416 | 1.85% |

Both arms use merged #399's automatic exchange controls. At 384 this means the
normal dense-SCF policy, not the occupied-SCF ablation used in #399. This is a
modest complete-endpoint saving, even though the changed kernel is much faster.
Separate Nsight activity captures measure 000 at 24.121 → 8.277 ms (2.91×)
for 384 and 133.505 → 54.971 ms (2.43×) for 768. Summed three-center shell GPU
time improves by 6.40% and 5.03%, respectively. Captured transfer counts and
sampled warm device-memory peaks are unchanged between policies.
The measured numerical difference between paired forces stays below `1.14e-13`
Ha/bohr; independent complete force errors stay below `1.60e-10` Ha/bohr.

The work ledger reconstructs exactly equal shell/signature domains. At 768 both
arms retain 28,385,280 shell visits, 175,132,672 primitive products,
973,859,328 active Cartesian component products and 13 derivative panels. Only
the 55,656,960 active 000 primitive products switch to Rys.

Resource diagnostics report 132 → 107 registers/thread and 8960 → 6656 shared
bytes for 000, raising the theoretical occupancy limit from 25% to 33.3%.
The production sm_120 library grows by 413,616 bytes (0.201%); native scientific
CUDA grows by zero code lines, runtime CUDA by 42.

Root and emitted-arithmetic tests cover small, boundary, randomized and extreme
finite arguments through `1e300`, with independent 75-digit incomplete-gamma
moments. A mixed-primitive SSS native fixture includes 1–7 primitives and enough
signatures to cross packet flush boundaries. Independent libcint contractions
and the actual generated CUDA evaluator pass. Forced-Rys regression passes 140
cases, including response layouts, RHF/UHF, spherical/Cartesian, geometry replay,
batches and constrained resources; native memcheck reports zero errors.

Clean endpoints, basic component traces, detailed work atomics and uninstrumented
kernel captures are separate measurements. The retained evidence sums every
repeated derivative panel. It does not combine host/GPU inclusive intervals or
claim achieved hardware occupancy.

## Rejected alternatives

Mechanical unrolling of the current coefficient-convolution loops previously
regressed the complete endpoint. Wholesale reuse of another engine's evaluator
would introduce separate ownership and provenance problems. Building all seven
classes before the first measurement would obscure the user's stop criterion.

## Revisit when

Only a material kernel saving supported by complete 384/768 AO endpoint evidence
justifies promoting 000 or starting the next explicitly authorized class slice.

## Consequences

Keep the generated 000 path as an explicitly selectable, numerically checked
alternative. The endpoint result supports considering 100/001 next; it does not
justify implementing all remaining classes at once or changing an unrelated
schedule. This change does not promote a new automatic policy.

## References

- The SSS-only runtime override was subsequently retired by the
  [selector retirement decision](../compatibility/2026-09-16-df-math-selector-retirement.md);
  the measurements and source-specific commands here remain historical evidence.
- Issue #394; supersedes its original seven-class implementation scope for this slice.
- Merged #399 / PR #402: `a90973d7aa43776348e0cdcb51927db01bab5cb4`.
- [Retained endpoint/work/kernel evidence](../../../../benchmarks/results/issue394-000-rys/README.md).
- [Current shell derivative contract](../../../../docs/developer/df_shell_derivatives.md).

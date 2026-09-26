# Auxiliary-f Rys derivatives for practical JKFIT

## Problem

Practical cc-pVDZ/cc-pVDZ-JKFIT has far more auxiliary than orbital functions
and includes auxiliary f shells. General work-based admission (#486) removes
molecule/shape fingerprints, but it cannot select mathematics the compiler has
not generated. After the existing s/p/d manifest is admitted, five auxiliary-f
classes still spend substantial complete-force time in polynomial evaluation.

## Decision

Reuse the existing cooperative shared-axis moment IR for `003`, `103`, `113`,
`203`, and `213`, with three/three/four/four/four roots respectively. Promote
these five entries in the independently qualified sm_120 manifest. The old 18
s/p/d mappings, FP64 arithmetic, Gaussian normalization, translation identity,
response formula, metric cutoff/rank and physical final-state rules stay intact.

`223` needs five roots. The strict DF root evaluator does not qualify that order;
retain the polynomial fallback rather than silently reuse a less accurate
quadrature. Other devices/classes remain explicitly unqualified. This is a
bounded capability extension, not a new runtime algorithm or handwritten CUDA
scientific implementation.

## Alternatives and measurement

Simply deleting dimension/histogram guards cannot fix missing f mathematics.
That general admission work is reused from #486 instead of duplicated here.
Screening (#437), approximate metric response, mixed precision and changing the
auxiliary basis are not substitutes for the exact capability extension.

First qualify the math with one binary embedding the old qualified manifest and
the new unqualified candidate. Both arms use the same frozen density and the
same shell/packet consumer. On 96/464, seven interleaved complete energy+force
pairs reduce wall time by 26.23% with angular grouping or 27.20% with packets.
On 192/928 the corresponding reductions are 18.48% and 19.19%. These are not
whole-default-path or GPU4PySCF comparisons. The complete raw measurements and
separate default-path qualification are indexed by the
[evidence README](../../../../benchmarks/results/issue435-jkfit-rys/README.md).

The independent gates remain 3e-11 Eh and 3e-11 Eh/Bohr. The initial practical
campaign's largest errors are 2.28e-12 Eh and 1.56e-12 Eh/Bohr. Explicit counts
preserve 202272 / 1606320 shell visits and 1464752 / 11528520 primitive products
for 96/464 and 192/928 respectively. Operator calls match. Equal 96/192/384/768
controls are neutral within 0.6%; no speed claim is made for these controls.

## Validation and ownership

Host checks cover all 23 generated classes against independent libcint and
high-precision center differentiation, including general contractions,
Cartesian/spherical normalization, coincident centers and extreme arguments.
The 32-test CUDA practical suite covers RHF/UHF, moved geometry, packed/dense
response and budgets. The f-class trace asserts actual Rys execution, not just
agreement between two fallbacks. Memcheck, racecheck and synccheck report no
errors for the bounded practical OH packet case.

The work-ledger reducer now reconstructs separate orbital and auxiliary shell
domains. Shared recurrence work is counted once per active primitive, before
checking the remaining component work. Detailed counters are intrusive and
never mixed into clean endpoint medians; the existing paired runner now also
accepts exact explicit bases and fresh independent references.

## Final default integration

The combined default build passes 40 GPU regressions. Seven interleaved clean
triples compare pinned post-#480, #486-only and combined default execution under
one protocol, with identical frozen densities and no route overrides. Complete
96/464 wall time changes 400.070 -> 116.221 -> 86.455 ms; 192/928 changes
341.615 -> 303.465 -> 248.065 ms. All paired differences improve, relative MAD
stays below 0.60%, and branch/operator work matches. The incremental f-only
benefit on the actual admitted defaults is 25.61% and 18.26%. These timings are
not a comparison against the old multi-second issue-opening benchmark.

The final angular-only trace uses a p0_0_0 group sentinel. Its postprocessor now
collapses host primitive signatures into that angular group and preserves their
exact task-cost distribution. Sparse groups have explicit tight active-work
bounds; dense groups retain exact work. This reporting-only correction was
validated against both default traces and six additional host cases.

## Consequences and revisit triggers

Default execution combines the general workload policy with the qualified
class manifest. Neither a water fingerprint nor equal orbital/auxiliary size is
needed for these classes. Keep the unqualified five-root fallback until a
strict independent quadrature implementation and a complete-endpoint benefit
justify it. Larger practical systems, other devices and future screening need
their own evidence; warm replay does not establish cold/changed-geometry speed.

Agent: ChatGPT
Model: GPT-6 Astra Pro

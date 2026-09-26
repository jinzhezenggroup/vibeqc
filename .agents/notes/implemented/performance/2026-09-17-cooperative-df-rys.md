# Decision: share cooperative DF Rys moments across components

Status: implemented
Date: 2026-09-17

## Problem

The post-#415 768-AO endpoint spends about 1.28 s in shell derivatives.
Eleven missing three/four-root classes account for about 643 ms; the old
component-local two-root lowering also loses for 101/110. The retained 2.6%
low-angular endpoint improvement remains production behavior.

## Decision and invariants

Reuse the compiler-owned Rys tables and shared moment/recurrence IR to prepare
bounded root/axis state once per primitive triple, shared by component lanes.
Keep packet scheduling, public weight folding, translation recovery, precision,
screening, metric cutoff/rank, force definition and SCF work unchanged. Preserve
the polynomial fallback and the already promoted low-angular mapping.

No candidate is promoted merely because mathematical generation is available.
First audit nodes, weights and defining Boys moments against an independent
high-precision incomplete-gamma/Jacobi construction, with a 5e-14 relative gate.
Then qualify primitive and contracted derivatives against libcint, including
Cartesian/spherical, packed/symmetric/full weights, coincident centers and
same-atom translation semantics. Real GPU validation uses finite Slurm jobs.

## Predeclared performance decision

Prioritize 111/210/211/201/102, then 112/212/202/221/220/222; recheck 101/110.
Compile bounded candidate objects before one combined library. Retain source,
object and library sizes, register/shared/stack/spill resources, occupancy,
root/recurrence counts and primitive/shell/component work. No builds or intrusive
profiling overlap clean timing, and every repeat or failure is retained.

The combined 768-AO candidate needs at least five interleaved clean pairs from
identical frozen density with matching actual SCF/J/K/eigensolve work. Continue
only for >=0.20 s complete warm energy+force improvement, or >=15% reduction of
the complete shell-derivative component with a clean endpoint win. Also require
384-AO control and 96/192 correctness. A correct candidate failing these gates
is a retained negative; an uncompiled or inaccurate candidate is not a timing
negative. These gates apply to #418 and do not revisit the accepted #415 gain.

## Rejected alternatives

Merely adding higher angular classes to component-local moment emission does
not provide the sharing required here. Copying external scientific CUDA or
inventing a separate handwritten per-class recurrence is out of scope.

## Evidence

Baseline: f5eb8dfb696ce4304c5151670c55fec78fdddb7d (upstream master).

The first 85-digit audit of the existing three/four-root tables, after pairing
nodes and weights by node order, failed 158/969 and 188/969 arguments at the
5e-14 gate. Worst relative weight errors were 2.95e-12 and 4.69e-12; moments
reached 2.78e-12 and 4.16e-12. Nodes passed. A first order-sensitive audit and a
missing-dependency launch are retained alongside the corrected audit.

The strict option reuses the common evaluator, 2.5-wide interval layout and
small/large branches. Only interpolation coefficients are regenerated at 90
digits with degree 17. Existing default Direct/value coefficients are unchanged;
the validated one/two-root DF arithmetic is unchanged. A scaled Hankel/Cholesky
oracle, independent of the generator's Stieltjes construction, passes all 969
samples for each order. The maximum node/weight relative errors are 2.66e-14 and
3.11e-14 (the retained tiny-argument linear branch). Host emission tests pass,
as do all four root orders on the GPU in Slurm job 9891.

The cooperative lowering interns the shared Gaussian moment IR in one root/axis
graph, using the existing raised-A/raised-B identity to bound the rectangular
cache. No component emits its own recurrence. All 180 host derivative tests pass
against libcint and high-precision center differentiation. The initial expansion
of the old tests failed two extreme T=1e300 cases (211/212): the recovered C
derivative underflows while nonzero A/B values cancel, leaving an FP64 addition
residual. New high-angular recovered-center checks propagate the independently
measured A/B errors plus the addition rounding bound. Every original low-angular
gate and all ordinary force/endpoint tolerances remain unchanged. Both test runs
are retained.

The first candidate objects preserve the compact schedule. Predeclared resource
limits are 255 registers/thread, 48 KiB shared/block, 128 bytes stack/thread,
zero spills and 2 MiB per object. The stack allowance covers the bounded
three/four-root adapter scratch. Objects for 111/210/211/201/102 and controls
101/110 compile within these bounds; three-root candidates use 48 bytes stack
and zero spills, with fewer registers than their polynomial controls. The full
26-object set also passes. Slurm 9894 completes every derivative holdout and all
four sanitizer modes for both object groups with zero errors or race warnings.

Slurm 9895 retains 252 class/signature records with every numerical and semantic
work gate satisfied. Maximum derivative error is 5.075e-17. All 13 classes win on
both 384 and 768 workload distributions: sums of per-signature medians are
157.557/60.726 ms and 1184.631/467.398 ms for polynomial/cooperative Rys. These
isolated dense-weight class timings exclude SCF, response production and the
other five classes. They select one combined candidate; they are not an endpoint
claim. The experimental manifest keeps the qualified old mapping for automatic
execution and exposes the new mapping only through the existing candidate arm.
Slurm 9896 passes all 48 native tests; Slurm 9897 passes all 80 clean endpoint
samples and 16 separate diagnostics. Complete energy+force medians are
0.108953/0.109003 s (96), 0.164843/0.130135 s (192), 0.694017/0.622889 s (384),
and 2.557756/2.000895 s (768), baseline/candidate. The 768 saving is 0.556861 s
(21.77%), passing the predeclared gate. Its complete shell derivative component
also falls 44.52%, from 1192.641 to 661.702 ms, in separate intrusive replays.
Energy-only medians remain effectively unchanged at 384/768.

Every numerical, frozen-density, graph replay/finalization, metric/rank and
semantic shell/primitive/component/response/transfer gate passes. Solve epochs
advance between calls, and each occupied response factor is independently bound
to its own final determinant. The binary grows 1.60%; charged memory and sampled
device residency agree between campaign arms. The independent baseline and final
production checks below preserve this qualification; the warm campaign makes no
stock GPU4PySCF claim and keeps #206 open.

Production admission retains the existing sm_120, 384/768-AO equal-auxiliary
boundary. The positive 192-AO candidate timing is not a default-policy promotion;
small/unequal-auxiliary/other-architecture automatic fallbacks remain unchanged.
Slurm 9898 subsequently verifies independent old-library baseline equivalence:
zero energy difference, maximum force difference below 1e-13, identical work and
charged/process device memory at 384/768. Slurm 9899 passes 16 integrated shell
and response holdouts; two auxiliary-only-center tests initially skip for a
missing suite-specific opt-in and then pass in Slurm 9900. Both attempts remain
retained. The qualified manifest removes the embedded campaign baseline.

Slurm 9901 passes all 48 native tests on the final library and checks all four
automatic force endpoints with separate diagnostics. At 384/768 the final default
matches the campaign candidate; at 96/192 it matches the original auto fallback.
Energy differences are zero and force differences remain below 1e-13. Selected
classes, resources, semantic work, final residuals, metric and charged memory
agree. The final library is 209,740,176 bytes and only its generated selection
header/build identity differ; the generated mathematics is byte-identical to the
campaign. Final single replays never replace the original five paired samples.

The initial PR Python CI passed 3652 tests but found a stale versioned CUDA
ownership snapshot. Regenerating it from the maintained source and matching
generated build fixes the 12-test ownership suite; the original CI failure is
retained with the campaign evidence.

`benchmarks/results/issue418-cooperative-rys/` retains the independently checked
records, failures, resource reports, workloads and reproduction drivers.

## References

- #206, #418; #419 remains a separate response-dataflow experiment.
- `benchmarks/results/issue404-combined-rys/768-forces.json`
- `python/vibeqc_compiler/integral/rys.py`
- `python/vibeqc_compiler/integral/df_rys_shell.py`

## Subsequent admission qualification

The [2026-09-19 admission decision](2026-09-19-df-rys-admission.md) supersedes
only the automatic dimension/equal-auxiliary boundary after a separate current-
source qualification. The measurements and original decision above remain
historical evidence; the new work does not count the 384/768 speedups again.

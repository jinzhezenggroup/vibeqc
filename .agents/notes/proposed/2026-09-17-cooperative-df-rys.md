# Decision: qualify cooperative three/four-root DF derivatives

Status: proposed
Date: 2026-09-17

## Problem

The post-#415 768-AO endpoint spends about 1.28 s in shell derivatives.
Eleven missing three/four-root classes account for about 643 ms; the old
component-local two-root lowering also loses for 101/110. The retained 2.6%
low-angular endpoint improvement remains production behavior.

## Proposed decision and invariants

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
Complete endpoint qualification remains pending.

`benchmarks/results/issue418-cooperative-rys/` retains the independently checked
records, failures, resource reports, workloads and reproduction drivers.

## References

- #206, #418; #419 remains a separate response-dataflow experiment.
- `benchmarks/results/issue404-combined-rys/768-forces.json`
- `python/vibeqc_compiler/integral/rys.py`
- `python/vibeqc_compiler/integral/df_rys_shell.py`

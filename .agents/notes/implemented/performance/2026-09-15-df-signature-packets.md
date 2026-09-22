# Decision: batch homogeneous DF derivative signatures in bounded packets

Status: implemented
Date: 2026-09-15

## Problem

After #383 (`422796f04553953a65e45bf1f0e41642f482a591`), the retained
768-AO warm energy-and-force endpoint took about 7.31 s, including 2.73 s of
three-center derivative kernels. The inherited #385 prototype grouped shells
by angular momentum and primitive count, but had incomplete launch contracts
and incorrectly made full-mode same-signature products triangular.

Exact reconstruction of every baseline launch grid and native work counters
found 3,631,488 / 28,385,280 shell tasks and 22,475,520 / 175,132,672 primitive
products at 384 / 768 AOs. The loop-count model gives approximately 46% warp
primitive-slot efficiency. That number is a padding model, not a measured stall
fraction or a predicted speedup. Nsight Compute hardware counters were denied
by `ERR_NVGPUCTRPERM`; no driver permissions were changed.

## Decision

Retain homogeneous runtime primitive bounds and batch up to 24 product ranges
per angular-class kernel. A range stores shell slices and a block prefix, not
individual triples. Each block selects one signature. Ranges are ordered by
descending primitive product within each packet so expensive small slices can
overlap cheaper slices. Both individual and packet kernels call the same
contraction helper and unchanged compiler-generated derivative implementation.

The packet payload is 2,888 bytes on the qualified build. All kernel arguments
fit below 4 KiB. `grid_constant` prevents address-taking from creating a private
multi-KiB packet copy in every thread. Packets flush at their fixed descriptor
capacity or the existing grid limit. There is no device queue allocation,
count/scatter pass, persistent worker, additional synchronization, or resident
metadata proportional to the triple domain.

Automatic selection is restricted to RTX 5090, full compact shell execution,
one density term, and the measured spherical orbital/auxiliary signature
histograms. It selects the dense symmetric 384-AO response and the existing
qualified packed occupied 768-AO response. Other profiles keep the old traversal.
Explicit `off`, `on`, and `packet` remain diagnostic comparisons; `auto` is the
default and all explicit controls participate in prepared replay provenance.

## Rejected alternatives

- Simple s/p compile-time `Math::accumulate()` unrolling was already measured
  by the issue author at approximately 7.31 → 8.48 s. It was not repeated or
  incorporated here; the issue's raw samples and provenance are retained.
- Individual homogeneous launches improve the qualified 768-AO endpoint
  (7.310 → 6.973 s in five paired samples), but regress 384 AOs
  (2.835 → 2.994 s). At 384 AOs, fragmentation expands 144 launches to 987 and
  kernel time from 537 to 696 ms. Do not enable that route universally.
- A per-triple compact queue is unnecessary for this improvement and would
  introduce unbounded cubic resident metadata. No Direct J/K Schwarz screening
  or derivative equations were copied.

## Invariants

Packed response algebra, primitive precision, metric thresholds, auxiliary
bases, force definitions and generated equations remain unchanged. Full mode
visits both orbital orientations, including same-signature rectangles. Only
symmetric/packed same-signature products are triangular. Shell order within a
signature preserves the public AO offsets used to clip auxiliary panels.
RHF/UHF, Cartesian/spherical, generic, dense, partial-shell and corrected-state
fallback semantics remain covered by oracle tests.

Dynamic trace labels must own their storage until deferred emission. Resource
attributes must be maxima within an operation, rather than sums over repeated
launches. The early uniform prototype's retained resource counters require
normalization by launch count; the final `shell_resource_values_are_maxima`
marker distinguishes the corrected representation.

## Evidence

The [retained evidence](../../../../benchmarks/results/issue385-df-signatures/README.md)
contains exact identities, raw samples, per-launch profiles, full histograms,
resource data and reproduction sources. All real GPU work used finite Slurm
`main` allocations with `--gres=gpu:5090:1` and unchanged assigned visibility.

The packet ablation's five clean paired medians are 2.836 → 2.653 s at 384 AOs
and 7.359 → 6.445 s at 768 AOs. Separate Nsight measurements give
536.791 → 344.402 ms and 2732.622 → 1810.546 ms, respectively, with the original
144 / 234 launches and unchanged primitive work. The component saving accounts
for the complete endpoint saving. Every sample passes the unchanged independent
1e-9 Ha / 1e-8 Ha/Bohr gates.

Additional shell ID uploads are 1,536 / 3,072 bytes. Packet parameter payloads
sum to 415,872 / 675,792 bytes per force call, reported separately from explicit
H2D traffic. Diagnostic packet preparation is approximately 0.91 / 1.28 ms,
excluding submission but including tracing overhead. No exact achieved occupancy
or stall attribution is claimed. CUDA occupancy APIs and Nsight launch resources
provide theoretical resource limits; every standalone occupancy calculation was
cross-checked against the native API results.

### Final automatic-selection confirmation

Five final clean samples confirm 2.837378 → 2.643900 s at 384 AOs and
7.313037 → 6.428660 s at 768 AOs against the refreshed exact #383 binary.
The simultaneous final off/auto ablation gives 2.830143 → 2.643900 s and
7.345499 → 6.428660 s. All those endpoints take three SCF iterations and
pass the independent DF gates. The packet-profile and final derivative kernel
objects are byte-identical. Small 96/192-AO off/auto controls remain unchanged
within run variation and pass their stricter retained single-system gates.

The original Direct96 batch-1/4 external force gates fail on both exact #383
and this candidate; Direct192 batch-1/4 passes. Native Direct96 baseline and
candidate forces agree within 7.41e-13 Ha/Bohr, with comparable timings.
Tightening only the GPU4 reference convergence does not satisfy the original
force gate, and its batch-4 diagnostic also reports nonconvergence. These
results are retained as an existing open #206 requirement, not relabeled as
passing Direct qualification. No numerical gate or Direct equation changed.

The user also requested the same benchmark procedure as Direct J/K. Additional
DF measurements therefore use the unmodified Direct comparison runner, including
ABBA engine interleaving, fixed engine-local post-cold densities, untimed priming,
synchronized endpoints and retained iteration branches. The evidence README
records that matrix separately from the native policy ablations and standalone
force probes. Procedure identity does not imply identical stopping algorithms
or matching observed SCF branches.

The shared runner confirms 2.847730 → 2.647974 s at 384 AOs and
7.342726 → 6.431249 s at 768 AOs, with five samples and three native SCF
iterations. No large-case GPU4 branch overlaps those native iterations.
Its original DF96 batch-4 force gate also fails on #383 and the candidate
(3.20914e-11 / 3.20874e-11 versus 3e-11); paired native forces agree within
5.51e-14. Keep this existing small-matrix gate open under #206 alongside the
Direct96 failures; do not relax thresholds to turn the audit green.

## Consequences

This removes a measured part of the remaining derivative gap. It does not
establish GPU4PySCF endpoint parity or close #206. Fresh GPU4PySCF force-only
samples are compared with clearly scoped native force-stage diagnostics;
unmatched SCF branches are not combined into a headline median.

The extra compiled packet family increases the frozen shared library from
about 287 to 352 MiB and adds about 24 MiB of loaded process residency when first
used. These costs are retained alongside bounded queue bytes and sampled peaks.
No scientific code ownership is transferred from the compiler to the runtime.
The conservative ownership inventory nevertheless grows by 147 scientific-glue
lines in the whole-file bridge classification and 281 runtime lines. Keep those
counts visible; they are layout/selection/dispatch growth, not new equations or
a reason to relabel the existing bridge as runtime.

### NVCC fast-compile compatibility

The CUDA CI configuration enables NVCC 12.9 `--Ofast-compile=max`, while all
retained performance measurements use the ordinary Release optimizer. CI and
a local reproduction found that both max and min fast levels exceed the 48-KiB static shared
limit for all 384 derivative entrypoints, including sss, after sharing the
contraction helper between individual and packet kernels. The ordinary optimizer
produces the measured per-class shared footprints and compiles successfully.

Keep the fast-build target option for other units, but override this source
with `--Ofast-compile=0`. This is a build compatibility exception; it does not
alter the measured Release options, CUDA source, generated equations or dispatch.
The ordinary Release build and oracle/endpoint qualification above already
validate this optimization mode. The generated fast-build CMake command is
checked to contain the source override after the target's max option. Do not
use fast-build outputs for production performance/resource claims.

## Revisit when

New shell histograms, devices or response layouts have complete endpoint and
memory evidence; hardware counters become available; or a bounded persistent
schedule saves enough additional endpoint time to justify its construction and
ownership costs. Do not infer these wins from loop unrolling or kernel-only
measurements.

## References

- [#385](https://github.com/jinzhezenggroup/vibeqc/issues/385)
- [#382](https://github.com/jinzhezenggroup/vibeqc/issues/382) and
  [#383](https://github.com/jinzhezenggroup/vibeqc/pull/383)
- [#206](https://github.com/jinzhezenggroup/vibeqc/issues/206)
- [Current execution contract](../../../../docs/developer/df_shell_derivatives.md)

## Admission superseded — 2026-09-19

The execution design remains unchanged, but the endpoint-specific automatic
admission is superseded by the
[shared derivative work profile](2026-09-19-df-work-admission.md). Historical
measurements above remain evidence for their original source and workload.

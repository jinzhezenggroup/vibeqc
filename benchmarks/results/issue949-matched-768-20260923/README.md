# Issue #949: matched 768-AO DF response qualification

This directory contains a bounded protocol for the remaining complete
energy-plus-force comparison. It is not a completed qualification receipt until
the raw output, exit sentinel, device and library identity, numerical gates,
work counts, and both policies have been audited.

The historical RTX 5090 campaign in
`../acceptance-closeout-20260922/df-768-panel-ablation.json.gz` completed its
strict independent PySCF 2.14.0 CPU oracle and one BLAS sample, then reached a
1200-second command limit with zero completed scalar samples. That GPU timing is
not pooled with measurements from another device. The adapter here reuses only
the CPU oracle after checking geometry, charge, spin, bases, AO/auxiliary counts,
and solver thresholds. The active runner checks cold, prime, and measured
energy/force values against the original `1e-9 Eh` and `1e-8 Eh/Bohr` gates.

`run_qz.sh` takes `probe`, `build`, or `measure` as its sole argument. Every Job
sets absolute `VIBEQC_SOURCE_DIR` and `VIBEQC_RUN_DIR`, exact
`VIBEQC_SOURCE_SHA`, and its actual Job name as
`VIBEQC_BENCHMARK_ALLOCATION`. The clean, pinned source is built with Release
settings for `sm_90` with the explicitly supported `portable_cuda` AOT profile;
the binary SHA-256 is printed and the build exit is
retained in the run directory. `measure` refuses a missing or failed build.
Its runner stores a phase marker before each prime and timed endpoint so that
an external time limit leaves a named incomplete stage and a lower bound,
rather than an apparent zero result. The output and frozen density checkpoint
are raw artifacts outside the reviewed `benchmarks/results/` tree.

The matched arms use one prepared calculator, geometry, basis, native library,
GPU allocation, frozen post-cold density, panel storage, FP64 native endpoint,
thresholds, and six expected SCF updates. Each arm is primed before its clean
sample. The only requested arm control is
`VIBEQC_DF_RESPONSE_ALGEBRA=blas|scalar`; the cold solve uses BLAS. This control
selects response algebra beyond the charge dot as well. Any complete-endpoint
ratio is therefore a **response-algebra ablation**, not a charge-only causal
speedup. A completed scalar sample and separate counters are needed before
claiming the #949 performance acceptance item. A timeout remains a timeout
lower bound, not a measured speedup or a reason to close #949.

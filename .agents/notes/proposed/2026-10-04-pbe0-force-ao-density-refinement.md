# Proposal: reduce admitted high-order force recurrence work at AO granularity

Status: experimental; not qualified for default promotion
Date: 2026-10-04

## Problem

The shared PBE0 force provider still has a large full-range derivative cost.
Schwarz-live shell paging improves dispatch, but the corrected 96-atom census
admits the same 92,233,228 shell quartets with paging OFF and ON. A separately
tested weighted DPPP root specialization lost on the complete integral-source
replay despite a small isolated-class win; do not restore that rejected code.
Order-five already uses an explicit generated Cartesian gradient, not a generic
Dual3 implementation that can simply be replaced again.

## Admission census

Slurm job 5624 used unmodified master 9c54107caa03f20d05e756ca7e3cd66fb13dabcf
and the previously independently qualified 96-atom density. The diagnostic
reuses the common shell gate, AO decoders and separate J/K density coefficients.
Every shell-class admission/capacity agrees with the earlier corrected census.

- Source identity: `989f486f1ddf4308eb888ec6f6d90497c77a88b32f4c5ef3cbece83ad7a53dc8`.
- Library SHA256: `73c54dc18529365e25b9d4bfb70717002170d37932becffe5193e5548f530e78`.
- Density input SHA256: `a9a669914ca2b75be5b6ad74ba168aa4f62cf61b818b274d1a491fae376e9775`.
- Diagnostic binary SHA256: `0fcc32d63df726afeffa99e90e3d493c82c9a865ee20f33dcc715e7d31a8744b`.
- Physical/direct Cartesian AO dimensions: 768/800.
- Nonzero-source AO quartets at total angular order >=4: 595,220,532.
- Hypothetical AO product rejections: 35.2179% at 1e-14, 10.0236% at
  1e-16, 1.1385% at 1e-18.
- Single-center work in that domain: only 0.0315%; not the major target.

These are admission-logic replay counts, not instrumented production root
counts. Exhaustive diagnostic candidate enumeration is not the production
scheduler's candidate count. AO rejection percentages do not predict GPU or
endpoint speedup: warp divergence and additional bound loads can absorb the
benefit. Low-order weighted shell roots do not evaluate one recurrence per AO,
so their AO counts must not be interpreted as executed-root counts.

The PBE0 RKS total-density force coefficients are **1.0 and -0.125**, not -0.25.
The old doubled-exchange diagnostic was detected by independent source-channel
comparison and retained separately. It is not an exact-PBE0 timing baseline.

## Experiment and boundaries

`GENERATIVEQC_BOUNDED_FORCE_AO_DENSITY=1` experimentally refines the shared
full-range high-order force gate before the expensive derivative recurrence.
Only the exact string `1` enables it. Default, SR/LR, low-order weighted roots,
and other consumer routes retain their previous policy.

The AO predicate preserves forward/reverse density entries, separate Coulomb
and same-spin exchange products, nonfinite admission, zero-screening requests,
and `min(screening_tolerance, kForceDensityProductScreeningTolerance)`. The
actual existing force cap is **1e-14**, not 1e-18. It never screens on a combined
J/K coefficient. This remains an additional screening approximation, not a
rigorous bound on ERI derivatives; independent numerical qualification is
mandatory even though the shell-level predicate uses the same products.

No new integral recurrence, mathematical force contraction, precision mode,
resident allocation or alternate PBE0 implementation is introduced.

## Initial controls

The unmodified-master README campaign in Slurm jobs 5621/5622 completed all six
sizes, 72 native and 72 fresh reference endpoints. Every native row passed all
six matching-geometry reference comparisons. Maximum E/F errors were
1.0868462e-10 Eh / 3.3319348e-11 Eh/Bohr; all observed reference XC objects had
functional 406 and `on_gpu=True`.

Native 96-atom warm/moved-warm medians were 78.084483/78.166849 s. Reference
medians were 15.566224/10.268098 s, but reference warm SCF iterations were
`[4,5,6,4,3]`, moved-warm `[1,1,5,1,1]`; native warm calls all took one iteration.
The ~5.02x observed warm ratio does not establish improvement over historical
~8x observations. Cold and moved native costs were 767.369535/260.247219 s,
versus reference 78.761141/87.990036 s; all costs and repeats are retained.

The prototype has five passing adjacent host controls. Job 5626 checks 31
explicit real-device predicate cases (RKS/UKS, both density orientations,
opposite-spin exclusion, cancellation, zero/loose thresholds, nonfinite input
and offsets). Memcheck, initcheck, synccheck and racecheck report zero errors
or hazards for that predicate probe. It is not an independent force oracle.

## Prototype replay and numerical qualification

Prototype native source identity is
`f580d9107b02c9ed7a529702f9a0b7fa096e0706a356013631c8b5b9098ab618`;
library SHA256 is
`8cb2ec0a8fbc7700f7a0371e2dc48c28f0552d1e0b9de3da8ee0739c7dc2a6b8`.
The independent source replay, Slurm job 5627, alternated OFF/ON, ON/OFF,
OFF/ON on one GPU. Three clean integral-source timings per variant were:

- OFF: 26.732234, 26.716232, 26.730820 s; median 26.730820 s.
- ON: 23.532602, 23.525209, 23.561354 s; median 23.532602 s.
- Observed source-median reduction: 11.9645%; not an endpoint speedup.
- Maximum error against both independent accepted-state source channels:
  OFF 4.97e-13, ON 5.96e-11.

At this replay stage production recurrence counters were unavailable. The 35.2%
diagnostic AO admission estimate could not replace that missing observation. Compiled RKS
force kernel resources remain 255 registers and a 90,392-byte static stack;
zero resource fields on out-of-line device functions do not imply zero use.

Job 5628 passed independent CPU-ERI through-f production-source checks with
refinement OFF and ON. Memcheck, initcheck, synccheck and racecheck also passed
on the production source, in addition to the predicate probe.

Job 5635 passed four complete moving-grid and reconverged finite-difference
controls: water RKS and water-cation UKS, OFF/ON, full spherical def2-SVP and
48x16x32 atomic grids. Independent CPU PySCF/LibXC was pinned to 2.14.0/7.0.0.
The maximum analytic-force oracle error was 6.061e-11 Eh/Bohr; directional
finite-difference agreement at steps 3e-4/1e-4 Bohr was better than 7e-10.
The shared CPU oracle gained an explicit AO-representation keyword with its
existing Cartesian default preserved; spherical controls do not compare to a
different Cartesian basis.

### Retain the unsuccessful OH control

The original job 5633 passed both water-RKS controls but **failed both OH-UKS
controls at SCF convergence**, before forces or finite differences. Its exact
test source, log, JUnit, outcomes and successful RKS points remain separately
under `.artifacts/fd-oh-5633/`; the original `.artifacts/fd/` is not overwritten.
Adding the water-cation control does not make OH pass or resolve this failure.

A separate OFF diagnostic (job 5636) exhausted 200 iterations, with density
RMS 6.7503e-4 and physical residual RMS 4.5763e-6, and published no forces.
The unmodified master binary independently reproduced failure in job 5638:
200 iterations, density RMS 6.6256e-4, physical residual RMS 1.0325e-7,
and no forces. Full histories are retained in each tree's ignored
`.artifacts/oh-diagnostic/`. This establishes a pre-existing SCF acceptance
problem; its root cause is not established here and no convergence fix is
claimed. Do not weaken thresholds, relabel failed points, or claim OH force
qualification from the successful water-cation test.

That diagnostic also exposes a benchmark-observation gap: the public top-level
`BatchItemResult.fock_builds` is absent for this KS result, but the actual native
`ks_diagnostic.fock_builds` is 200. Future measurement should read the available
KS counter rather than infer builds from iteration counts. Already completed
campaign records with missing counters remain missing; the frozen live runner
must not be changed midway through a campaign.

## Required gates and retained evidence

### Completed ordered endpoint campaign

Jobs 5629/5630 and the separate H2CO holdout 5632 completed 168 native and
84 fresh-reference E/F endpoints at prototype source `f580d910...`. All 1,008
native-to-matching-geometry reference pairs pass the unchanged 1e-8 Eh / 1e-7
Eh/Bohr gates; maximum errors are 1.055014e-10 / 8.461238e-11. Every observed
GPU4PySCF XC object reports functional 406 and `on_gpu=True`.

| Atoms | OFF warm median | ON warm median | Fresh reference warm median |
| --- | --- | --- | --- |
| 3 | 0.350300 s | 0.349689 s | 1.140694 s |
| 6 | 0.750147 s | 0.742867 s | 1.822183 s |
| 12 | 1.666102 s | 1.629068 s | 1.383558 s |
| 24 | 5.053665 s | 4.848362 s | 2.207270 s |
| 48 | 18.017958 s | 17.216479 s | 6.002082 s |
| 96 | 78.015489 s | 74.788140 s | 10.292367 s |
| 4 (H2CO) | 0.771321 s | 0.712108 s | 1.079536 s |

These are ordered reference/OFF/ON runs with shared caches, not interleaved
endpoint causality. The 48/96 warm decreases are 4.45%/4.14%, not 35.2% or
12%. Native warm iterations are all one; reference 48 warm iterations are
[3,3,4,3,3], whereas reference 96 warm iterations are all one. The large
remaining native/reference gap is not resolved.

Retain cold and moved results rather than normalizing by iteration count:

- 48 cold OFF/ON: 183.897191 s (23 iterations) / 191.859048 s (25), a regression.
- 48 moved OFF/ON: 76.104507 s (13) / 70.906743 s (12).
- 96 cold OFF/ON: 695.004184 s (25) / 615.300945 s (31).
- 96 moved OFF/ON: 354.828279 s (18) / 256.939985 s (12).
- 96 moved-warm OFF/ON: 78.176954 / 74.905966 s, all one iteration.

Force-only screening cannot be credited for SCF iteration changes. These
timings include cold setup/cache effects and retain their own trajectory counts.

### Actual producer observations, separate source

Job 5647 instruments actual AO gates and gradient call sites at source
`d884bfc505651b350580996c16b933094083bc8ad307e536286636de695bd440`, library
`71b43977c436b8ec9d55e0b48fd582315b7a710948b242d72e2c6b540f7c08ac`.
The base is `dc6ea9940ab51b58a985ea0f804ee6a3f8f7179f`; the only upstream
change after 9c54107c is evidence retention, not runtime code. The probe binary
is `e0fc9b24c5419f9340ee38e368e8c2e7d84f854ce2627ad7a8a503f611443083`.

For the same accepted density and RKS coefficients 1.0/-0.125:

| Actual generic-force observation | OFF | ON |
| --- | --- | --- |
| Decoded AO quartets | 596,115,936 | 596,115,936 |
| AO Schwarz rejections | 895,404 | 895,404 |
| AO density-product rejections | 0 | 209,624,346 |
| Admitted nonzero-source AO quartets | 595,220,532 | 385,596,186 |
| Explicit all-center gradient calls (orders 4–6) | 579,805,284 | 379,086,118 |
| Unique-center Dual3 gradient calls (orders 7–8 here) | 33,819,696 | 11,952,272 |

Every class matches the independent admission replay; decoded capacities and
gate conservation also hold. Each independent source-channel error is <=5.96e-11.
This establishes 35.2179% fewer generic admitted AO quartets on this density.
Neither gradient-call counter is a primitive/root/FLOP count. Low-order weighted
shell-root work is not observed, not zero. Density rejections can subsume former
zero-weight work in other inputs; compare actual admission/call totals.

The observer borrows zeroed class-major storage through stream completion,
preserves the existing public shell-profile ABI and allocates nothing on normal
execution. Null is the normal path. Atomic profiling durations are retained only
as diagnostics; they cannot replace clean source or full-endpoint timing. The
new source must be independently qualified before shipping; f580 timing is not
relabelled as d884 timing.

### Shared J/K follow-up and measurement

The KS iteration still calls `enqueue_prepared_cuda_fock` on full density. HF
already has common prepare/finalize incremental kernels and accepted anchors.
Its nonzero-screening policy allows at most one delta application between full
refreshes; interval one means full/delta/full, not disabling incremental work.
The retained anchor includes hcore plus linear J/K, whereas KS separately owns
J, K and nonlinear XC. Reusing that machinery therefore requires a compatible
linear-provider boundary and strict full-density final checks, not caching a
complete nonlinear KS Fock. No incremental-KS integration is claimed here.

The next-run shared README harness now reads the actual
`ks_diagnostic.fock_builds` counter, records its origin, and retains the physical
residual. Legacy and missing counters remain explicit; historical campaign JSON
is unchanged. Host tests cover failure, zero and unavailable counter cases.

Latest-master retained evidence is 51,721,618 bytes. A conservative union with
published #1767/#1779 additions is 52,322,143 bytes, leaving 14,786,721 bytes
under the unchanged 64 MiB cap. This is a capacity preflight, not publication
validation or a reason to retain logs/build products. The incoming 2 MiB review
budget and per-file cap still apply.

- Same-GPU interleaved full integral-source replay, separately from profiling.
- Independent through-f CPU-ERI source oracles, including RKS/UKS, and sanitizer
  qualification of the real production source, not only the predicate probe.
- Actual production work counters before claiming executed-root reduction.
- Complete E/F at 48/96 atoms, small controls, non-water holdout, both geometries,
  all repeats and failures, and reconverged finite-difference checks.
- Exact source/library identity, true resource costs, unchanged fallback and
  genuine review approval before any promotion or merge.

Ignored diagnostics live under
`/data/jzzeng/qc-pbe0-master-benchmark-20261003/.artifacts/ao-census/` and the
prototype worktree's `.artifacts/`. They are local evidence, not a public
reproducibility package. Retention cap compliance remains a publication gate;
do not raise the cap or use Releases to bypass it.

## References

### Integration checkpoint

Draft PR #1798 initially published head 81392aaca with source
`b319ecfb7c5437621d1b9fc3bb13f51dc99c5411ccc38dc1a6803b3dac7aea45` and
library `e0eb6d2fccb8eb173a6b676e4f4ecbb1941fd4bcca815431728295395417cfc5`.
Job 5653 reproduces every actual-work count above on this source. Its clean,
same-GPU interleaved source medians are 26.532184 s OFF /23.363008 s ON (11.9446%
decrease), again not complete endpoint timing; all source errors are <=5.96e-11.
Its complete endpoint/FD campaigns remain separate from the original prototype.
The first small-campaign launcher, job 5649, exited 127 before any endpoint due
to SSH flattening the multi-size environment argument. Its log is retained; job
5652 uses a quoted export in a remote script and is not labelled the first run.

Master subsequently merged the shared grid planner #1773 at 701e00db7. The
integration preserves both its actual reference-XC backend observer and this
PR's native Fock-count observer; the conflict was only adjacent helper insertion.
The planner's runtime changes require a new source identity and qualification.
Do not mutate or relabel the in-flight b319 campaign to claim integrated timing.
No default promotion, convergence closure or merge approval follows from this
source integration or from the 93 passing/3 skipped host controls.

- [Shared priority and acceptance contract](https://github.com/jinzhezenggroup/generativeqc/issues/1423#issuecomment-5968957958).
- [Fresh master results and census](https://github.com/jinzhezenggroup/generativeqc/issues/1423#issuecomment-5971121015).

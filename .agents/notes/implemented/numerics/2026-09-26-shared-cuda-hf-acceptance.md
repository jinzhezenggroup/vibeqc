# Decision: share CUDA direct/DF HF acceptance

Status: implemented
Date: 2026-09-26

The subsequent [qualified singleton RHF warm-state change](../performance/2026-09-26-df-qualified-one-step-warm.md)
adds exact density/frame/energy reuse without changing this acceptance rule.
The measurements and missing-cache diagnosis below describe this earlier step.

## Problem

The [warm convergence diagnosis](../performance/2026-09-26-df-qualified-one-step-warm.md#historical-five-step-diagnosis)
found a five-step DF replay whose density-step criterion already passed at step
one. Direct retained a qualified energy baseline and allowed a scale-dependent
FP64 comparison guard; DF reconstructed its seed, discarded the baseline and
used an unguarded energy comparison. Identical public tolerances therefore did
not imply identical iterative energy acceptance.

## Decision

`src/scf/cuda/scf_convergence_policy.cuh` owns the scalar FP64 HF comparator and
the one-warp physical-residual maximum. Direct and compact DIIS-enabled DF,
both RHF and UHF, require finite current/previous energies, density RMS and
physical residual, and all of:

- `abs(E - previous_E) < energy_tolerance + 16*epsilon*max(1, abs(E), abs(previous_E))`;
- `density_step_rms < density_tolerance`; and
- `max_abs(FDS-SDF) <= min(1e-8, density_tolerance)` before DIIS extrapolation.

The same factor previously qualified for direct now applies to DF energy
contractions/reductions. Its finite scale is independent of molecule-specific
selectors and user density tolerance. The physical maximum covers both UHF
channels and rejects every nonfinite entry. DF reuses its already allocated
DIIS residual; it adds no persistent matrix or extra residual GEMM. Direct
energy-only target iterations now also check the already computed residual.
Coarse mixed stages still defer to mandatory FP64 target refinement.

The raw absolute energy change remains observable. UHF compares against the
old baseline before preserving its existing update of the previous-energy
buffer; comparing after that update would incorrectly turn every delta into
zero. Density-retention/generation, active masks, graph tails, limits and final
validation are unchanged. Low-level no-DIIS compatibility callers do not have
iterative residual storage; their final validator remains mandatory. Host
numerical recovery retains its stricter unguarded energy comparison.

## Retained boundaries and rejected shortcuts

No cached host final energy is copied into DF's device loop, and no orbital or
factor identity is manufactured from a small density step. DF still starts
with an absent baseline; direct may reuse an exactly matched density/geometry
baseline. This is a work/reuse difference under the shared comparison rule.
Solvers are not forced to report matching iterations or J/K counts.

The earlier seed-to-canonical energy shift near `2e-10 Eh` remains larger than
the common guard on the 768-AO example. It must still fail that energy test.
Its complete decomposition and a qualified DF warm-state cache remain future
work. Removing physical final validation or arbitrarily enlarging absolute
tolerances would not be a valid replacement.

The CUDA ownership ledger keeps electronic energy equations scientific and
the extracted convergence/reduction policy runtime, as before. The dependency
checker permits the common leaf policy in DF kernels without admitting direct
queues or host plans.

## Evidence and benchmark contract

The native convergence suite applies independent positive/adversarial fixtures
to all four kernel routes: few-ULP deltas, a larger real delta, missing baseline,
nonfinite energies/density/residuals, localized beta residuals and inactive
neighbors. Both families must return the independently specified verdict and
preserve raw deltas and counts. Existing physical-force, mixed-precision,
final-snapshot and occupied-response native suites also pass. Forty-four GPU
molecular cases cover RHF/UHF, both representations, DIIS histories, warm and
changed geometry, property transitions and exact/corrected final K, against
independent PySCF energies and forces.

`benchmarks/run_hf_acceptance_benchmarks.sh` measures complete cold, frozen warm,
moved and moved-warm RHF energy/analytic-force endpoints at 24–768 spherical
def2-SVP AOs. DF uses explicit cc-pVDZ-JKFIT and the qualified packed-single,
occupied, fitted-response route; this does not promote a new default planner.
Both native methods use the same library, `1e-12` energy, `1e-10` density and
`1e-12` screening inputs. Each approximation has its own GPU4PySCF reference
with `1e-10` orbital-gradient tolerance; the independent output gates remain
`1e-8 Eh` and `1e-7 Eh/Bohr` for every native endpoint, including diagnostics.

Engine processes run sequentially under one finite Slurm allocation so neither
retains its large DF B alongside the other's memory. Five warm repeats use
each engine's frozen post-cold/post-move density. GPU4PySCF's actual SCF
`get_veff` count includes its pre-loop Fock; native unavailable Fock counters
remain null and separate DF traces retain executed work. Counts are evidence,
not a filter that discards inaccurate or slower branches. Diagnostic timings
never enter the clean medians. That campaign's figure was generated from the
checksum-bound [qualification](https://github.com/njzjz-bot/vibeqc/blob/b2e57efe9af86bcaf08936c5a2ca287942658a27/benchmarks/results/hf-unified-acceptance-20260926/README.md).

All 156 native endpoints passed, with maximum errors `2.6421e-10 Eh` and
`1.4052e-10 Eh/Bohr`; the convergence suite also passed Compute Sanitizer
memcheck with zero errors. At 768 AOs, the shared guard accepts the third DF
warm iteration (`4.5475e-12 Eh` raw energy change), reducing the earlier
five-step replay to three. The complete median is `8.009645 s`, versus
`3.055674 s` for direct's one-step replay. At 384 AOs the medians are
`1.003100 s` and `0.765330 s`; at 24–192 AOs this DF route is faster than
direct. The earlier `9.842618 s` DF measurement uses a different binary and
is historical context, not a paired same-binary speedup experiment.

GPU4PySCF DF warm repeats vary across 1/2/3 iterations at 384 AOs and 2/5/7
at 768 AOs. The figure preserves these repeats as scatter rather than joining
their median to a stable-work line. Its own stopping policy is not changed to
match native acceptance, and no iteration-normalized kernel claim is made.

## Revisit when

A provenance-bound DF warm density/occupied-frame/compatible-energy cache can
remove seed reconstruction while preserving the same acceptance, independent
energy/force gates, final-state identity checks and bounded fallbacks. Recheck
the common guard on larger or ill-conditioned workloads before changing its
factor or interpreting iteration parity as equivalent scientific work.

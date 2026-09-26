# Proposal: retain one determinant in the direct HF Pulay response

Status: proposed
Date: 2026-09-17

## Problem and bounded diagnosis

Direct CUDA finalization retains P with its physical F(P) for energy and
two-electron response. It separately diagonalizes F(P) and currently constructs
the Pulay weight from those new canonical orbitals. At finite SCF residual those
orbitals reconstruct a slightly different density from the retained P.

The unchanged strict 96-AO batch-4 case has native force error about 3.94e-11
against the independent tighter CPU oracle. In a separate accuracy-only probe,
the frozen one-update branch is reconstructed from the actual post-cold seed
with CPU F(P0), then CPU F(P1). The reconstructed canonical full force agrees
with actual native forces within 3.32e-12. Replacing only the reconstructed Pulay
weight with P1 F(P1) P1 / 2 lowers the independent force error to 1.49e-11.
This is a reconstruction experiment, not a measured native implementation gain.

An earlier diagnostic enabled warm-state updates to export a final density.
That clears the frozen energy baseline and produces two SCF updates. Preserve
it as a distinct experiment; it cannot establish the unchanged one-update result.
At the tightest control the returned warm density may also have advanced after
force evaluation, so a returned restart state is not universally the force state.

## Candidate and invariants

Experiment with W = P F(P) P / 2 for RHF and W_sigma = P_sigma F_sigma P_sigma
for UHF. These expressions equal the canonical weight at stationarity and retain
the same determinant as the other force terms at finite residual. Reuse final
iteration scratch and preserve P/F, all convergence controls and numerical gates.
The native matrix-product and cuBLAS routes must apply the same RHF occupation
normalization. Energy-only execution must avoid the new products.

## Required qualification before promotion

- Compare actual native full forces against independent tight CPU oracles and
  the predeclared strict gates, preserving every failure.
- Validate RHF/UHF, small/no-cuBLAS and batched/cuBLAS routes, cold/warm/changed
  geometry, and existing state/finite-difference/native checks.
- Keep stock reference convergence errors separate: a native improvement does
  not waive an external paired failure.
- Measure complete endpoint cost after builds/tests finish. Do not infer a
  speedup from the algebra or omit the two additional matrix products.

This is an unqualified local candidate. #206 remains open; no production
promotion or general GPU4PySCF superiority is claimed.

## Initial native observations

The candidate builds successfully with source identity
`a2d0bbde1f612ddf2a25ad34926373837ce17eefd691f752304104d84260f7ad`
and binary SHA-256
`944f41d1d11f464ba813475f82e2e3dc891e9e5f490ee82a74f679fae377e374`.
Slurm job 9879 passes all 48 native tests, including the explicitly enabled MP2
allocation-status check. This is separate from endpoint timing.

Accuracy job 9878 uses the original native controls. Native 96-AO batch-1/4
warm errors against the tight CPU oracle are at most 1.49e-11; cold errors are
at most 2.93e-11. Matched explicit two-AO RHF/UHF fixtures exercise the native
matrix-product route and pass cold/warm/changed force comparisons near 1e-14.
An earlier job 9877 used named STO-3G tables with differing coefficient digits;
both arms failed those tiny checks. Keep that attempt as a fixture mismatch,
not as evidence of candidate accuracy or an accepted relaxed tolerance.

The 19-AO batched OH/UHF probe still fails its declared 1e-9 force gate: cold
1.2696e-9 and warm 1.0295e-9, predominantly a transverse force component.
The original library is worse (about 1.55e-9 and 1.27e-9), but improvement does
not make this probe pass. Changed geometry is within the probe gate.

The unchanged seven-repeat external matrix in job 9880 also retains a strict
96-AO batch-4 failure: maximum paired force error 3.506e-11 exceeds 3e-11.
The 96-AO batch-1 and both 192-AO points pass all seven pairs. The complete matrix
exits 2 and remains failed; these observations do not promote this candidate.

Exact scripts, input/output arrays, old attempts, source patch and binary
identities remain under `.artifacts/issue206/` in the experimental worktree.
Further work must distinguish remaining native stationarity error from the
stock reference's convergence sensitivity. Any additional final-state operator
or iteration must be counted and charged to the complete endpoint.

## Canonical force-density experiment

A second candidate consumes the final physical-Fock eigenvectors to construct
the force determinant, explicitly rebuilds F at that density, and computes
every force term and the returned warm state at that same P. This adds one
physical Fock build and one density update per converged force item. It does
not remove the preceding final build, and energy-only execution skips this work.
The two PFP products also remain necessary. The host-only progress journal
records submissions separately from completed item work; public iterations
continue to count the iterative loop.

Source `2e586a71c4d7525b8c7115bbb3b50bce603e06fa126be103376911f0843be05a`,
binary `44c9eb03664cfe4017993b09bb3e24846198b37a65dc0872078e033ff60bb4c8`,
passes all 48 native tests. The version-three accuracy probe passes exact
two-AO RHF/UHF and lowers 96-AO warm errors to 5.80e-12, but retains a failed
OH cold force error of 1.02953e-9 against its original 1e-9 gate. The first
two runner attempts failed script assertions: generic journal growth was
mistaken for added force work, then an energy-only call omitted its changed
coordinates. Neither script error changes molecular acceptance. All attempts
remain archived; only version three completed its full declared checks.

## Physical-residual diagnosis and third candidate

Independent CPU F(P) reconstruction at the second candidate's actual OH force
densities finds maximum commutators of 9.21e-10 to 1.05734e-9, even though
canonical density RMS is 7.38e-11 to 8.45e-11. Electron traces and metric
idempotency agree near 1e-14; CPU consistent-density forces reproduce native
forces within 4.55e-14. The shared strict final-state criterion is maximum
commutator <= min(1e-8, requested density tolerance), here 1e-10. Four plain
CPU correction steps still leave residuals above that criterion. This is
stationarity error; adding an arbitrary small fixed number of steps is not
an acceptance policy.

The third candidate tests this physical maximum inside the existing direct
force SCF loop, reusing the pre-DIIS residual and the caller's iteration limit.
Both spin channels participate; nonfinite entries fail closed. After the
canonical force update and explicit F(P) rebuild it computes four device matrix
products to recheck the final physical residual. Failure clears convergence
before force output. This adopts the shared residual criterion only, not its
entire eigenframe/identity/correction provider contract. Energy-only and detached
reference paths keep their prior iterative stop. No CPU scientific work enters
production.

The added validation reuses the dead final-Fock selection counter and existing
matrix scratch. Its completed journal includes rejected determinants, so failed
validation cannot erase executed density/Fock work. It adds a four-byte scalar
download at the existing completion fence. Total final work includes one density
projection, one physical Fock, four residual products and two PFP products per
tested item; any extra loop iterations are reported normally.

Source `79524ae99d1d929b63f96df7250b57aa35285d18a07215640b73383b4114879a`,
binary `6279948a27565dfa4a1ff8ce0bd7edc7b7f39b43d87953e0beb877c9e9a01118`,
passes 49 native tests (including physical-residual rejection, inactive items,
nonfinite input, both spins and both density-retention modes) and 106 host
structure/ownership tests. The unchanged independent CPU accuracy probe passes:
96-AO cold/warm maxima are below 5.79e-12/4.24e-12; OH cold/warm/changed maxima
are 6.12e-11/3.95e-11/3.98e-11. The OH cold loop grows from 22–24 to 35–37
iterations and the 96-AO cold loop grows from 16 to 18. These costs must remain
visible in complete endpoint comparisons. No timing or promotion claim follows
from this accuracy-only result.

The third candidate's clean job 9888 passes all seven pairs at each of the
four original direct points. Maximum paired force errors are 2.13902e-11,
2.33910e-11, 5.31861e-11 and 2.35223e-10 for 96/b1, 96/b4, 192/b1 and
192/b4. Complete ordinary warm medians are 0.198478, 0.455733, 0.487223 and
1.634662 seconds versus stock 1.692703, 6.776789, 2.240891 and 7.818306.
The iteration branches differ; these are ordinary converged endpoint results,
not an equal-work kernel speedup. No builds or profiler ran during this matrix.

## Final source review and validation

Review retains the existing stopping strategy for each item's coarse mixed
stage and applies the physical criterion to its mandatory exact FP64 refinement.
Exact neighbors keep the criterion even in a mixed bucket. The gate uses the
same per-item census as the refinement owner; global mixed capability alone
cannot disable an exact item's check. The native regression additionally
tests coarse-stage deferral and final rejection independently. Source formatting
follows the repository's pinned hooks.

Final source identity
`7951fbce9d6bb0d39332e4a40c9a3919cfad0869e7f6fb360951c609dc33daa7`
and binary SHA-256
`e2e92b07f1fcfc681a88277710cbac68305b89626bc78464873853ebcadb4d45`
pass 49 native tests, the retained independent-CPU OH public regression and
the complete accuracy probe in Slurm job 9889. Independent CPU reconstruction
of the actual final OH densities gives maximum physical commutators between
5.44e-11 and 6.24e-11, below 1e-10; its forces agree with the native consumer
within 4.10e-14. Final-source external timing is recorded separately from
job 9888 so those earlier samples are never silently attributed to this build.

The final-source job 9890 retains a new 96/b1 paired failure: repeat 4 has
maximum force difference 4.24283e-11 against the unchanged 3e-11 gate. Both
engines take one iterative update in that pair. Exact-geometry independent CPU
decomposition finds native error 3.28845e-12 and stock error 4.08849e-11;
stock's final orbital-gradient norm is 3.82957e-10, below its original 1e-9
control but insufficient for this force gate. All seven native samples remain
within 3.30e-12 of the CPU oracle. This diagnosis does not turn the paired
failure into a pass, discard that repeat or replace it with job 9888. The
original external acceptance remains unsatisfied by this final build's campaign.

The completed job 9890 also retains a 96/b4 failure (paired maximum
3.70581e-11). The native/stock maxima against its independent CPU oracle are
4.24476e-12/3.56404e-11. Both 192-AO points pass all seven pairs. Overall
matrix exit status is 2: two passed points, two failed points. This source
remains a draft candidate; do not merge or claim #206 completion on the basis
of the earlier passing job. The retained stock convergence evidence must be
resolved under an explicit acceptance policy, without dropping observations.

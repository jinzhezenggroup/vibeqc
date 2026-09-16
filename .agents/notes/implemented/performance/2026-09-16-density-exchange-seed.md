# Decision: factor a supplied RHF density for seed exchange

Status: implemented
Date: 2026-09-16

## Problem

#399 identifies a mandatory dense seed RI-K even when the supplied density has
bounded PSD rank. Canonical orbital provenance is unavailable at solve entry,
but exchange only needs an algebraic factor of that exact density.

## Decision

Reuse the compact device solver and transient SCF matrices. Accept only a PSD
spectrum, bounded rank and complete reconstruction under explicit absolute
FP64 gates. Store sqrt-occupation-scaled columns in existing factor storage and
use exchange weight one. Do not tag this algebraic factor as canonical; the
first SCF update overwrites it before graph capture. Unsupported plans retain
dense seeds and CUDA errors remain errors. Both automatic selectors use only
the measured 768/768/160 singleton resident RHF domain on the actual RTX 5090.
Explicit controls permit guarded comparisons elsewhere; 384 remains opt-in.

The second dense K is `finalize_density_fitting_rhf`'s physical-provider callback
inside `select_cuda_df_final_state` / `select_final_state`. This is a required
physical F[D] evaluation, not a redundant request for the previous Fock. A
separately controlled retained-C route still builds physical J/K, requiring an
exact token, device generation, successful solver and exact supplied D match.
Correction steps with new density/generation retain dense K.

## Invariants

- Occupation is absorbed in the seed factor; canonical/final RHF C uses weight two.
- Never retain a truncated spectrum solely to fit exchange capacity.
- Preserve SCF update counts and all physical convergence/force gates.
- Scratch is borrowed only outside capture, before the first iteration or after
  completion. No extra nbf-by-nbf-by-naux tensor or persistent factor is added.
- Measure seed-only, final-only and combined controls separately on frozen D.
- UHF, batches, streaming and source-backed plans remain guarded fallbacks.

## Evidence

Five interleaved seed-only 768 AO repeats reduce the complete energy+force
median from 5.316988 to 4.689939 s, and energy-only from 3.445536 to 2.818769 s.
Every sample replays the same density and executes three SCF updates. Independent
retained GPU4PySCF energy/force gates are 1e-9 Ha and 1e-8 Ha/bohr. A separate
final-K comparison reduces energy+force from 4.685339 to 4.045592 s and energy-only
from 2.820675 to 2.177101 s. Do not add percentages from the separate ablations.

The real 768 AO seed retains rank 160: maximum/RMS density reconstruction errors
are 2.665e-15/6.245e-17, direct K errors 1.870e-13/2.098e-15, and complete Fock
errors 9.348e-14/1.049e-15. H and J are identical, so Fock error is half K error.
The added factorization costs about 16.4 ms in the separate component trace;
the complete seed iteration falls from about 924 to 276 ms. No persistent
allocation changes; only the spectrum/status and two scalar residuals are read
back (6164 bytes and two explicit stream synchronizations at 768 AO).

The 384 AO opt-in comparisons also improve, but use occupied SCF in both arms
and are not a comparison against the default dense 384 SCF policy. Automatic
selection therefore remains restricted to 768. Native fixtures cover fractional
PSD, zero, tiny discarded modes, excess rank, negative spectrum and asymmetry,
against an independent four-index RI contraction. Final identity tests reject
changed density, stale host/device generations and failed solver status.

Reproduction, complete arrays, final selector qualification and separately
instrumented transfers/residency are under
`benchmarks/results/issue399-density-seed/`.

## Rejected alternatives

Trusting imported orbital provenance is unnecessary: seed exchange needs only an
algebraic factor of the supplied D. A second eigensolver or persistent factor
would duplicate existing ownership and workspace. Silently truncating density
to the reserved rank would change the model and is rejected. Reusing an old
physical Fock would skip the strict final consumer's semantics; retained C is
used only to rebuild J/K under exact identity gates.

The validation executable initially read an asynchronous nonblocking-stream
result through default-stream cudaMemcpy; same-stream readback fixed that harness
race. Its nonsymmetric fixture also needed explicit row-to-column-major input
conversion. Frozen shared libraries require a libvibeqc.so.0 SONAME symlink for
separately linked native probes. These are harness requirements, not weakened
production gates.

## Revisit when

Expand automatic policy only after matching-density endpoint evidence supports
the new domain. Rank gates are scientific tolerances, not performance knobs.

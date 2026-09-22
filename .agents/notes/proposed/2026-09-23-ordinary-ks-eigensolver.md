# Candidate: ordinary KS must not force graph-native diagonalization

Status: proposed (independent GPU solver test passed; endpoint qualification incomplete)
Date: 2026-09-23

## Problem and evidence

After generated pure J removed the first direct-DFT bottleneck (#1077,
PR #1086), a 24-atom/192-AO PBE cold endpoint still exceeded a 60-second
deadline. A separate fixed-two-iteration diagnostic was stopped after 45
seconds. On RTX 5090, its first completed maximum-pivot eigensolver kernel
took 9.2065 seconds, 72.5% of 12.7048 seconds of completed GPU kernel time.
The incomplete kernel at cancellation is not included in those totals.
This is diagnostic evidence, not a complete endpoint speedup claim.

KS explicitly selected graph-native Jacobi above 16 AOs in all three solve
locations, despite executing on an ordinary stream. The repository already
has ordinary Xsyevd dispatch with inactive-input sanitization. Memory bounded
native Jacobi is not work bounded for large dense molecular Focks.

## Candidate

Add a prepared ordinary-stream owner beside the shared eigensolver dispatch.
It retains the existing small native solver through 16 AOs and uses existing
Xsyevd dispatch above that size. It owns handles and explicitly charged
numeric workspace, borrowing the KS stream and matrix buffers. Both spins
reuse one workspace serially. All three KS solve sites use this owner,
including final-state canonicalization and the optional two-slot submission.

The shared shape-only workspace allowance is extracted from the established
DF provider contract without changing that bound. Both host and device query
sizes are checked before allocation; the KS resource estimate reserves both.
Excess capacity fails explicitly, rather than selecting an unexpectedly slow
native fallback. The ordinary owner rejects stream capture. Existing graph
consumers keep their separate qualification and native fallback policies.

This is runtime/provider ownership, not new eigensolver mathematics. Reuse the
canonical native dispatch rather than generating a second diagonalization
algorithm or staging SCF matrices through a CPU solver.

## Acceptance

The new allocated GPU test uses independent dense Householder spectra at
7/24/192/768 AOs, two spin-sized matrices and repeated solves. It checks every
eigenvalue, analytic eigenvector/residual, inactive nonfinite sanitization,
workspace bounds and explicit capture rejection. Public RKS/UKS energy,
final-state, warm/changed-geometry and resource-ledger tests remain required.
Complete bounded PBE/r²SCAN endpoints must pass unchanged numerical and
convergence gates before promotion; keep #1077 open until larger sizes pass.

## Allocated GPU evidence

The independent 7/24/192/768-AO test passed on RTX 5090 (Slurm 11272),
including two matrices, replay, inactive NaNs, budget bounds and capture
rejection. The integration binary also includes PRs #1073, #1076, #1086 and
#1089 on an older master; this is not an exact PR-head qualification.

The same integration completed PBE24 (192 AOs): cold 30.744 seconds / 18
iterations, warm 3.436 and 3.440 seconds / 2 iterations. The identically
discretized GPU4PySCF endpoint took 5.645 seconds / 12 iterations cold and
0.752 and 0.753 seconds / 1 iteration warm. These branches are not iteration
matched. Maximum all-sample energy error was 6.296e-8 Hartree, failing the
unchanged 1e-8 gate despite tightly converged densities. The numerical source
is still under investigation; these are diagnostic timings, not accepted
benchmark results or a claim that #1077 is fixed.

Local evidence: `ks-solver-v4.log` and `pbe24-solver-v4.json` under the
integration checkout's ignored `.artifacts/gpu-blocker-fixes/`; binary SHA256
`e7759a605546ca82020f0c618bce4b2dada57dc6549e2e4442391f9db2cfb29e`.

## PBE24 numerical follow-up

The mismatch above was isolated to generated ddpp scheduling, not XC or the
ordinary eigensolver. Issue #1095 / PR #1096 fixes premature worker retirement
after a screened task. With that compiler fix, Slurm 11293 PBE24 passed the
same complete cold/priming/warm comparison at screening 1e-12: maximum error
1.444e-11 Hartree, native cold 27.249 seconds and warm 3.397/3.438 seconds.
The integration binary SHA256 is
`cdb92be28ccc73f9b3279ed2b61df7321ecb0934b40f2cbede65b23d20f0dc0e`.
Larger endpoint qualification remains open under #1077. This observation does
not change the standalone eigensolver's scientific or resource contract.

## Public ordinary-owner qualification

Slurm 11309 passed RKS and open-shell UKS water energy-only endpoints above
16 AOs (24 spherical AOs). Exact-input PySCF oracles and identical explicit
quadrature meet an absolute-only 1e-8 Hartree gate (`rel=0`) for cold/warm and
changed geometry; physical residuals stay below 1e-9. Final-state export
exercises canonicalization through the same solver. Exact-budget admission,
zero warm provider allocations and observed storage below the conservative
shape bound are checked. This supplements the independent 768-AO solver test;
it does not publish CPU performance or qualify 96-atom complete DFT timing.

The allocated CUDA resource suite also passed all seven selected tests:
RKS/UKS preparation, replay, geometry rebuild, release, larger-solver shape
queries and rejected allocation cleanup. Its SCF ledger probes explicitly
request energy, separating provider allocations from optional force consumers;
retained/rebuild allocations are compared to actual observations while the
shape query remains a conservative bound.

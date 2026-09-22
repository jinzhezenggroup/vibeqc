# Candidate: ordinary KS must not force graph-native diagonalization

Status: proposed (implementation present; GPU qualification pending)
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

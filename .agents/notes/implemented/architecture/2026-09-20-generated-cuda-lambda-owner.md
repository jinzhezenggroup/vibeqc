# Decision: generated CUDA RCCSD Lambda actions with a host solver

Status: implemented, bounded tooling consumer
Date: 2026-09-20

## Ownership and equations

`lambda_equations` and the shared TensorIR reverse generator remain the only
sources of the RCCSD energy RHS and residual-transpose equations. Six prepared
providers execute shared/expanded primal, RHS and transpose programs on CUDA.
`PreparedCUDALambda` binds the converged reference and CC state using the existing
`BoundCCSDLambda` validation. The existing host GMRES controller retains Krylov
vectors, stopping decisions and the square-root orbit-weighted independent
coordinates; CUDA actions preserve the dense-Frobenius adjoint convention.
This is not an all-device Krylov loop, a GPU CC solver or a public nuclear force.

## Admission, lifetime and publication

Explicit finite host and device budgets are required. All six providers coexist
in one resource plan/session; the retained reference and response host storage
is reserved before admission. Provider preparation rolls back after failure.
The owner checks reference generation before and after execution and rejects
any provider reporting a different backend instead of silently using CPU math.
Both shared and separately expanded primal replays must match the converged CC
state. A Lambda solution is published only after true-residual convergence and
independent expanded transpose/RHS stationarity checks. Returned amplitudes are
immutable; the mutable prepared owner must be closed after use.

## Evidence and limits

`tests/python/test_cc_lambda_cuda.py` separates host ownership tests using fake
providers from explicitly allocated real CUDA execution. The retained
`benchmarks/results/cc-lambda-cuda/qualification.json` identifies source commit
`4c581e20fbdb9ff42e750e087c05e945843f65ac`, CUDA 12.9 and RTX 5090. It records
water Lambda agreement, both stationarity residuals, six-provider resource
peaks and measured transfers. These are reference-run results, not a claim
that the host-only tests executed on a GPU. Further runs belong in their exact
PR/check records and must identify the tested source.

Compilation and capacity feasibility do not establish a speedup. Host GMRES
requires repeated vector transfers. Parameter-source response weights remain
host-generated, and complete orbital/nuclear derivatives are outside this
execution owner. No public CCSD(T), RDM or fully resident workflow follows from
this Lambda action boundary.

## Rejected alternatives and revisit criteria

A second handwritten CUDA Lambda equation stack would make the primal and
adjoint diverge. A CPU fallback hidden behind a CUDA result would falsify
backend provenance. Six independently admitted allocations would miss their
simultaneous capacity, and convergence of the shared action alone would miss a
shared-equation error. Revisit this boundary for a device-resident Krylov
consumer, broader molecule/basis coverage, lower-transfer plans or full nuclear
response only with independent numerical, lifetime and endpoint qualification.

Agent: ChatGPT
Model: GPT-6 Astra Pro

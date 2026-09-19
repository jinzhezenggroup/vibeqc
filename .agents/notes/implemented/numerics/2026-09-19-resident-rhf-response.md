# Decision: retain RHF response/Krylov vectors on the direct-J/K CUDA stream

Status: implemented
Date: 2026-09-19

## Problem

The direct CUDA RHF response path already executed exact unscreened J/K on the
GPU, but each operator action staged the response density from host and
downloaded J/K matrices before host AO/MO transforms and GMRES
orthogonalization. #180 B2 requires a truthful resident response path without
creating a second CPHF equation or a second Krylov solver.

## Decision

The existing #179 restarted-GMRES controller remains the only scalar RHF
solver. Its vector algebra is expressed through a storage engine: the default
engine preserves the NumPy implementation, while CudaResidentRHFResponse owns
device vector slots and supplies copy, scale, AXPY, dot, norm,
orthogonalization, linear combinations and operator applications.

The resident owner borrows the exact conventional FockPlan direct-J/K source,
device and stream. It uploads the canonical MO coefficients and diagonal
occupied/virtual orbital-energy matrices once. Each operator action builds the
AO response density with cuBLAS GEMM, invokes the existing
enqueue_cuda_direct_jk_device seam without host matrix staging, performs the
AO/MO transforms with cuBLAS and adds the energy-gap term as
Evirt*X - X*Eocc using GEMM. No chemistry-specific resident J/K implementation
is duplicated.

The host retains the small Hessenberg matrix and least-squares solve. Nuclear
and metric RHS preparation plus final occupied/D/W reconstruction also remain
host-side. The execution label is therefore cuda-resident-host-controlled, not
all-device CPHF.

## Transfer contract

Creation uploads C and the two diagonal orbital-energy matrices. RHS vectors
are uploaded explicitly. A resident operator action transfers no density, J/K
or response vector; it returns only the four-byte numerical-status flag.
cuBLAS dot/norm calls return scalars needed by the host controller. Final
solution publication is explicit, and directional Hessian consumers request no
final Arnoldi-basis publication.

The host-orchestrated CudaDirectJKBackend action remains available and
independently qualified. A resident request never silently falls back to it.

## Resource contract

The Hessian directional consumer checks retained direct-J/K provider bytes plus
the resident owner allocation against response_device_budget_bytes before the
solve. The owner allocation includes vector slots, canonical reference data and
all AO/MO/J/K scratch it owns. Provider-preparation temporaries, loaded library
code and CUDA-context/private runtime memory remain explicit exclusions.

## Fail-closed boundaries

- Resident execution requires the exact unscreened restricted conventional
  direct-J/K backend.
- A host preconditioner is rejected in resident mode.
- Block-Arnoldi remains host-only and rejects a resident engine; sequential and
  recycled strategies reuse the scalar GMRES controller.
- UHF, DF, KS/CPKS and public Calculator Hessian capabilities are not inferred.
- B3 second-integral HVP and final relaxation contractions remain separately
  qualified consumers.

## Evidence

A host-only wrapped-vector regression proves that the shared GMRES controller
can operate without NumPy vector storage and forbids operator.apply fallback.
The ordinary CPU clang/gcc suites protect the host implementation. NVIDIA CUDA
12.9/sm_120 compilation qualifies the native ABI and cuBLAS/direct-J/K linkage.
The CuMetal pull-request gate compares the resident device operator and solve
with the independently qualified host-orchestrated direct-CUDA response while
checking zero per-action H2D matrix traffic and only the native status D2H
before explicit publication.

## Rejected alternatives

- A second CUDA GMRES implementation: duplicates convergence/resource semantics
  and would drift from #179.
- Calling FockPlan.evaluate inside resident actions: downloads J/K and therefore
  is not resident.
- Relabeling blocked host Arnoldi or a host preconditioner as resident: fails
  the execution-residency contract.
- Differentiating through a global runtime tape: unnecessary for this explicit
  stationary response operator.

## Consequences

B2 removes the repeated host AO/MO and response-vector traffic from iterative
conventional RHF response while preserving #179 numerical controls. It does not
by itself make the complete HVP all-device because its RHS/reconstruction and
B3 assembly consumers retain their existing execution boundaries.

## Revisit when

Extend the same storage-engine contract only after spin-resolved UHF or native
KS response operators have independently qualified device actions, or when a
resident block-Arnoldi implementation can preserve the same true-residual and
workspace semantics.

## References

- #180
- #179
- #511
- #564
- tools/vibeqc_response/resident_cuda.py
- tools/vibeqc_response/krylov.py
- tools/vibeqc_hessian/directional.py

Agent: ChatGPT
Model: GPT-5.6 Sol


## 2026-09-19 review correction

The earlier Evidence paragraph overstates the CuMetal numerical coverage: the
current resident comparison explicitly skips under `CUMETAL_ROOT` and requires
an allocated NVIDIA device. The workflow now labels this as a coverage report
and displays the skip reason. A green CuMetal run or NVIDIA compile is not
proof that the resident numerical comparison executed. No new NVIDIA numerical
campaign is claimed by this correction.

The exceptional context-cleanup path also now revokes traceback-retained leases
before native destruction. Three hardware-independent regressions first
reproduced original-error replacement and missing destruction for MemoryError,
ValueError and RuntimeError, then passed after the repair. Normal explicit close
still rejects live vectors. This changes cleanup, not the response equation or
CUDA arithmetic.

Agent: ChatGPT
Model: GPT-6 Astra Pro

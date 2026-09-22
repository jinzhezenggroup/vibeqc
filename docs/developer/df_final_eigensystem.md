# Ordinary CUDA DF final eigensystems

The fused CUDA RHF/UHF DF finalizers use the existing ordinary FP64 Xsyevd
dispatch for their required physical Fock eigensolves. RHF physical-reference
export uses the same operation. The independent CPU reference eigensolver is
unchanged. Cold overlap/core preparation and the independent host-iterative
Fock API retain their existing routes in this increment of #310.

## Operation and numerical contract

`solve_cuda_density_fitting_eigen` accepts detached, finite, symmetric,
row-major AO matrices. Generalized input requires both S and the caller's
matching symmetric X = S^(-1/2). It preserves the existing overlap cutoff and
does not select a new subspace or substitute a Cholesky orthogonalizer.
Null S/X requests an ordinary symmetric solve. Inputs and the two outputs
must be distinct vectors; alias rejection preserves them.

The adapter explicitly packs column-major F and X, forms X^T F X with the
existing DF cuBLAS wrapper, calls the existing LOWER-triangle FP64 Xsyevd
provider, then forms C = X U. Output eigenvalues are ascending and coefficient
columns are returned in row-major storage. No handwritten eigensolver is
introduced and no Graph eligibility or 512-AO ceiling is imposed.

An independent backend-neutral check uses the original F/S inputs to require
finite, ordered eigenpairs, zero solver info, max |F C - S C epsilon| <= 1e-8,
max |C^T S C - I| <= 1e-8, and a Frobenius scaled eigen residual <= 1e-12.
The scaled denominator is ||F||_F ||C||_F + ||S C epsilon||_F. Nonfinite
intermediate products and scales fail. Elementary host matrix products are
shared with the reference utilities; their eigensolver is never called.
Signs and rotations within degenerate subspaces are irrelevant to these gates.

The finalizer still projects density, rebuilds physical Fock, computes energy,
and assembles the complete one-electron/Pulay/DF response for force requests.
The existing physical-reference validator also runs when exporting a frame.
Mutually consistent final-state retention and bounded correction remain #311.

## Ownership and failures

Each DF bucket lazily owns one ordinary AO frame, serialized across its items
and UHF spins. It borrows the plan's existing stream, BLAS handle and cuSOLVER
handle/parameters. Captured SCF buffers and occupied factors are not reused as
scratch. Calls require a noncapturing stream; the owning plan already serializes
access. Warm replay does not create handles, query capabilities or allocate
another device workspace.

The metric setup handle and parameters now survive until plan teardown; their
former setup-time destruction would leave the ordinary adapter uninitialized.
Temporary metric solver scratch still expires at setup completion. The retained
handles use the existing opaque library lifetime allowance; numeric workspace
has its separate explicit reservation below.

The device ledger charges three AO matrices, eigenvalues, info, active mask
and the actual queried solver workspace. Native and Python shape planners
reserve 1 MiB + 16 n^2 doubles for that workspace and separately for its host
workspace. A provider query exceeding either allowance fails as out-of-memory
before allocating the workspace. Existing serialized host SCF scratch covers
packing and frame validation; retained host workspace is additionally charged.
Partially allocated candidates are destroyed before publication, allowing a
later retry. Teardown releases the frame on its owning device before the
borrowed stream and handles. Error paths drain submitted work before temporary
host buffers die. Outputs are cleared on any non-alias failure. There is no
implicit CPU-reference fallback from this operation.

Real CUDA 12.9.1 workspace queries at 1–1536 AOs include a fixed cost of about
0.5 MiB at the smallest dimensions. Budgets must include that cost even for
tiny molecular fixtures; unsupported smaller budgets fail as OOM. Actual
query sizes are checked on every newly created frame and included in a
capacity-rejection detail, rather than assumed from the asymptotic formula.

## Causal measurement

The private diagnostic `VIBEQC_DF_REFERENCE_FINAL_EIGEN=1` restores the actual
independent reference final solves. The existing #206 runner exposes
`--host-workloads --final-eigen-ablation`: baseline uses that reference control,
candidate uses ordinary device solves, and both retain identical lazy/cached
preparation. This mode is separate from #309's preparation ablations.
The input identity records the control; intrusive tracing uses separate runs.
Actual final reference/device leaves, overlap/core/cache scopes, SCF iterations
and retry branches gate the comparison. Finalizer device traces record uploads,
transforms, solver calls, downloads and validation separately from complete
synchronized endpoint time. External numerical/performance parity remains #206.

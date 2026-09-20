# Decision: shared resident block and recycled RHF response

Status: implemented
Date: 2026-09-20

## Problem

The existing scalar direct-CUDA adapter kept long vectors on device, but block
Arnoldi constructed host panels and recycle replacement consumed published host
bases. #179 needs multi-RHS reuse and #180 needs the same solver for complete HVP
blocks, with explicit resource and publication policies.

## Decision

Reuse the existing GMRES/block controller through its vector-engine seam.
Block range expansion uses twice-reorthogonalized thin QR plus a small SVD of R;
only projected factors cross into NumPy. The singular-value cutoff remains
`max(breakdown_tolerance, eps * max(dimension, block_width) * largest_singular)`.
Long basis, RHS, solution and action vectors use the selected engine throughout.
Final solution publication is explicit; `collect_basis=False` suppresses final
basis publication independently of internal recycle updates.

Retain recycled vectors as leases on the exact resident owner. Projection and
atomic bounded replacement use the same algorithm as the host space, before
temporary solver scopes are drained. Failure leaves the old space/generation
intact. Owner/reference identity is checked even on empty/zero solves. Automatic
spaces close on every exit; callers close explicit spaces before their owners.
No action depends on traceback frames being garbage collected.

Preflight logical vector slots separately from physical arena bytes and solver
numeric storage. Consumer planners reject above the native 4096-slot ceiling.
The #180 block consumer jointly budgets direct J/K plus resident storage, then
subtracts the retained allocation from the solver's available phase budget.
The resulting bound intentionally overcounts logical device vectors already in
the arena; it is conservative, not a measurement of allocator peak usage.

## Rejected alternatives

- A second GPU-specific block solver would duplicate convergence and failure
  policies. The shared controller remains the only algorithm.
- Downloading a full block for NumPy SVD would defeat long-vector residency.
  Factoring W.T@W instead would square conditioning and weaken rank decisions.
- Download/reupload recycling would hide PCIe traffic behind a device label.
- Clamping slots or falling back after resource rejection would change the
  caller's execution contract. Rejection is explicit before upload/action.

## Invariants and evidence

The independent committed Libcint AO tensor is transformed into a tiny MO
response matrix, separately from the native CUDA J/K action. H2/LiH/water,
three strategies, rank-deficient RHS and scales 1/1e-100 satisfy solution and
true-residual gates. Host operator, integral-tile, J/K and basis-stack seams
are forbidden during resident solves. Each RHS uploads once and each solution
downloads once; native D2H bytes equal final solutions plus 8 bytes per dot/norm
and 4 bytes per action status. No numerical gate is relaxed for residency.

The 37-by-4 range-factor test compares dense SVD at scales 1e-150/1/1e150 and
both sides of the cutoff. Lifecycle tests cover wrong owner, changed reference,
closed owner, vector exhaustion, interrupted atomic replacement and retained
exception frames. Complete H2 HVP blocks for all three strategies agree with
the native dense Hessian assembly at 1e-9 absolute tolerance and forbid host
response actions. Impossible consumer budgets fail before first derivatives.

Performance reproduction uses `tools/response_resident_benchmark.py` under
Slurm. Both modes use conventional unscreened FP64 CUDA J/K, identical inputs,
thresholds and explicit final output. Every repeat is checked against the
independent matrix. Endpoint timings include publication; setup and teardown
are separately visible. Raw resident counters and audited host API payload
counts distinguish measured counters from derived counts. Solver action time
covers application only, and blocked aggregate cost is counted once.

## Consequences and revisit conditions

Host convergence, small SVD/least-squares and initial RHS-rank diagnostics remain
explicit. The complete HVP still prepares/reconstructs host data; resident
response is not an all-device molecular Hessian claim. Frequent scalar reductions
can dominate tiny GPU problems. Recycling can increase work; no speedup or
strategy/default promotion follows from fewer long-vector transfers. Revisit
automatic selection only after larger-system, changed-geometry and constrained
budget complete-endpoint evidence.

## References

- Issues #179, #180 and the existing #153 shared Z-vector consumer.
- `tests/python/test_response_block_resident_cuda.py`
- `tests/python/test_hessian_block_resident_cuda.py`
- `tests/python/test_response_krylov.py`
- `docs/response.md`, `docs/hessian.md`, `docs/performance_engineering.md`

## Consumer oracle correction

The initial consumer campaign used `tools.vibeqc_hessian.analytic.analytic_hessian`,
whose relaxation calls the same #179 solver. Calling that reference independent
was incorrect: its recorded errors qualify assembly parity only. The historical
raw record is preserved and its README now states that scope explicitly.

The validation helper now separately constructs PySCF's analytic RHF Hessian
from exact source geometry/shell primitives and independently converged SCF.
It supplies external integral derivatives and CPHF. Every complete H2 consumer
in the benchmark must pass an additional 1e-9 maximum absolute error gate;
all three resident-strategy tests enforce the same gate and forbid native
Hessian/shared response entry points during oracle construction. The native
parity gate is retained. Schema v2 records the oracle version, directions,
reference/results and separate errors, without rewriting old measurements.
This adds a validation dependency only, with no production fallback or new
response implementation. The response-level Libcint matrix gates were already
independent and are unchanged.

The supplemental clean-source campaign at `b3265686` (Slurm 1136, PySCF 2.14.0)
passes all six external consumer gates with maximum error `7.993606e-15`,
plus 103/103 integration/consumer/Krylov tests and 78/78 CUDA point directions.
`benchmarks/results/response-179-resident/consumer-oracle-evidence.json` retains
every consumer, matching binary SHA and complete diagnostics; the README states
which evidence uses the independent oracle and which records native parity.

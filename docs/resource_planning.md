# Resource planning and dry runs

`ResourceBudget` constrains the declared host/device allocation scope of a
complete prepared calculation. Providers offer finite, executable alternatives;
`plan_resources` chooses a deterministic combination that fits every cap.
Estimation builds basis/topology metadata without SCF iteration or production
integral, density, derivative or tensor arrays.

```python
from vibeqc import Calculator, ResourceBudget, estimate_hf_resources

h2 = [(1, (0, 0, -0.7)), (1, (0, 0, 0.7))]  # coordinates in bohr
budget = ResourceBudget(host_bytes=128 << 20, headroom_fraction=0.1)
plan = estimate_hf_resources([h2], budget=budget)  # no native CPU library needed
print(plan.status, plan.peak_bytes, plan.diagnostic)

calculator = Calculator(resource_budget=budget)
with calculator.prepare_batch([h2]) as batch:
    result = batch.execute(strict=True)
    print(batch.resource_plan.to_dict())
    print(batch.resource_diagnostics)
```

`Calculator.estimate_resources(systems, charges=..., multiplicities=...)`
resolves that calculator's actual basis, auxiliary basis, spin and solver
controls. `singlepoint` also accepts the calculator's budget and attaches
diagnostics to its result. Unsupported or infeasible plans fail before native
context creation in these execution endpoints. Constructing a CUDA calculator
itself can select a profile and initialize the CUDA runtime; use the standalone
estimator or CLI for a dry run without those side effects.

```bash
python -m vibeqc resources molecule.xyz --basis def2-svp --method rhf \
  --backend cpu --host-bytes 1073741824 --headroom-fraction 0.1
```

XYZ units default to angstrom; `--units bohr` is available. The CLI emits JSON
and exits 2 for unsupported or infeasible plans. CUDA estimation uses scalar
layout queries from the library selected by `VIBEQC_LIBRARY`, loaded without
profile selection or a device query. It does not choose a GPU by free memory.

## Accounting contract

Host limits cover pageable plus pinned bytes; the pinned cap applies in
addition. The device limit covers the aggregate of devices, and optional
`per_device_bytes=((0, limit), ...)` caps apply in addition. `None` is unlimited;
zero permits no allocation. Fixed reserves are withheld from their aggregate
cap, and fractional headroom is withheld from each specified cap using integer
arithmetic. A reserve without a corresponding aggregate cap cannot constrain
otherwise unlimited memory.

Each `ResourceEstimate` describes one owned capacity over an inclusive
`first_phase`/`last_phase` interval. Persistent state, caches, outputs, library
storage and runtime allowances contribute to resident bytes. Workspaces also
contribute to peak bytes. A shared workspace appears once under its owner.
Concurrent streams must declare overlapping intervals. Caller-retained outputs
beyond the declared interval require their own reservation.

Sizes/products are checked against a portable signed 64-bit limit; native
adapters additionally check `size_t`. Estimates and identities are versioned.
`ResourcePlan.to_dict()` / `from_dict()` serialize data only, verify checksums,
and recompute accounting on load. Execution adapters compare the complete
provider request with the actual scientific inputs and schedule; a checksum
alone does not authorize a foreign estimate.

The planner orders candidates by provider cost and stable names. It enumerates
up to 65,536 combinations by default; a larger search reports unsupported.
Infeasible diagnostics include the closest candidate, violated caps, per-space
minima and dominant allocations. Per-space minima can come from different
candidates and need not be simultaneously attainable.

## Current provider inventory

| Provider | Retained storage | Temporary peak and execution schedule |
| --- | --- | --- |
| CPU direct RHF/UHF | All fleet topology, warm densities and outputs | Largest serialized item: forward-mode Jet/recurrence arrays, Cartesian/public integral coexistence, SCF/DIIS/eigensolver matrices and forces |
| CPU density fitting | Fleet state as above | Also includes the current implementation's four-center ERIs/derivatives, raw metric/three-center derivatives, transformed factors and analytic metric response |
| Small CUDA direct RHF/UHF | Sum of every retained bucket arena | Exact production arena layout, plus a single-item arena for existing cold numerical recovery alongside warm caches; conservative host topology/staging bound |
| Small CUDA density fitting | Sum of all bucket J/K caches and solver workspace allowances | Maximum serialized setup/force workspace, source metadata uploads and a cold numerical recovery reservation; native tile planner supplies actual tile dimensions |
| TensorIR CUDA | Native arena, input/output storage, host staging and library allowance | Existing TensorIR liveness, tiling and recomputation alternatives; full inputs/outputs remain resident |

Budgeted CPU execution explicitly uses one native worker, including multi-item
buckets. Workspace therefore scales with the largest item and retained state
with the fleet. CUDA buckets execute serially but retain all their caches.
Geometry updates preserve resource topology identity when dimensions remain
compatible; scientific warm-state and integral-cache invalidation still apply.

CUDA direct inventory v1 supports at most 16 public AOs and DIIS history 64.
CUDA DF additionally limits auxiliary AOs to 128. Larger shapes, missing native
inventory queries and optional profiling/graph-eigensolver overrides report
unsupported. Explicit auxiliary templates must be compatible across a fleet,
matching the existing native batch contract. DFT/grid, response and correlated
providers can implement `ResourceRequest`; these methods do not yet have
execution adapters in this inventory.

CUDA DF offers resident and source-regeneration choices. A positive explicit
DF sub-budget retains the native source mode; otherwise the planner may choose
either mode. The source candidate regenerates integral tiles on CUDA and
uses the existing **CPU SCF/DIIS/eigensolvers with CUDA J/K** route. The resident
candidate uses CUDA SCF and preserves existing CPU numerical recovery. These
decisions appear in the plan. Allocation failures cannot trigger that host
recovery. The default source sub-budget includes the native one-electron
preparation minimum; v1 does not enumerate arbitrarily small DF tiles.

CUDA DF numeric solver workspaces have an explicit conservative 64 MiB
allowance plus shape terms. Opaque library retention has a separate 256 MiB
allowance per bucket and one recovery owner. These are capacity policies, not
measured library usage or minimum physical GPU requirements.

## Composing HF with TensorIR

`tensor_resource_choices` in `vibeqc_compiler.tensor.resources` adapts the existing
TensorIR planner. Its `request` joins an HF request in the same global plan;
`choices.selected(plan)` returns the corresponding exact TensorIR plan.

```python
from vibeqc import ResourceSession, plan_resources
from vibeqc_compiler.tensor.resources import tensor_resource_choices

# program, compiler and cache are ordinary TensorIR objects/paths.
budget = ResourceBudget(host_bytes=1 << 30, device_bytes=1 << 30)
calculator = Calculator(device="cuda", resource_budget=budget)
hf = calculator.estimate_resources([h2]).requests[0]
tensor = tensor_resource_choices(program, compiler.target)
plan = plan_resources([hf, tensor.request], budget).require_feasible()
with ResourceSession(plan, {
    "hf": lambda selected: calculator.prepare_batch([h2], resource_plan=selected),
    "tensor": tensor.factory(compiler, cache),
}) as session:
    session.advance(0)
    hf_result = session.provider("hf").execute(strict=True)
    tensor_result = session.provider("tensor").execute(feeds)
```

Provider input controls must match the request; an explicit
global plan must also match any budget already configured on the calculator.

`ResourceSession.advance(phase)` closes expired owners before preparing newly
live owners. Each session owner must use one uniform interval across all its
buffers/candidates. Richer internal phase schedules remain available in the
planner and are implemented by the provider or separate owners.

## Allocation failures and observations

A factory can raise `ResourceAllocationError` with an allocator-identified
space after releasing partial allocations. The session closes its newly
created group, selects an untried enumerated alternative that reduces active
usage in that space, and records the failure/selection. It preserves already
live owners. Each selection is attempted once per phase, including alternating
host/device failures. Numerical errors are never allocation retries.

TensorIR allocates at this preparation boundary and supports automatic retries.
HF currently allocates native scientific buffers during `execute`, after its
Python prepared object exists. An HF execution failure preserves diagnostics
and returns/raises the failure; it does **not** roll back scientific state and
automatically replay the solve with a new plan. Callers can inspect enumerated
alternatives and explicitly prepare a new calculation. This boundary prevents
replaying successful neighbors or silently changing a partially executed fleet.

The native device ledger charges actual owned CUDA buffer capacities, enforces
the selected HF owner's numeric capacity, retains charges across warm calls,
and releases charges with native cache destruction. A shared global budget
also reserves TensorIR's separate allocations. The ledger excludes CUDA
context/modules, graphs, stacks, pool/page retention and library internals.
Those exclusions and any allowances are explicit in the plan. It is not a
process-wide physical-memory cap.

CPU diagnostics sample simultaneously owned integral/SCF/DIIS vector capacities
at iteration boundaries. They **exclude transient recurrence/eigensolver
allocations** and are not a complete allocator peak. Host execution is bounded
by the inventory/schedule, rather than a host allocator limiter. The standalone
CPU heap audit measures C++ allocations throughout preparation and execution
to check those broader bounds. CUDA TensorIR reports owned ndarray bytes,
native buffers and measured retained provider storage. Unknown CUDA OOM
placement stays unknown; only actual ledger evidence identifies device OOM.

Reproduction commands and production receipts are in
[`benchmarks/resource-planning`](../benchmarks/resource-planning/README.md).

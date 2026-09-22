# MP2 A1 native connection contract for E/D

Historical interface audit: the current coordinator subsequently authorized
the shared reference/provider/TensorIR connections described here. Current
implementation details are in [mp2.md](mp2.md); references below to locked or
missing interfaces describe the initial audit, not current authorization.

Part of #193 A1, not completion of acceptance A. Audited base:
`1e93c3cfa10afdf99ee26c70312bcfe533114331`. This is a proposal for the next
coordinated patch, not an implemented interface. E assigned B sole ownership
of the native MP2 adapter and authorized minimal public registration edits;
shared post-HF/tensor extensions still require E/D boundary review first.

## Existing chain and missing links

| Boundary | Delivered at this base | Required connection |
| --- | --- | --- |
| Public preparation | `methods::prepare_calculation` → registry → `PreparedCalculation`; `Capabilities::supported_properties` | Add an MP2 adapter and independent energy-only capability; reject forces at every property boundary before execution |
| HF reference | `scf::run_rhf`/`run_rhf_cuda` return `ScfResult` with density/scalars/forces, but no owned S/h/F/C/epsilon snapshot | Export the physical converged reference from the existing HF finalization rather than solve a second HF or copy a new post-HF stack |
| Validated ownership | Python `ReferenceSnapshot` checks immutable arrays, full identities, ordered 2/0 occupations, canonicality and SCF residuals | One shared native representation/validation adapter for the same contract, reusable by the Python development boundary and future C consumer |
| Initial exporter | `vibeqc_posthf_rhf_density_v1` plus `export_rhf` (12-AO cap, host Fock rebuild/canonicalization; CPU HF setup includes dense ERIs/forces) | A bounded native reference export or an explicitly separate, limited supplied-snapshot entry point; lifting the Python cap is insufficient |
| Conventional integrals | `posthf::RawSource::read`; CG10 `ConventionalProvider` and `plan_block` in Python | Native orchestration of the same source/transform contract, with a single source of budget arithmetic; do not independently port and diverge the planner |
| CUDA transforms | `posthf_cuda_create_v1/add_v1/pointer_v1/metrics_v1/destroy_v1` in a separately compiled private runtime | A shared declaration/ownership adapter to existing symbols and a native block-consumer loop; no replacement four-index transform |
| Energy kernels | A1 `energy_program`; CG09 `plan_cuda` → `compile_cuda` → generated `tensor_create/tensor_run/tensor_destroy` | Native orchestration of generated tile plans; a reviewed device-buffer input/stream contract or explicitly accounted bounded host exchange |

The CG09 current Python executor stages NumPy inputs; a device pointer from
CG10 cannot be passed as if it were that host-array ABI. The CPU eigensolver
helpers in `src/scf/rhf.cpp` are private, not an already exported reference API.

## Lifetime and budget contract

The prepared method owns an immutable reference generation and its private
provider/energy state; no borrowed pointer survives its owner or generation.
Changed geometry, basis, occupations, reference algorithm or precision forces
new state. A failed HF item supplies no snapshot; failures publish no partial
MP2 energy. Canonical denominator diagnostics precede integral reads and never
clamp a gap. All-electron real closed-shell RHF is the only enabled reference.

Compose the HF/export peak, persistent source/reference buffers, one provider
tile, detached staging, tensor arena, scalar accumulators and provider/library
allowances by phase. Reuse CG10 and CG09 estimates; a public budget must fit
the #203 method-level contract rather than introduce a competing global
planner. Account each provider handle's retained allowance. No full AO ERI,
full molecular T2 or nuclear Jacobian belongs in the energy consumer. A future
amplitude request needs its own size estimate and capability gate.

Permitted disclosed host stages for the proposed bounded route: reference
export/canonicalization if explicitly bounded and validated; existing CPU AO
value tiles; finite scalar diagnostics; explicitly bounded MO D2H/H2D staging
if the device handoff has not been delivered. Host staging is not permission
to run the entire MP2 contraction in a CPU reference library. The current
Python tile loop and scalar fold remain development behavior and are not
evidence for a complete native prepared method.

## Minimal staged patch request

1. **E/D decision needed:** expose shared owned reference/export and native
   consumption/planning adapters under `src/posthf/`, extracting/reusing the
   existing implementation; define CG09 device-input/stream semantics only if
   necessary. Specify which HF finalization symbols must be exposed before
   touching `src/scf/`. These are the still-locked shared interfaces.
2. **Then B:** add `src/methods/mp2_method.*` and the MP2 native consumer, using
   A1 equations. Use only the E-authorized minimal registry, public C/C++/Python
   metadata, property rejection and CMake source registration changes. PR #194
   is concurrently editing CMake; preserve unrelated lines.
3. **Then verification:** native CPU molecular setup, exact-C OS/SS, bounded
   GPU transforms/contraction/accumulation, unsupported properties, state and
   memory failures; real-device tests and sanitizer with explicit staging
   records. Freeze the new SHA for D before C optionally integrates it.

Do not add RI, open-shell, frozen-core, ECP or gradients here. #151/#179/#141/
#144/#143 remain later derivative integrations, not blanket energy blockers.

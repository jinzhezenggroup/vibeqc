# Occupied-factor CUDA exchange and force response

`VIBEQC_DF_EXCHANGE=auto` (also the unset default) selects occupied RI-K for
the qualified resident RHF domain: one system, 768 orbital and auxiliary AOs,
occupied rank 160, full host-raw J/K scratch, and NVIDIA GeForce RTX 5090
(`sm_120`). Other domains retain dense exchange. `dense` and `occupied` remain
explicit comparison overrides. This changes execution of the existing DF
exchange and complete analytic force model without changing its approximation.

For one spin, `D = w C C^T` with canonical occupation w=2 (RHF) or w=1 (UHF).
For each auxiliary slice compute `U = C^T L_row^T`, then accumulate
`K_row,column += w U_row^T U_column`. Host `OccupiedDensityFactor` snapshots
already contain `sqrt(w)*C` in row-major AO/occupied layout and use w=1 in the
CUDA product. Device SCF retains column-major C and applies w once at the
second product. Existing Fock consumers retain their RHF -1/2 and UHF -1
exchange prefactors. Empty spin rank produces zero K without a GEMM.

The same plan supports resident tensors, generated panels and compatibility
host-backed tiles. Full AO panels follow #282's capacity rebalance and reuse
the diagonal T before any column replacement. Tight row traversal retains
bounded column regeneration; its generated tile trace exposes that work.
The T, column and output matrices fit the original three/four tile buffers
because occupied rank never exceeds nbf. No additional three-center tensor is
allocated. Compatibility host uploads drain before overwriting pageable
staging; generated plans stay on their existing stream and permit capture.

The native fixed-density API checks the immutable factor's exact density
witness, spin, orbital/density generations and a process-unique physical plan
identity. Rebuilding geometry, basis or metric policy creates a new plan
identity. Matching dimensions or caller-provided generation labels alone
cannot authorize use. A missing or incompatible factor runs dense K for that
item while compatible neighbors retain factorized K. Factor upload borrows
the existing density-transpose staging; no allocation is needed for host B.

Every device SCF invocation starts with one dense iteration because imported
and warm densities have no trustworthy orbital factor. The iteration stores
the exact C that constructs the next D before the convergence update. Each
spin owns `batch*nbf*max_occupied` values plus generation controls. Inactive
systems retain both density and factor; eigensolver scratch is never borrowed
as persistent C. Occupation/policy changes rebuild captured GEMM shapes.
The first dense iteration counts against the original iteration limit, even
when that limit is one. Generation checks run before occupied K and at final
readback; a stale generation rejects the device result and preserves the
caller's established numerical recovery. Force evaluation receives only the
validated converged density and keeps its full metric/center/Pulay response.

Native and common resource ledgers reserve two full AO matrices for spin
factors plus generation flags for explicit occupied selection or a potential
768/768 batch-one automatic domain; actual
allocation uses the bucket's occupied ranks. Dense mode retains its previous
minimum-budget and residency boundaries. A native plan freezes this reservation
at creation and rejects occupied SCF before allocation if it reserved only dense
storage. Ordinary prepared batches rebuild the value/SCF plan on policy changes,
retaining their geometry response cache. Batches with a global `ResourceBudget`
freeze `VIBEQC_DF_EXCHANGE` in the resource identity: changing it after preparation
requires preparing a new batch and is rejected before native execution. The
fixed-density factor API needs no additional allocation and still borrows the
existing tiles independently of the SCF reservation.

`ri_k_occupied` traces report factor bytes, rank,
projection/exchange products and panel hits. Captured records describe graph
construction; `occupied_scf_provenance` separately reports executed iteration
counts, dense seeding and final generation validation. Uninstrumented complete
endpoints remain the performance selection gate.

## Exact occupied force response

`VIBEQC_DF_RESPONSE_SPACE=auto` (also unset) selects occupied response in the
same measured 768/768 RHF rank-160 device domain, with automatic resident
storage and the default generated shell schedule. Explicit panel storage and
diagnostic schedules preserve their original route. `dense` retains the full-AO
comparison; `occupied` requests factor validation on compatible resident plans.
Small systems keep dense automatic response.

The method passes its verified final-state token. The response owner checks
source identity, solve epoch, system, model, occupations, exact canonical device
density, and each device factor generation before borrowing C. Missing/stale
tokens, corrected determinants, external densities, unreserved plans and
unsupported factors keep dense response. UHF additionally verifies the exact
sum of its spin densities and admits both rank-squared projections together.

The response computes `T_Q=C^T A_Q C` and `U_P=sum_Q V_PQ T_Q` from raw
three-center values, preserving finite discarded metric directions. It feeds
at most 64 auxiliary AO pseudo-density matrices at a time to the existing
generated derivative consumers. It does not retain a full response-weight
tensor in the qualified domain. Projections, transformed projections and raw
values borrow the three already charged resident J/K tensors; the consumed
projection buffer becomes panel storage. Additional response workspace contains
four auxiliary matrices, three AO matrices, densities and auxiliary charges.
`DfGradientResources` reports the executed route and borrowed capacity.

The [derivation and lifetime note](../.agents/notes/implemented/performance/2026-09-15-occupied-df-response.md)
records RHF/UHF coefficients, metric response and rejected schedules.
The [qualification evidence](../benchmarks/results/issue377-379-df/README.md)
retains frozen-density policy comparisons, independent strict force gates,
component/work counters, reservation and priming costs. The automatic selector
is deliberately limited to that measured domain; it establishes no TZ/QZ,
other-device or COSX crossover. Memory diagnostics report charged capacity,
not a measured global GPU peak.

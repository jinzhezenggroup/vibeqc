# Bounded occupied-factor CUDA exchange (#284)

The dense RI-K implementation remains available. `VIBEQC_DF_EXCHANGE=occupied`
opts the device RHF/UHF SCF loop into the factor path; the initial default is
`dense` until complete fixed-K and endpoint measurements select a policy.
This changes execution of the existing DF exchange model, leaving J and the
complete analytic derivative model in their existing consumers.

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

Native and common resource ledgers conservatively reserve two full AO
matrices for spin factors plus generation flags, while actual allocation uses
the bucket's occupied ranks. `ri_k_occupied` traces report factor bytes, rank,
projection/exchange products and panel hits. Captured records describe graph
construction; `occupied_scf_provenance` separately reports executed iteration
counts, dense seeding and final generation validation. Uninstrumented complete
endpoints remain the performance selection gate.

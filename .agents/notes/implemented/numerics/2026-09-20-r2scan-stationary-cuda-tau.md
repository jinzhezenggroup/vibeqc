# Decision: lower r²SCAN stationary tau geometry through shared CUDA codegen

Status: implemented
Date: 2026-09-20

## Problem

The shared stationary-gradient plan and CPU consumer can now express the r²SCAN
tau geometric derivative, but the CUDA diagnostic remained deliberately
fail-closed. Its generated geometry artifact still selected only LDA/PBE with a
boolean and the runtime header consumed only rho/gradient point coefficients.
Promoting r²SCAN by calling the CPU derivative path, omitting tau, or adding a
method-specific CUDA force kernel would break the existing compiler ownership
and derivative contract.

## Decision

Extend the existing stationary CUDA lowering with the same explicit native
semilocal selector used by KS: 0=LDA, 1=PBE, 2=r²SCAN. Preserve the historical
`pbe=` spelling only as a compatibility frontend for existing tests/callers.

For selector 2, generate the r²SCAN production scalar DAG directly from the
audited compiler expression with `production=True`. This retains the native
spin-endpoint continuation and the existing total-density C2 tail policy instead
of silently using the interior-only reference expression.

The borrowed `GridTaskView` already carries the fixed feature layout
`rho, grad(rho), tau` for each spin. The stationary kernel consumes tau from
that live view, evaluates energy/rho/gradient/kinetic coefficients on device,
and feeds the fifth compact coefficient into the existing generated
`jet_pullback_program("mgga")`.

The kinetic coefficient has exactly the native meaning `vtau/2` for

```text
tau_s = 1/2 sum_mn D_s,mn grad(phi_m) . grad(phi_n).
```

It is multiplied by the quadrature weight exactly once before the generated AO
pullback. UKS consumes one coefficient per spin; restricted execution inherits
the existing two-spin half-density representation and therefore performs no
extra tau normalization in the CUDA gradient kernel.

AO/grid geometry uses second spatial AO jets for both GGA and meta-GGA. The
existing Becke/grid-response adjoint and the seven stationary source inventory
are unchanged. Integral, Pulay, Coulomb and nuclear derivative owners are not
duplicated.

## Rejected alternatives

- CPU fallback from the CUDA force endpoint: violates the qualified CUDA
  scientific-path boundary and hides device work.
- Reusing the PBE geometry artifact while dropping tau: gives an incomplete
  r²SCAN derivative.
- A handwritten r²SCAN force kernel: duplicates the compiler-owned semilocal
  pullback and creates method-specific derivative ownership.
- Using the default interior r²SCAN expression in CUDA: differs from the native
  SCF production endpoint near spin/tail boundaries.

## Invariants

- LDA/PBE generated execution remains numerically/scientifically equivalent through
  the compatibility selector; source text may change as shared plumbing evolves.
- r²SCAN point coefficients come from the audited production DAG.
- `vtau/2` is applied exactly once.
- All CUDA AO geometry remains generated from the common jet pullback graph.
- Grid topology, source coverage, live snapshot identity, resource admission,
  borrowed-stream lifetime and final publication checks remain unchanged.
- r²SCAN ECP forces do not inherit support from this all-electron slice.
- Public r²SCAN force capability must remain off until real-device analytic and
  reconverged finite-difference qualification passes.

## Evidence

Device-free source-generation tests cover selectors 0/1/2, deterministic
generation, the five-coefficient meta-GGA pullback and legacy PBE-flag
compatibility. The opt-in RTX/Slurm suite adds r²SCAN RKS and UKS comparison
against an independent PySCF/Libxc analytic gradient on the identical explicit
grid, plus a fully reconverged r²SCAN RKS directional finite difference that is
sensitive to a missing or duplicated tau contribution.

Exact-head CI/device evidence is recorded by the stacked PR before capability
promotion.

## Consequences

The stationary CUDA compiler has one semilocal family path for LDA, GGA and
tau-dependent meta-GGA geometry. A later public r²SCAN force promotion can reuse
this qualified diagnostic instead of introducing another scientific
implementation. The larger response/Hessian problem remains separate.

## Revisit when

Revisit if a new semilocal ingredient requires an AO operator beyond the current
rho/gradient/tau compact bilinears, if production r²SCAN tail semantics change,
or if real-device qualification reveals a schedule/resource issue that requires
a distinct generated schedule rather than different mathematics.

## References

- #164
- #163
- #396
- #620
- #646
- `.agents/notes/implemented/numerics/2026-09-20-r2scan-public-ks.md`
- `.agents/notes/implemented/numerics/2026-09-20-r2scan-stationary-tau-gradient.md`

Agent: ChatGPT
Model: GPT-5.6 Sol

# Automatic Libxc capability claims

The bulk Libxc importer has a machine-readable qualification layer in
`vibeqc_compiler.xc.libxc_bulk_capabilities`.

It deliberately distinguishes **compiler/pointwise qualification** from
**public method admission**.  A registration returned here has:

- lowered through the common Maple frontend into the canonical Graph;
- both polarized and unpolarized layouts covered by independent PySCF
  2.14.0 / Libxc 7.0.0 reference fixtures;
- energy, first physical-feature derivatives (`vxc`) and packed second
  derivatives (`fxc`) checked on the declared
  `libxc-bulk-interior/v1` domain;
- C and CUDA source emitters available from the common Graph backend.

It does **not** automatically claim compiled CPU/CUDA binaries, GPU runtime,
vacuum/tail/fully-polarized production continuations, molecular SCF,
nuclear forces, response properties, or a public Calculator method.

## Query

```python
from vibeqc_compiler.xc.libxc_bulk_capabilities import (
    available_capabilities,
    claimable_functionals,
    functional_capability,
)

names = claimable_functionals()  # current pointwise-validated inventory
pbe_sol = functional_capability("GGA_X_PBE_SOL")
print(pbe_sol.to_payload())
```

The default claim level is `pointwise-validated`.  The lower
`graph-imported` level is also queryable.  Requests for runtime, SCF or
public-method levels fail explicitly instead of silently promoting a
registration.

## Admission path

This registry is the first automatic gate under #744.  Later gates should add
evidence monotonically rather than replacing this boundary:

```text
Libxc registration
  -> graph-imported
  -> pointwise-validated        [this registry]
  -> compiled CPU/CUDA
  -> production-domain
  -> molecular SCF
  -> forces / response
  -> public MethodIR/Calculator admission
```

A composite or hybrid method must additionally prove its MethodIR composition
and the corresponding exchange/nonlocal/correction providers.  Therefore a
semilocal component becoming pointwise-validated never, by itself, makes a
hybrid or range-separated method public.

The invariant is enforced in CI: the capability inventory must be exactly the
bulk imported inventory, and every advertised pointwise claim must have both
spin reference fixtures.  The existing bulk numerical test evaluates all of
those fixtures against the generated Graph through energy, `vxc`, and
`fxc`.

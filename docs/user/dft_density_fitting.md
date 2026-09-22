# Density-fitted DFT energy interface

`Calculator(method="pbe-rks", device="cuda", density_fitting="auto",
auxiliary_basis="def2-svp")` selects the shared native DF Coulomb provider for
KS energy calculations. `auto` follows the calculation backend; explicit
`cpu`/`cuda` DF selections must match `device`. Omitting the auxiliary basis
uses the orbital basis as the auxiliary basis, as in the existing HF interface;
choose an appropriate fitting basis for scientific production calculations.

CPU supports the existing local/semilocal RKS/UKS methods and full-range global
hybrids (DF-JK). CUDA supports LDA, PBE and r2SCAN RKS/UKS (DF-J). CUDA hybrids,
range-separated/nonlocal DF compositions, ECP DF and automatic mixed precision
are rejected. This interface does not change the selected functional or grid.

```python
from vibeqc import Calculator

calc = Calculator(method="pbe-rks", basis="sto-3g", device="cuda",
                  density_fitting="auto", auxiliary_basis="def2-svp")
result = calc.singlepoint([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
                          properties=("energy",))
```

The prepared owner copies auxiliary shells before the public descriptor is
released. Batch geometry changes rebind both orbital and auxiliary centers.
CUDA iterations enqueue DF J and XC on the DF owner's stream, retaining density
and Fock matrices on the device; no CPU integral/reference retry is selected.
The ordinary host-controlled KS convergence loop is used for DF. The opt-in
multi-iteration direct-J solver region remains restricted to direct J.

`density_fitting_relative_threshold` controls metric rank selection.
`density_fitting_memory_budget_bytes` bounds the native CUDA DF provider's
explicit source/value storage through its existing bounded tile planner; it
is not a cap on the full KS calculation. Infeasible budgets fail explicitly.
Prepared CUDA batches expose the provider's metric diagnostics. Whole-KS
`estimate_resources`/resource-plan admission is rejected for DF until its
combined inventory is qualified; conventional inventories must not describe DF.

DF energy capability excludes forces. The conventional derivative-snapshot
export is also rejected natively because it lacks auxiliary/metric response.

`tests/python/test_dft_df_public.py` compares independently converged PySCF
energies with copied orbital/auxiliary primitives and identical quadrature
(absolute energy gate `1e-8 Eh`, physical residual below `1e-9`). GPU acceptance
requires `VIBEQC_DFT_CUDA_TEST=1` and a scheduler-allocated CUDA device.

See the [ownership note](../../.agents/notes/implemented/architecture/2026-09-22-public-df-ks-provider.md)
for the retained boundaries.

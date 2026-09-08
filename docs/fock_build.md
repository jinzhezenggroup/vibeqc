# Shared Fock construction contract

The internal C++ boundary in `src/scf/fock_build.hpp` is the first HF-direct
slice of issue #202. It separates a method's mathematical J/K request from
its resolved execution strategy. The public C/Python descriptors and their
legacy density-fitting defaults remain unchanged.

## Densities, operators, and coefficients

`FockBuildSpec` version 1 records spin, requested derivative order, and
independent Coulomb and exchange terms. Each term records presence, a signed
Fock coefficient, the full-/short-/long-range operator and range parameter,
and the exact or density-fitted approximation.

All raw matrices are FP64, row-major, in the caller's public AO representation.
The exact CPU consumer accepts nonsymmetric finite density matrices without
silently symmetrizing them. ERIs use chemists' `(ij|kl)` ordering:

- `J_ij = sum_kl D_total,kl (ij|kl)`.
- `K_spin,ij = sum_kl D_spin,kl (ik|jl)`.
- Restricted density includes double occupation: `F = H + J[D] - 0.5 K[D]`.
- Unrestricted densities have unit occupation:
  `F_alpha = H + J[D_alpha+D_beta] - K[D_alpha]`, and similarly for beta.

`build_exact_direct_jk` returns unscaled J/K matrices.
`assemble_fock` applies the requested coefficients exactly once. An absent
term has no raw output allocation. Presence and a zero coefficient are
different requests: a present zero-coefficient term still requests its raw
matrix.

For a fixed density, `contract_exact_direct_energy_derivative` uses the same
resolved terms and coefficients on derivative ERIs:

```text
RHF: dE_2e = 0.5 D : (c_J dJ + c_K dK)
UHF: dE_2e = 0.5 c_J D_total : dJ
             + 0.5 c_K (D_alpha : dK_alpha + D_beta : dK_beta)
```

One-electron, overlap/Pulay, and nuclear-repulsion derivatives remain outside
this two-electron consumer. The force is the negative total derivative.

## Resolution and supported execution

`resolve_fock_build` preflights the complete request before any contraction.
Unsupported versions, spin layouts, operators, derivative orders, or provider
combinations fail explicitly. Merely having an enum value does not make that
operator executable.

| Consumer | Executable scope |
| --- | --- |
| Exact CPU raw J/K | Full range, RHF/UHF density conventions, independently present J/K, arbitrary finite coefficients, values and first derivatives |
| Existing CPU HF solver | Complete standard RHF/UHF pair and first derivatives |
| CUDA fused direct HF | Complete standard RHF/UHF pair and first derivatives; existing fused kernels, device storage, streams, and graphs |
| Legacy CPU/CUDA DF adapter | Existing standard HF DF pair, using existing DF implementations |
| SR/LR, mixed exact/DF terms, independent CUDA J-only/K-only | Rejected; follow-up integrations required |

The CPU reference consumes already materialized four-center integral and
derivative tensors. This refactor does not make that algorithm bounded or
on-demand. CUDA keeps its existing persistent, quartet, and bounded execution
paths; independent mathematical terms do not require separate GPU launches.

`FockProviderCapabilities` reports these implementation limits. It does not
replace the existing system/basis preflight or probe whether a CUDA device is
available. Existing basis limits and backend initialization still apply.

## Prepared state and compatibility

A prepared HF calculation resolves its request before execution and retains
it in `ScfOptions::resolved_fock_build`. CPU iteration, final-density rebuilds,
and analytic two-electron forces consume the same immutable resolved value.
CUDA direct bucket compatibility also includes this value.

The mathematical request (`spec`) is distinct from backend/schedule fields.
Screening, precision, and the DF metric cutoff are explicit resolved fields;
an unused exact-provider metric cutoff is canonicalized away. Zero screening
retains the existing unscreened reference semantics used by post-HF exports. Geometry,
orbital basis, auxiliary basis, and device-resource ownership remain in the
enclosing existing prepared plans and their compatibility checks.

The old DF selector still first chooses the DF approximation, then selects
its backend. `AUTO` does not authorize changing exact into DF or DF into exact.
The legacy DF adapter is preserved for compatibility; its presence is not a
claim that independent/mixed DF providers have been migrated.

## Validation and remaining issue scope

`vibeqc_fock_build_tests` uses independent pinned two-AO J/K values, distinct
alpha/beta densities, nonsymmetric densities, and non-unit coefficients. It
checks absent terms, derivatives, preflight failures, and resolved identities.
The existing RHF/UHF, batch, density-fitting, Cartesian, and spherical suites
exercise the real consumers.

This slice does not complete #202: independent DF selection, broader
derivative/provider invalidation, and a real DFT consumer remain subsequent
integration work. A fixed-density raw-J/K fixture does not count as a DFT SCF
consumer. Performance conclusions require matched, synchronized endpoint
measurements on explicitly identified hardware.

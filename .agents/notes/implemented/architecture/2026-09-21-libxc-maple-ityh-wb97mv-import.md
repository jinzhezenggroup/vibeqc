# Decision: qualify ITYH and omegaB97M-V through the common Libxc Maple importer

Status: implemented
Date: 2026-09-21
Parent: #743 / #739

## Scope

Advance the common Libxc 7.0.0 frontend from `libxc-maple-graph/v8` to `v9`
without adding a general Maple runtime or changing production XC ownership.

The new common semantics are the source-evidenced pieces needed by the remaining
ITYH/range-separated and omegaB97M-V semilocal sources:

- Libxc 3D `RS_FACTOR`, `erf`, `n_spin`, density/zeta screening and `z_thr`;
- screened non-separable GGA exchange and Stoll parallel/perpendicular
  correlation decomposition;
- bounded nested external parameter arrays;
- `add(body, i=N..n)` when the upper bound resolves to a compile-time integer
  within the existing bounded reduction limit;
- the exact Libxc 7.0.0 `enforce_smooth_lr` specialization used by
  `attenuation_erf0` at cutoff 1.35/order 16.

The smooth-LR path is not a Maple `proc`/`series` interpreter. The admitted
specialization reproduces the eight even inverse-power terms emitted by the
pinned Libxc 7.0.0 generator and fails closed for another helper, cutoff or
order.

## Scientific contracts

The imported graphs use Libxc 7.0.0 thresholds:

- `zeta_threshold = DBL_EPSILON`;
- `GGA_X_ITYH` density threshold `1e-14`;
- `HYB_MGGA_XC_WB97M_V` density threshold `1e-13`.

omegaB97M-V external arrays and `omega=0.3` are bound from the pinned
`hyb_mgga_xc_wb97mv.c` parameter owner. The fourth spin marker passed by the
7.0.0 `b97mv.mpl` call to `lda_stoll_par` is accepted as the source-evidenced
unused extra argument; no generic variadic Maple-call rule is introduced.

## Evidence

For both unpolarized and polarized feature layouts:

- imported ITYH E/vxc/packed-fxc agrees with the retained audited ITYH DAG to
  machine precision;
- imported omegaB97M-V E/vxc/packed-fxc agrees with the retained audited DAG to
  below 5e-15 on representative interior points;
- the retained independent Libxc omegaB97M-V oracle is recovered directly;
- full polarized Hessian graphs lower through the existing Scalar C and CUDA
  emitters;
- dedicated tests cover both direct-erf and large-a smooth-LR branches and the
  bounded symbolic-add/nested-binding path.

The full related importer/RSH/WB97M-V regression set passed 118 tests before the
additional v9 helper tests. Source-registry offline verification also passes
after rebinding the importer/generator digest and semantic identity to v9.

Production cutover remains owned by #744 and handwritten retirement by #745.

Agent: ChatGPT
Model: GPT-5.6 Sol

## Review hardening before v9 admission

The large-a specialization now validates the whitespace-normalized definitions
of attenuation_erf0 and all three att_erf_aux helpers against the verified
Libxc 7.0.0 source. A same-named changed formula must not inherit a fixed series
that belongs to different mathematics. Five mutation cases failed before the
repair and are rejected afterward; whitespace-only edits remain accepted.
No admitted equation, series coefficient, point threshold or tolerance changed.
The parser also carries the bounded structured-binding type, and scalar/
function branches retain their existing checked types without type ignores.

The complete Maple selection passes 223 tests. Registry freshness is rebound
to the reviewed importer bytes without changing upstream file identities.
This is CPU/importer validation, not a new real-GPU or public-method promotion.

Agent: ChatGPT
Model: GPT-6 Astra Pro

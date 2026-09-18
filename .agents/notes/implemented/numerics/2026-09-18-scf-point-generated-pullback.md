# Decision: bind native SCF point derivatives to the generated XC geometry pullback

Status: implemented
Date: 2026-09-18

## Problem

The live native KS handoff established by
[the stationary native handoff note](2026-09-16-stationary-native-handoff.md)
correctly rejected generated XC geometry because the generated
`interior-v1` scalar model did not share the native SCF tail/spin domain. That
protected ownership, but it also left #163-A incomplete: there was no audited
way to differentiate the exact scalar XC model that produced the stationary
native state.

The implementation now has an exact pointwise derivative bridge for the native
`semilocal-scaled-v1/pbe-spin-c2-1e-18` SCF domain. The ownership decision
therefore needs to distinguish the scalar point model from the generated
density/AO geometric chain instead of treating all generated geometry as a
different energy model.

## Decision

Keep stationary authorization with the live #162 native KS owner/token and keep
the native SCF domain identity unchanged. A private CPU point bridge calls the
same native LDA/PBE point evaluator used by the SCF model and returns per-point
energy, `dE/d rho_s`, and, for PBE, `dE/d grad(rho_s)` in the physical
two-spin Cartesian convention.

The compiler-generated XC contraction owns only the deterministic pullback from
those audited Cartesian coefficients through density/AO jets to AO-centre,
grid-point, and quadrature-weight partials. It does not reinterpret or recreate
the scalar XC model. RKS combines the two spin derivatives only at this
pullback boundary, consistently with `D_alpha = D_beta = D/2`.

This supersedes only the earlier requirement that a native SCF state must fail
all generated XC geometry binding. The live-native authorization, replay and
lifetime rules from the 2026-09-16 note remain in force. The separate
`interior-v1` diagnostic model is still not a native-state authorization
domain.

## Rejected alternatives

- Relabeling the native SCF state as `interior-v1` would differentiate a
  different scalar model, especially in scaled tails and spin endpoint cases.
- Reimplementing the native SCF point algebra independently in the compiler
  would create two sources of truth for the production scalar model.
- Authorizing geometry from copied arrays without the live native token would
  reopen the stale/replayed-state problem solved by the native handoff.
- Treating the point bridge as a complete molecular gradient would blur the
  missing grid/partition response, one-electron, Hartree, Pulay, nuclear, and
  later response terms.

## Invariants

- Every stationary bind validates the original live native owner/token and
  rechecks it after derivative evaluation.
- The point bridge differentiates exactly the native SCF domain identified by
  `semilocal-scaled-v1/pbe-spin-c2-1e-18`; a model/domain identity change must
  not silently reuse this authorization.
- The compiler pullback consumes point coefficients and owns only the
  density/AO geometric chain.
- `interior-v1` remains a separate fixed-density diagnostic contract and
  cannot authorize a native state.
- This bridge is private and CPU-side; #163-A does not expose public DFT forces
  or claim a complete molecular gradient.

## Evidence

`tests/python/test_xc_scf_point_bridge.py` checks the private bridge against
all 97 independent Libxc/mpmath SCF-domain reference rows for LDA and PBE,
including scaled-tail and spin-domain cases, and rejects invalid domains.

`tests/python/test_dft_stationary_gradient.py` independently displaces AO
centres, grid points, and weights and compares multistep scalar SCF-domain
finite differences with the generated Cartesian pullback for LDA/PBE RKS/UKS.
It also cross-checks the coefficient adapter against the existing generated
`interior-v1` geometry path when both consume the same point differential.

`tests/python/test_dft_stationary_native.py` exercises real native LDA/PBE
RKS/UKS snapshots, preserves stale/replay/relabel rejection, and checks rigid
translation consistency after binding the SCF-domain bridge.

## Consequences

#163-A now has a stationary XC geometry contract without changing the scalar
SCF energy model or weakening native ownership. The split also keeps the
compiler reusable: a point provider can own scalar XC semantics while generated
code owns the geometric chain.

The result is intentionally not a public force implementation. Atom-centred
grid/partition motion, remaining energy terms and Pulay assembly, CUDA
derivative lowering, CPKS/response, and public DFT force capability remain
separate #163 B/C or later work.

## Revisit when

Reopen this split if the production SCF point-domain identity changes, the
compiler becomes the single exact owner of that same SCF scalar model, CUDA
gradient lowering removes the CPU point bridge, or complete moving-grid/public
molecular gradients require a different ownership boundary.

## References

- Issue #162: native LDA/PBE RKS/UKS SCF ownership.
- Issue #163, slice A: stationary XC geometry contract.
- PR #455: SCF point bridge plus generated Cartesian pullback.
- [Current method status](../../../../docs/methods.md).
- [Prior live-native handoff decision](2026-09-16-stationary-native-handoff.md).

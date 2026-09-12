# Occupied-factor RI-K contract (#284)

The existing metric-orthonormalized tensor is
`L[i,j,Q] = sum_P (ij|P) X[P,Q]`, with auxiliary-contiguous storage.
For one positive semidefinite spin density, let
`B[j,o] = C[j,o] sqrt(occupation[o])` and `D = B B^T`. Then

```text
T[i,o,Q] = sum_j L[i,j,Q] B[j,o]
K[i,k]   = sum_Q sum_o T[i,o,Q] T[k,o,Q]
         = sum_Q sum_j sum_l L[i,j,Q] D[j,l] L[k,l,Q].
```

This is the existing dense-density RI-K equation with a shorter contraction
dimension. The CPU reference processes one Q at a time with `NAO*rank` scratch.
It adds no full tensor and leaves J unchanged. RHF uses occupation 2 in B and
the existing `J - K/2` assembly. UHF uses separate alpha/beta occupation-1
factors and `J[Dalpha+Dbeta] - K[Dspin]`. No spin coefficient is applied twice.

`OccupiedDensityFactor` owns immutable factors, occupations and a density
witness. Basis/reference identities and orbital/density generation IDs are
required and nonzero. An eligible consumer checks every identity field, the
spin channel, shape and exact density witness. The witness uses the established
SCF `occupation*C*C` evaluation order; forming B with a square root can change
roundoff, so the RI-K comparison uses the existing numerical tolerance.

An absent factor, an external or response density, a damped/mixed density, a
different source, a stale generation or a missing spin channel requests the
dense fallback. Equal dimensions and caller labels alone do not authorize
reuse. There is no density diagonalization to manufacture a factor. Fractional
or negative occupations are rejected in this first contract. Empty spin blocks
use rank zero and produce zero K.

The native CPU test compares against the existing dense RI-K at rank zero,
one, two and full rank for RHF and both UHF spin channels. It checks the SCF
density constructor independently and exercises each identity mismatch, a
same-label density change, missing factors and invalid occupations.

This slice defines equations, host provenance and the reference only. CUDA
factor upload/lifetime, occupied GEMMs, SCF generation wiring and #206/#246
endpoint selection are pending. It makes no performance or completed-SCF claim.

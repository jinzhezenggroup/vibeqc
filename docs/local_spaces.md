# Restricted local occupied and pair spaces

`tools.vibeqc_local_cc` builds experimental local spaces from #147's immutable
canonical RHF reference and bounded integral providers. It implements Mulliken
Pipek–Mezey occupied localization, projected AO virtual domains, and restricted
pair-natural orbitals (PNOs). It does not register local MP2, local CCSD(T), or
local-correlation gradients as production methods.

## Mathematical conventions

Occupied localization maximizes `sum(A,i) q[A,i]**2`, where `q` contains spatial
orbital Mulliken atomic populations. Symmetrized atomic population operators
resolve the occupied identity in the actual AO overlap metric. Jacobi rotations
optimize each occupied pair analytically. The result records the initial/final
objective, gradient, sweeps, timing, atom partition and occupied rotation. Failure
to satisfy the gradient gate raises; a converged local maximum is not declared
the unique global solution.

For canonical occupied coefficients `Co`, virtual coefficients `Cv`, and
localization rotation `U`, the local occupied coefficients are `Co @ U`.
The complete occupied Fock `U.T @ diag(eps_occ) @ U` is retained. Localized
occupieds are noncanonical, so diagonal local denominators would change MP2.
This implementation instead rotates the independently audited canonical MP2
amplitudes exactly:

```text
T_local[i,j,a,b] = sum(k,l) U[k,i] U[l,j] T_canonical[k,l,a,b]
G_local[i,j,a,b] = sum(k,l) U[k,i] U[l,j] (k a|l b)
```

A selected AO domain `E` is projected out of the occupied metric. In canonical
virtual coordinates its columns are `X = Cv.T @ S @ E`. Spectral normalization
of `X.T @ X` removes unsupported directions under explicit absolute/relative
cutoffs. Duplicate or occupied-contaminated input directions produce diagnosed
rank loss, not invented virtual orbitals. All AOs recover the complete virtual
space for a valid full-rank reference.

For each unordered occupied pair `i<=j`, project its amplitude into the chosen
orthonormal virtual domain. The restricted PNO density is

```text
tildeT = 2*T - T.T
D = (tildeT.T @ T + tildeT @ T.T) / (1 + delta_ij)
```

Equivalently, for `T=S+A` with symmetric `S` and antisymmetric `A`, the numerator
is `2*S@S + 6*A.T@A`, which is positive semidefinite. PNO eigenvalues are spatial
pair-density occupations with this convention; they are not energy-error bounds.
Adjacent eigenvalue clusters are retained together if any member passes the
explicit occupation threshold. Near-threshold clusters are flagged as rank
crossings. `keep_full_space=True` also retains zero-occupation directions, which
is necessary for an unambiguous full-space recovery gate.

For PNO columns `Q` in canonical virtual coordinates, the recorded pair energy is

```text
t = Q.T @ T_local[i,j] @ Q
g = Q.T @ G_local[i,j] @ Q
E_pair = (2 - delta_ij) * sum(t * (2*g - g.T))
```

The multiplicity accounts for the two ordered off-diagonal occupied pairs.
Full virtual retention recovers the canonical MP2 energy and amplitudes for the
same Hamiltonian. With truncation, this is a projection of canonical MP2 into
pair spaces. It does not solve the coupled approximate local-MP2 equations and
is not marketed as DLPNO-MP2 or another established local method.

## Ownership, identities and budgets

Records own immutable FP64 arrays. Parent reference, Hamiltonian, localization,
domain, pair labels, spectra, thresholds and ranks remain explicit. Projectors
describe mathematical subspaces; `gauge_identity` additionally records the basis
used for amplitude coordinates. Exact numeric hashes are reproducibility keys;
compare projectors numerically when testing equivalent eigensolver gauges.
`PairSpace.overlap(other)` supplies the rectangular `Q_pair.T @ Q_other` needed
for future pair-coupled equations. Changed parent states cannot reuse overlaps or
amplitudes based only on equal dimensions.

`build_local_mp2` reserves its numeric storage first and passes the remaining
budget to a fresh conventional or DF provider. Only one localized pair transform
is formed at a time; retained pair spaces and amplitudes are counted. The
provider supplies bounded native AO tiles and never creates a molecular AO
four-index tensor. Small-denominator failure propagates from audited canonical
MP2 without clipping. Unfrozen, real, closed-shell references are required.

The initial local-space consumer runs on CPU with explicit NumPy transforms.
Budgets cover numeric arrays and provider reservations; Python/interpreter and
library allocator overhead are outside that scope. Dense canonical MP2
amplitudes remain necessary for this prototype and are budgeted. This is not
yet a linear-scaling local-correlation implementation.

```python
from tools.vibeqc_local_cc.localization import localize_occupied
from tools.vibeqc_local_cc.spaces import projected_virtual_space
from tools.vibeqc_local_cc.mp2 import build_local_mp2

# snapshot and source are validated #147 objects for the same Hamiltonian;
# ao_atoms contains the physical atom owner of every AO, in source order.
localized = localize_occupied(snapshot, ao_atoms)
domain = projected_virtual_space(snapshot)
result = build_local_mp2(
    snapshot, source, localized, domain,
    occupation_threshold=1e-7, budget_bytes=128 << 20,
)
print(result.observed_energy_difference)
```

Supply the matching `metric=...` for a DF reference. The provider validates the
exact auxiliary metric identity. No conventional/DF substitution is implicit.

Localization/domain response and PNO-rank derivatives are **unsupported**.
Even away from rank crossings, a force cannot omit that response. The
[benchmark report](../benchmarks/results/local-spaces-182/README.md) gives
independent references, full-space recovery, truncation errors and measured costs.

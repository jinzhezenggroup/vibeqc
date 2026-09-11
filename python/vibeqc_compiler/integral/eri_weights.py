"""Exact unscreened external-weight conventions for real four-center ERIs.

Ordered weights W[i,j,k,l] are arbitrary scalars, such as a correlated 2-RDM
contribution; no HF factorization or permutation symmetry is assumed. A unique
AO quartet consumes the sum of all distinct orbit weights. These callbacks
read a bounded caller-owned tile/cache and never allocate a molecular N**4
tensor. Folding is valid for physical-atom derivatives; permuting a raw
shell-center derivative also requires permuting its center slot.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import fsum, isfinite, sqrt

from .shell_signature import checked_index

Quartet = tuple[int, int, int, int]


def _quartet(indices: Sequence[int]) -> Quartet:
    values = tuple(indices)
    if len(values) != 4:
        raise ValueError("ERI weights require four AO indices")
    for index in values:
        checked_index(index, "AO index")
    return values


def canonical_eri_indices(indices: Sequence[int]) -> Quartet:
    """Order each AO pair and the pair pair without losing equality cases."""
    i, j, k, l = _quartet(indices)
    first, second = (
        tuple(sorted((i, j), reverse=True)),
        tuple(sorted((k, l), reverse=True)),
    )
    if second > first:
        first, second = second, first
    return (*first, *second)


def eri_weight_orbit(indices: Sequence[int]) -> tuple[Quartet, ...]:
    """Return each distinct chemists' ERI index tuple once in stable order."""
    i, j, k, l = _quartet(indices)
    first = ((i, j),) if i == j else ((i, j), (j, i))
    second = ((k, l),) if k == l else ((k, l), (l, k))
    orbit = {}
    for a, b in first:
        for c, d in second:
            orbit[(a, b, c, d)] = None
            orbit[(c, d, a, b)] = None
    return tuple(orbit)


def _finite_weight(value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError("external ERI weights must be finite")
    return value


def fold_dense_eri_weight(
    provider: Callable[[Quartet], float], indices: Sequence[int]
) -> float:
    """Fold a full ordered W orbit; diagonals and identical pairs occur once.

    Each requested orbit entry must be available from the provider even if it
    lies outside the output tile. Returning only one weight and multiplying by
    eight is incorrect unless all eight entries exist and are identical.
    No density bound or magnitude threshold screens arbitrary weights.
    """
    return _finite_weight(
        fsum(_finite_weight(provider(q)) for q in eri_weight_orbit(indices))
    )


def normalized_pair_index(i: int, j: int) -> tuple[int, float]:
    """Return triangular pair index and its svec factor (1 or sqrt(2))."""
    checked_index(i, "AO index")
    checked_index(j, "AO index")
    i, j = max(i, j), min(i, j)
    index = i * (i + 1) // 2 + j
    checked_index(index, "normalized AO-pair index")
    return index, 1.0 if i == j else sqrt(2.0)


def fold_normalized_pair_weight(
    provider: Callable[[int, int], float], indices: Sequence[int]
) -> float:
    """Fold weights Z[I,J] for T[I,J]=s[I]*s[J]*(ij|kl).

    Z is a full matrix over triangular AO pairs, with no pair-exchange symmetry
    assumption. The scalar is sum(I,J) Z[I,J]*T[I,J]. A unique pair-pair entry
    consumes Z[I,J]+Z[J,I] off diagonal and Z[I,I] once on the diagonal,
    multiplied by the two svec factors. There are no additional AO-orbit factors.
    """
    i, j, k, l = _quartet(indices)
    first, first_scale = normalized_pair_index(i, j)
    second, second_scale = normalized_pair_index(k, l)
    weights = [_finite_weight(provider(first, second))]
    if first != second:
        weights.append(_finite_weight(provider(second, first)))
    return _finite_weight(first_scale * second_scale * fsum(weights))


def hf_eri_weight(
    indices: Sequence[int],
    total_density: Callable[[tuple[int, int]], float],
    *,
    spin_densities: tuple[
        Callable[[tuple[int, int]], float], Callable[[tuple[int, int]], float]
    ]
    | None = None,
) -> float:
    """Explicit HF adapter returning an ordered electronic-energy weight.

    RHF takes the spin-summed density. UHF additionally takes alpha and beta
    densities whose sum is the supplied total; consistency belongs to the
    caller. Force output applies minus the nuclear derivative separately.
    Generic external-weight execution never calls this adapter implicitly.
    """
    i, j, k, l = _quartet(indices)
    coulomb = 0.5 * total_density((i, j)) * total_density((k, l))
    if spin_densities is None:
        exchange = 0.25 * total_density((i, k)) * total_density((j, l))
    else:
        if len(spin_densities) != 2:
            raise ValueError("UHF weights require alpha and beta densities")
        exchange = 0.5 * fsum(d((i, k)) * d((j, l)) for d in spin_densities)
    return _finite_weight(coulomb - exchange)

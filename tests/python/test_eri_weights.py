"""External ERI weights must preserve exact orbits without HF assumptions."""

import itertools
import math

import numpy as np
import pytest
from vibeqc_compiler.integral.eri_weights import (
    canonical_eri_indices,
    eri_weight_orbit,
    fold_dense_eri_weight,
    fold_normalized_pair_weight,
    hf_eri_weight,
    normalized_pair_index,
)


def independent_orbit(indices):
    i, j, k, l = indices
    return {
        (i, j, k, l),
        (j, i, k, l),
        (i, j, l, k),
        (j, i, l, k),
        (k, l, i, j),
        (l, k, i, j),
        (k, l, j, i),
        (l, k, j, i),
    }


def test_every_ao_equality_pattern_has_exact_dense_weight_multiplicity():
    rng = np.random.default_rng(144)
    weights = rng.normal(size=(4,) * 4)
    for indices in itertools.product(range(4), repeat=4):
        orbit = independent_orbit(indices)
        assert set(eri_weight_orbit(indices)) == orbit
        canonical = canonical_eri_indices(indices)
        assert all(canonical_eri_indices(q) == canonical for q in orbit)
        assert fold_dense_eri_weight(lambda q: weights[q], indices) == pytest.approx(
            math.fsum(weights[q] for q in orbit), abs=1e-15
        )
    assert len(eri_weight_orbit((0, 0, 0, 0))) == 1
    assert len(eri_weight_orbit((1, 1, 0, 0))) == 2
    assert len(eri_weight_orbit((1, 0, 1, 0))) == 4
    assert len(eri_weight_orbit((3, 2, 1, 0))) == 8


def test_arbitrary_weight_full_and_folded_gradient_scalars_agree():
    rng = np.random.default_rng(17)
    weights = rng.normal(size=(3,) * 4)
    # An independent symmetric integral response has one atomic gradient per
    # unique orbit. The weights deliberately have none of those symmetries.
    gradients = {}
    full = np.zeros(9)
    for q in itertools.product(range(3), repeat=4):
        key = min(independent_orbit(q))
        if key not in gradients:
            gradients[key] = rng.normal(size=9)
        full += weights[q] * gradients[key]
    folded = np.zeros(9)
    for q, gradient in gradients.items():
        folded += fold_dense_eri_weight(lambda i: weights[i], q) * gradient
    np.testing.assert_allclose(folded, full, atol=2e-14, rtol=2e-14)


def test_normalized_pair_matrix_uses_svec_factors_and_pair_exchange_once():
    rng = np.random.default_rng(91)
    # Full pair matrix need not be symmetric. T[I,J]=s[I]*s[J]*(ij|kl),
    # s[ij]=sqrt(2) off diagonal, one on diagonal.
    pair_weights = rng.normal(size=(10, 10))

    def dense_weight(q):
        i, j, k, l = q
        first, first_scale = normalized_pair_index(i, j)
        second, second_scale = normalized_pair_index(k, l)
        return pair_weights[first, second] / (first_scale * second_scale)

    for q in itertools.product(range(4), repeat=4):
        assert fold_normalized_pair_weight(
            lambda i, j: pair_weights[i, j], q
        ) == pytest.approx(
            math.fsum(dense_weight(i) for i in independent_orbit(q)),
            abs=2e-15,
            rel=2e-15,
        )


def test_hf_adapter_is_explicit_and_recovers_existing_rhf_uhf_energy_weights():
    rng = np.random.default_rng(12)
    alpha = rng.normal(size=(4, 4))
    alpha += alpha.T
    beta = rng.normal(size=(4, 4))
    beta += beta.T
    total = alpha + beta
    for q in itertools.product(range(4), repeat=4):
        i, j, k, l = q
        rhf = 0.5 * total[i, j] * total[k, l] - 0.25 * total[i, k] * total[j, l]
        uhf = 0.5 * total[i, j] * total[k, l] - 0.5 * (
            alpha[i, k] * alpha[j, l] + beta[i, k] * beta[j, l]
        )
        assert hf_eri_weight(q, total.__getitem__) == pytest.approx(rhf)
        assert hf_eri_weight(
            q, total.__getitem__, spin_densities=(alpha.__getitem__, beta.__getitem__)
        ) == pytest.approx(uhf)
    # This adapter is optional: arbitrary four-index weights above never
    # construct or fit an artificial density matrix.


@pytest.mark.parametrize("indices", [(0, 1, 2), (0, 1, 2, -1), (0, 1, 2, True)])
def test_invalid_eri_indices_are_rejected(indices):
    with pytest.raises((ValueError, TypeError)):
        eri_weight_orbit(indices)


def test_nonfinite_external_weights_are_rejected_without_screening():
    with pytest.raises(ValueError, match="finite"):
        fold_dense_eri_weight(lambda _: float("nan"), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        fold_normalized_pair_weight(lambda i, j: float("inf"), (0, 0, 0, 0))

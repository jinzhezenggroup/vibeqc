# ruff: noqa: ANN001, ANN201, ANN202
"""DF-CCSD(T) raw three-index pullback tests for #158 slice A."""

import numpy as np
import pytest
from vibeqc_compiler.method.matrix_function import SymmetricMatrixFunctionSpec

from tools.vibeqc_cc.df_gradient import (
    DFThreeIndexCotangent,
    pullback_df_three_index,
)


def _sym(value):
    return 0.5 * (value + value.T)


def _inverse_root(metric, threshold, expected_rank=None):
    return (
        SymmetricMatrixFunctionSpec(
            metric.shape[0],
            "test-df-ccsdt-gradient",
            relative_threshold=threshold,
        )
        .prepare(metric, expected_rank=expected_rank)
        .value
    )


def _three_index(raw_a, metric, coefficients, p, q, threshold):
    transformed = np.einsum(
        "mp,nq,mnP->Ppq",
        coefficients[:, p],
        coefficients[:, q],
        raw_a,
        optimize=True,
    )
    return np.einsum(
        "Ppq,PQ->Qpq",
        transformed,
        _inverse_root(metric, threshold),
        optimize=True,
    )


def test_pullback_matches_same_hamiltonian_dense_eri_finite_difference():
    rng = np.random.default_rng(158)
    nao = nmo = 4
    naux = 3
    coefficients, _ = np.linalg.qr(rng.normal(size=(nao, nmo)))
    raw_a = rng.normal(size=(nao, nao, naux))
    raw_a = 0.5 * (raw_a + raw_a.transpose(1, 0, 2))
    rotation, _ = np.linalg.qr(rng.normal(size=(naux, naux)))
    metric = rotation @ np.diag([3.0, 0.8, 0.04]) @ rotation.T
    threshold = 0.1
    columns = tuple(range(nmo))
    b = _three_index(raw_a, metric, coefficients, columns, columns, threshold)
    weight = rng.normal(size=(nmo, nmo, nmo, nmo))
    bar_b = np.einsum("pqrs,Qrs->Qpq", weight, b, optimize=True)
    bar_b += np.einsum("rspq,Qrs->Qpq", weight, b, optimize=True)

    pullback = pullback_df_three_index(
        raw_a,
        metric,
        coefficients,
        [DFThreeIndexCotangent(columns, columns, bar_b)],
        relative_threshold=threshold,
        expected_rank=2,
    )

    da = rng.normal(size=raw_a.shape)
    da = 0.5 * (da + da.transpose(1, 0, 2))
    dm = _sym(rng.normal(size=metric.shape))
    da /= np.linalg.norm(da)
    dm /= np.linalg.norm(dm)

    def objective(a, m):
        factor = _three_index(a, m, coefficients, columns, columns, threshold)
        eri = np.einsum("Qpq,Qrs->pqrs", factor, factor, optimize=True)
        return float(np.einsum("pqrs,pqrs->", weight, eri, optimize=True))

    step = 2.0e-6
    finite_difference = (
        objective(raw_a + step * da, metric + step * dm)
        - objective(raw_a - step * da, metric - step * dm)
    ) / (2 * step)
    analytic = float(np.vdot(pullback.bar_a, da) + np.vdot(pullback.bar_m, dm))

    np.testing.assert_allclose(analytic, finite_difference, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(
        pullback.bar_a, pullback.bar_a.transpose(1, 0, 2), atol=0, rtol=0
    )
    np.testing.assert_allclose(pullback.bar_m, pullback.bar_m.T, atol=0, rtol=0)
    assert pullback.metric_rank == 2
    assert pullback.metric_relative_gap > 1e-3
    assert abs(float(np.vdot(pullback.bar_m, dm))) > 1e-5


def test_multiple_factorized_blocks_accumulate_the_same_raw_weights():
    rng = np.random.default_rng(159)
    nao = nmo = 5
    naux = 4
    coefficients, _ = np.linalg.qr(rng.normal(size=(nao, nmo)))
    raw_a = rng.normal(size=(nao, nao, naux))
    raw_a = 0.5 * (raw_a + raw_a.transpose(1, 0, 2))
    metric = _sym(rng.normal(size=(naux, naux)))
    metric = metric @ metric.T + np.eye(naux)
    threshold = 1.0e-8
    occupied, virtual = (0, 1), (2, 3, 4)
    blocks = [
        DFThreeIndexCotangent(
            occupied,
            virtual,
            rng.normal(size=(naux, len(occupied), len(virtual))),
        ),
        DFThreeIndexCotangent(
            virtual,
            virtual,
            rng.normal(size=(naux, len(virtual), len(virtual))),
        ),
    ]
    total = pullback_df_three_index(
        raw_a, metric, coefficients, blocks, relative_threshold=threshold
    )
    pieces = [
        pullback_df_three_index(
            raw_a, metric, coefficients, [block], relative_threshold=threshold
        )
        for block in blocks
    ]

    np.testing.assert_allclose(
        total.bar_a, pieces[0].bar_a + pieces[1].bar_a, atol=2e-13, rtol=0
    )
    np.testing.assert_allclose(
        total.bar_m, pieces[0].bar_m + pieces[1].bar_m, atol=2e-13, rtol=0
    )


def test_pullback_rejects_unresolved_rank_branch_and_insufficient_budget():
    raw_a = np.zeros((2, 2, 2), dtype=np.float64)
    coefficients = np.eye(2, dtype=np.float64)
    metric = np.diag([2.0, 0.2]).astype(np.float64)
    block = DFThreeIndexCotangent((0,), (1,), np.ones((2, 1, 1), dtype=np.float64))

    with pytest.raises(ValueError, match="unresolved at the cutoff"):
        pullback_df_three_index(
            raw_a,
            metric,
            coefficients,
            [block],
            relative_threshold=0.1,
        )

    with pytest.raises(MemoryError, match="logical numeric bytes"):
        pullback_df_three_index(
            raw_a,
            np.diag([2.0, 0.3]).astype(np.float64),
            coefficients,
            [block],
            relative_threshold=0.1,
            max_bytes=64,
        )

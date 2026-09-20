"""Algebraic and layout guards for the final-K/force handoff."""

import typing

import numpy as np
import pytest


@pytest.mark.parametrize("discarded", [0, 2])
def test_whitened_projection_recovers_only_retained_metric_directions(
    discarded: typing.Any,
) -> None:
    """A nonzero discarded component cannot be reconstructed by a pseudoinverse.

    This deliberately keeps discarded eigenvalues finite. Their spectral
    Frechet response is nonzero, so the native reuse path must require full
    rank instead of interpreting rank truncation as a zero raw direction.
    """
    rng = np.random.default_rng(407)
    n, a, r = 7, 5, 3
    q = np.linalg.qr(rng.normal(size=(a, a)))[0]
    eigenvalues = np.geomspace(0.1, 4, a)
    scales = 1 / np.sqrt(eigenvalues)
    scales[:discarded] = 0
    x = (q * scales) @ q.T
    root = (q * np.sqrt(eigenvalues)) @ q.T
    raw = rng.normal(size=(n, n, a))
    raw += raw.transpose(1, 0, 2)
    c = np.linalg.qr(rng.normal(size=(n, r)))[0]
    b = raw @ x
    u = np.einsum("mnq,ni->miq", b, c)
    # Match the native cuBLAS flattening: U[mu,i,Q] -> [j,i,Q], then
    # transpose (Q,ij) and multiply the symmetric metric square root.
    white = c.T @ u.reshape(n, r * a)
    recovered = white.reshape(r * r, a) @ root
    expected = np.einsum("mi,mnq,nj->ijq", c, raw, c).reshape(r * r, a)
    if not discarded:
        np.testing.assert_allclose(recovered, expected, atol=2e-13, rtol=0)
    else:
        assert np.max(np.abs(recovered - expected)) > 0.1
        retained_projector = q[:, discarded:] @ q[:, discarded:].T
        np.testing.assert_allclose(
            recovered, expected @ retained_projector, atol=2e-13, rtol=0
        )

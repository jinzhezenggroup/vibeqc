"""Independent finite-Fock references for the bounded triples resolvent frontend."""

from itertools import permutations

import numpy as np
import pytest
from generativeqc_compiler.cc.triples import _v, _views, _w, r3, triples_energy
from generativeqc_compiler.cc.triples_fock_response import (
    build_runtime_triples_resolvent_program,
    build_triples_fock_moment_program,
)
from generativeqc_compiler.cc.triples_response import full_triples_vjp
from generativeqc_compiler.tensor import execute

NAMES = ("ovvv", "ovoo", "ovov", "fov", "t1", "t2")
SPACES = ("ovvv", "ovoo", "ovov", "ov", "ov", "oovv")


def _case(
    o: int, v: int, *, degenerate: bool = True
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1910 + 10 * o + v)
    arrays = [
        rng.normal(scale=0.1, size=tuple(o if s == "o" else v for s in spaces))
        for spaces in SPACES
    ]
    arrays[-1] = (arrays[-1] + arrays[-1].transpose(1, 0, 3, 2)) / 2
    eo, ev = np.linspace(-2, -0.5, o), np.linspace(0.4, 1.5, v)
    if degenerate:
        eo[1], ev[1] = eo[0], ev[0]
    return arrays, eo, ev


def _rotate(
    arrays: list[np.ndarray], uo: np.ndarray, uv: np.ndarray
) -> list[np.ndarray]:
    """Transform all tensor indices into columns of the new orbital basis."""
    result = []
    for original, spaces in zip(arrays, SPACES, strict=True):
        tensor = original
        for axis, space in enumerate(spaces):
            u = uo if space == "o" else uv
            tensor = np.moveaxis(np.tensordot(tensor, u, axes=(axis, 0)), -1, axis)
        result.append(tensor)
    return result


def _canonical_energy(
    arrays: list[np.ndarray], fo: np.ndarray, fv: np.ndarray
) -> float:
    """Recanonicalize with NumPy, then evaluate the old triangular primal."""
    eo, uo = np.linalg.eigh(fo)
    ev, uv = np.linalg.eigh(fv)
    return triples_energy(len(eo), len(ev), *_rotate(arrays, uo, uv), eo, ev)


def _dense_reference(
    arrays: list[np.ndarray], eo: np.ndarray, ev: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All ordered triples, no pair-domain shortcut and no marginal equivalence.

    The complete inverse derivative has six separate index contractions. This
    oracle deliberately retains each one rather than assuming they are equal.
    """
    o, v = len(eo), len(ev)
    views = _views(*arrays)
    left, right = np.zeros((v, v, v, o, o, o)), np.empty((v, v, v, o, o, o))
    for abc in np.ndindex(v, v, v):
        right[abc] = r3(_w(views, *abc) + 0.5 * _v(views, *abc))
        for perm in permutations(range(3)):
            left[abc] += _w(views, *(abc[i] for i in perm)).transpose(
                tuple(np.argsort(perm))
            )
    denominator = np.zeros_like(left)
    for axis in range(6):
        e = -ev if axis < 3 else eo
        shape = [1] * 6
        shape[axis] = len(e)
        denominator += e.reshape(shape)
    x, y = left / denominator, right / denominator
    bo, bv = np.zeros((o, o)), np.zeros((v, v))
    for axis in range(6):
        xx = np.moveaxis(x, axis, 0).reshape(x.shape[axis], -1)
        yy = np.moveaxis(y, axis, 0).reshape(y.shape[axis], -1)
        target = bv if axis < 3 else bo
        target += (2 if axis < 3 else -2) * (xx @ yy.T)
    return (bo + bo.T) / 2, (bv + bv.T) / 2, left, right


def _paged(
    arrays: list[np.ndarray], eo: np.ndarray, ev: np.ndarray, capacity: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two live pages; tails and off-diagonal page blocks are exercised explicitly."""
    o, v = len(eo), len(ev)
    vectors = build_runtime_triples_resolvent_program(o, v, capacity=capacity)
    oo = build_triples_fock_moment_program(o, capacity=capacity, block="oo")
    vv = build_triples_fock_moment_program(o, capacity=capacity, block="vv")
    base = dict(zip(NAMES, arrays, strict=True), eps_o=eo, eps_v=ev)
    bo, bv = np.zeros((o, o)), np.zeros((v, v))
    for b in range(v):
        for c in range(b + 1):
            multiplicity = 1 if b == c else 2

            def page(
                start: int, b: int = b, c: int = c
            ) -> tuple[dict[str, np.ndarray], int]:
                count = min(capacity, v - start)
                amap = np.zeros(capacity, dtype=np.int64)
                amap[:count] = np.arange(start, start + count)
                active = np.zeros(capacity)
                active[:count] = 1
                value = execute(
                    vectors,
                    {
                        **base,
                        "a_map": amap,
                        "b_map": np.full(capacity, b, dtype=np.int64),
                        "c_map": np.full(capacity, c, dtype=np.int64),
                        "active": active,
                    },
                ).outputs
                for key in ("x", "y"):
                    np.testing.assert_array_equal(value[key][count:], 0)
                return value, count

            for start in range(0, v, capacity):
                left, count = page(start)
                feeds = {"x_left": left["x"], "y_left": left["y"]}
                bo += multiplicity * execute(oo, feeds).outputs["foo"]
                for other in range(start, v, capacity):
                    right, right_count = (
                        (left, count) if other == start else page(other)
                    )
                    value = execute(
                        vv, dict(feeds, x_right=right["x"], y_right=right["y"])
                    ).outputs["fvv"][:count, :right_count]
                    bv[start : start + count, other : other + right_count] += (
                        multiplicity * value
                    )
                    if other != start:
                        bv[other : other + right_count, start : start + count] += (
                            multiplicity * value.T
                        )
    return bo, bv


@pytest.mark.parametrize("o,v", ((2, 3), (3, 2), (3, 3)))
@pytest.mark.parametrize("capacity", (1, 2, 4))
def test_two_page_fock_moments_match_independent_dense_reference(
    o: int, v: int, capacity: int
) -> None:
    arrays, eo, ev = _case(o, v)
    bo, bv, _, _ = _dense_reference(arrays, eo, ev)
    actual = _paged(arrays, eo, ev, capacity)
    for observed, expected in zip(actual, (bo, bv), strict=True):
        np.testing.assert_allclose(observed, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize("splitting", (None, 0.0, 1e-12, 1e-8))
def test_full_fock_derivative_matches_recanonicalized_energy_and_dense_inverse(
    splitting: float | None,
) -> None:
    arrays, eo, ev = _case(2, 3, degenerate=splitting is not None)
    if splitting is not None:
        eo[1] += splitting
        ev[1] += splitting
    bo, bv = _paged(arrays, eo, ev, 2)
    _, _, left, right = _dense_reference(arrays, eo, ev)
    rng = np.random.default_rng(1905)
    ho, hv = rng.normal(size=(2, 2)), rng.normal(size=(3, 3))
    ho, hv = (ho + ho.T) / 2, (hv + hv.T) / 2
    expected = float(np.sum(bo * ho) + np.sum(bv * hv))
    for step in (1e-3, 3e-4, 1e-4):
        plus = _canonical_energy(
            arrays, np.diag(eo) + step * ho, np.diag(ev) + step * hv
        )
        minus = _canonical_energy(
            arrays, np.diag(eo) - step * ho, np.diag(ev) - step * hv
        )
        assert abs((plus - minus) / (2 * step) - expected) < 3e-8
    # Dense full Kronecker inversion is independent of the diagonal/moment path.
    fo, fv = np.diag(eo) + 1e-3 * ho, np.diag(ev) + 1e-3 * hv
    matrix = np.zeros((left.size, left.size))
    for axis, fock in enumerate([-fv] * 3 + [fo] * 3):
        term = np.array([[1.0]])
        for k, size in enumerate(left.shape):
            term = np.kron(term, fock if k == axis else np.eye(size))
        matrix += term
    dense = 2 * left.ravel() @ np.linalg.solve(matrix, right.ravel())
    assert abs(dense - _canonical_energy(arrays, fo, fv)) < 2e-12
    bars = full_triples_vjp(2, 3, *arrays, eo, ev, inputs=("eps_o", "eps_v"))
    np.testing.assert_allclose(np.diag(bo), bars["eps_o"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(np.diag(bv), bars["eps_v"], atol=2e-12, rtol=0)


def test_degenerate_blocks_are_covariant_including_nonzero_offdiagonals() -> None:
    arrays, eo, ev = _case(2, 3)
    bo, bv = _paged(arrays, eo, ev, 2)
    angle = 0.5 * np.arctan2(bo[0, 0] - bo[1, 1], 2 * bo[0, 1])
    uo = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    uv = np.eye(3)
    uv[:2, :2] = [[np.cos(0.3), np.sin(0.3)], [-np.sin(0.3), np.cos(0.3)]]
    rotated = _rotate(arrays, uo, uv)
    bro, brv = _paged(rotated, eo, ev, 2)
    np.testing.assert_allclose(bro, uo.T @ bo @ uo, atol=2e-12, rtol=0)
    np.testing.assert_allclose(brv, uv.T @ bv @ uv, atol=2e-12, rtol=0)
    # Equal diagonal eps cotangents alone cannot qualify this subspace.
    assert abs(bro[0, 0] - bro[1, 1]) < 2e-12
    assert abs(bro[0, 1]) > 1e-7
    assert (
        abs(
            triples_energy(2, 3, *rotated, eo, ev)
            - triples_energy(2, 3, *arrays, eo, ev)
        )
        < 2e-12
    )

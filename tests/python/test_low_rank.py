"""Independent dense PSD oracles for incremental, bounded pair factorization."""

from hashlib import sha256

import numpy as np
import pytest
from vibeqc_compiler.common.resources import ResourceBudget

from tools.vibeqc_posthf.low_rank import IncrementalCholesky
from tools.vibeqc_posthf.pair_space import PairSpace


class DenseColumns:
    """Tiny test oracle, deliberately independent of the native ERI source."""

    def __init__(self, matrix, nbf):
        self.matrix = np.array(matrix, dtype=np.float64)
        self.space = PairSpace(nbf)
        assert self.matrix.shape == (self.space.size,) * 2
        assert np.array_equal(self.matrix, self.matrix.T)
        self.identity = sha256(self.matrix.tobytes()).hexdigest()
        self.numeric_bytes = self.matrix.nbytes
        self.calls = []
        self.closed = False
        self.fail_pivot = None

    def check(self):
        if self.closed:
            raise RuntimeError("closed test source")

    def diagonal(self, begin, count):
        self.calls.append(("diagonal", begin, count))
        return self.matrix.diagonal()[begin : begin + count].copy()

    def column(self, pivot, begin, count):
        self.calls.append((pivot, begin, count))
        if self.fail_pivot == pivot:
            raise RuntimeError("injected source failure")
        return self.matrix[begin : begin + count, pivot].copy()


def factors(plan):
    return np.concatenate([plan.factor_tile(i, 1) for i in range(plan.rank)], axis=0)


@pytest.mark.parametrize("nbf", [1, 2, 7])
def test_pair_inner_products_and_exact_triangular_inverse(nbf):
    space = PairSpace(nbf)
    rng = np.random.default_rng(191)
    a, b = rng.normal(size=(2, nbf, nbf))
    a, b = a + a.T, b + b.T
    packed = space.pack(a)
    np.testing.assert_allclose(space.unpack(packed), a, atol=1e-15)
    assert packed @ space.pack(b) == pytest.approx(np.sum(a * b), abs=1e-13)
    for i in range(space.size):
        mu, nu = space.pair(i)
        assert space.index(mu, nu) == space.index(nu, mu) == i
    large = PairSpace(100_000)
    assert large.pair(large.size - 1) == (99_999, 99_999)
    with pytest.raises(ValueError):
        space.pair(space.size)
    if nbf > 1:
        a[0, 1] += 1e-15
        with pytest.raises(ValueError, match="symmetric"):
            space.pack(a)


@pytest.mark.parametrize("pair_tile", [1, 4, 100])
def test_incremental_prefix_matches_scratch_and_recovers_target(pair_tile):
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(10, 10))
    matrix = raw @ raw.T + np.eye(10) * 0.01
    source = DenseColumns(matrix, 4)
    plan = IncrementalCholesky(source, rank_capacity=10, pair_tile=pair_tile)
    first = plan.refine(0, maximum_rank=3)
    assert first.status == "rank_limited" and first.invalidates_solver_history
    prefix = factors(plan).copy()
    calls = len(source.calls)
    result = plan.refine(1e-13)
    assert result.status == "threshold_met" and plan.rank == 10
    np.testing.assert_array_equal(factors(plan)[:3], prefix)
    assert all(call[0] != "diagonal" for call in source.calls[calls:])
    np.testing.assert_allclose(factors(plan).T @ factors(plan), matrix, atol=3e-14)
    scratch = IncrementalCholesky(
        DenseColumns(matrix, 4), rank_capacity=10, pair_tile=3
    )
    scratch.refine(1e-13)
    assert plan.identity == scratch.identity
    np.testing.assert_array_equal(factors(plan), factors(scratch))
    assert [p.pair_index for p in plan.history] == [
        p.pair_index for p in scratch.history
    ]
    with pytest.raises(ValueError, match="generation"):
        plan.factor_tile(0, 1, identity=first.identity)
    assert not plan.refine(1e-13).invalidates_solver_history
    assert all(call[-1] <= min(pair_tile, 10) for call in source.calls)


def test_same_approximation_residual_bounds_and_normalization():
    rng = np.random.default_rng(91)
    raw = rng.normal(size=(6, 6))
    matrix = raw @ raw.T
    plan = IncrementalCholesky(DenseColumns(matrix, 3), rank_capacity=6)
    plan.refine(0, maximum_rank=2)
    low = factors(plan)
    residual = matrix - low.T @ low
    diagnostics = plan.diagnostics()
    np.testing.assert_allclose(residual.diagonal(), plan._residual, atol=2e-15)
    assert np.linalg.eigvalsh(residual)[0] > -5e-15
    assert np.max(np.abs(residual)) <= diagnostics["conditional_entry_bound"] + 5e-15
    assert (
        np.linalg.norm(residual)
        <= diagnostics["conditional_spectral_and_frobenius_bound"] + 5e-15
    )
    density = rng.normal(size=(3, 3))
    density += density.T
    packed_density = plan.space.pack(density)
    physical = np.array([plan.space.unpack(row) for row in low])
    j = np.einsum("Pmn,Prs,rs->mn", physical, physical, density)
    np.testing.assert_allclose(
        plan.space.pack(j), low.T @ (low @ packed_density), atol=1e-13
    )
    assert diagnostics["observable_certification"] == "unverified"


def test_zero_rank_ties_roundoff_and_ill_conditioning():
    zeros = IncrementalCholesky(DenseColumns(np.zeros((3, 3)), 2), rank_capacity=0)
    assert zeros.refine(0).status == "threshold_met"
    assert zeros.factor_tile(0, 0).shape == (0, 3)
    with pytest.raises(ValueError):
        zeros.factor_tile(0, 1)
    tied = IncrementalCholesky(DenseColumns(np.eye(3), 2), rank_capacity=3)
    tied.refine(0)
    assert [row.pair_index for row in tied.history] == [0, 1, 2]
    ill = IncrementalCholesky(
        DenseColumns(np.diag([1, 1e-10, 1e-18]), 2), rank_capacity=3
    )
    assert ill.refine(0).status == "roundoff_limited"
    assert ill.rank == 2
    assert ill.refine(1e-17).status == "threshold_met"
    tiny = IncrementalCholesky(DenseColumns(np.eye(3) * 1e-100, 2), rank_capacity=3)
    assert tiny.refine(0).status == "threshold_met"
    assert tiny.rank == 3


def test_psd_failures_and_source_failure_leave_valid_prefix():
    with pytest.raises(ValueError, match="negative diagonal"):
        IncrementalCholesky(DenseColumns(np.diag([1, -1, 1]), 2), rank_capacity=3)
    indefinite = DenseColumns([[1, 2, 0], [2, 1, 0], [0, 0, 1]], 2)
    plan = IncrementalCholesky(indefinite, rank_capacity=3)
    identity = plan.identity
    with pytest.raises(ValueError, match="PSD Cauchy"):
        plan.refine(0)
    assert plan.rank == 0 and plan.identity == identity
    source = DenseColumns(np.diag([3, 2, 1]), 2)
    source.fail_pivot = 1
    plan = IncrementalCholesky(source, rank_capacity=3)
    with pytest.raises(RuntimeError, match="injected"):
        plan.refine(0)
    assert plan.rank == 1
    prefix = factors(plan).copy()
    source.fail_pivot = None
    plan.refine(0)
    np.testing.assert_array_equal(factors(plan)[:1], prefix)
    source.identity = "changed-geometry"
    with pytest.raises(ValueError, match="identity changed"):
        plan.refine(0)


def test_budget_preflight_before_source_reads_and_immutable_exports():
    source = DenseColumns(np.eye(6), 3)
    with pytest.raises(MemoryError):
        IncrementalCholesky(
            source, rank_capacity=6, budget=ResourceBudget(host_bytes=10)
        )
    assert source.calls == []
    plan = IncrementalCholesky(source, rank_capacity=6, pair_tile=2)
    exact = plan.resource_plan.peak_bytes["pageable"]
    limited = IncrementalCholesky(
        source, rank_capacity=6, pair_tile=2, budget=ResourceBudget(host_bytes=exact)
    )
    limited.refine(0)
    out = limited.factor_tile(0, 1)
    with pytest.raises(ValueError):
        out.setflags(write=True)
    with pytest.raises(ValueError, match="reserved tile"):
        limited.factor_tile(0, 2)
    with pytest.raises(AttributeError):
        limited.rank_capacity = 10
    limited.close()
    with pytest.raises(RuntimeError, match="closed"):
        limited.refine(0)

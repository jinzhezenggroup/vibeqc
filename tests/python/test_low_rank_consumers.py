"""Same-approximation contractions before independent exact-target comparisons."""

from dataclasses import replace

import numpy as np
import pytest
from test_low_rank import DenseColumns
from test_low_rank_source import source_for
from vibeqc.accuracy import AccuracyAssessment, ObservableTarget, TargetAccuracy
from vibeqc_compiler.common.resources import ResourceBudget

from tools.vibeqc_posthf.conventions import MOBlock
from tools.vibeqc_posthf.coulomb_columns import CoulombColumns
from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture
from tools.vibeqc_posthf.low_rank import IncrementalCholesky
from tools.vibeqc_posthf.low_rank_accuracy import audit_fixed_density
from tools.vibeqc_posthf.low_rank_consumers import LowRankProvider
from tools.vibeqc_posthf.mp2 import restricted_mp2
from tools.vibeqc_posthf.sources import NativeSource


def dense_approximation(factor):
    """Independent tiny tensor reconstruction from physical symmetric factors."""
    matrices = [
        factor.space.unpack(factor.factor_tile(i, 1)[0]) for i in range(factor.rank)
    ]
    if not matrices:
        return np.zeros((factor.space.nbf,) * 4)
    return np.einsum("Pmn,Prs->mnrs", matrices, matrices)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("rank", [0, 1, 4, 6])
def test_raw_jk_matches_same_approximation_with_exact_spin_factors(spins, rank):
    rng = np.random.default_rng(190)
    seed = rng.normal(size=(6, 6))
    factor = IncrementalCholesky(DenseColumns(seed @ seed.T, 3), rank_capacity=6)
    factor.refine(0, maximum_rank=rank)
    provider = LowRankProvider(factor)
    density = rng.normal(size=(spins, 3, 3))
    density += density.swapaxes(1, 2)
    actual = provider.jk(density[0] if spins == 1 else density)
    tensor = dense_approximation(factor)
    j = np.einsum("mnrs,rs->mn", tensor, density.sum(axis=0))
    k = np.einsum("mrns,xrs->xmn", tensor, density)
    np.testing.assert_allclose(actual.coulomb, j, atol=3e-13)
    np.testing.assert_allclose(actual.exchange, k[0] if spins == 1 else k, atol=3e-13)
    assert actual.hamiltonian_id == factor.hamiltonian_id
    assert np.array_equal(actual.exchange, actual.exchange.swapaxes(-1, -2))
    with pytest.raises(ValueError):
        actual.coulomb.setflags(write=True)
    density[0, 0, 1] += 0.1
    with pytest.raises(ValueError, match="symmetric"):
        provider.jk(density[0] if spins == 1 else density)


def test_shared_consumer_budget_preflight_and_generation_invalidation():
    factor = IncrementalCholesky(DenseColumns(np.eye(3), 2), rank_capacity=3)
    factor.refine(0, maximum_rank=1)
    provider = LowRankProvider(factor)
    plan = provider.jk(np.eye(2)).diagnostics["resource_plan"]
    required = plan["peak_bytes"]["host"]
    limited = LowRankProvider(factor, budget=ResourceBudget(host_bytes=required - 1))
    with pytest.raises(MemoryError):
        limited.jk(np.eye(2))
    exact = LowRankProvider(factor, budget=ResourceBudget(host_bytes=required))
    exact.jk(np.eye(2))
    factor.refine(0)
    with pytest.raises(ValueError, match="generation changed"):
        provider.jk(np.eye(2))
    LowRankProvider(factor).jk(np.eye(2))
    with pytest.raises(AttributeError):
        provider.snapshot = None


def test_mo_blocks_retain_exact_reference_and_approximate_correlation_identity():
    source, arrays = source_for("water")
    metadata, _ = load_fixture("water")
    snapshot = fixture_snapshot(metadata, arrays)
    with source:
        factor = IncrementalCholesky(
            CoulombColumns(source), rank_capacity=28, pair_tile=5
        )
        factor.refine(0, maximum_rank=4)
        provider = LowRankProvider(factor, snapshot)
        block = MOBlock(((0, 2), (1, 4), (3,), (0, 5, 6)))
        result = provider.get(block)
        approximate = dense_approximation(factor)
        panels = [snapshot.coefficients[:, slot] for slot in block.slots]
        expected = np.einsum(
            "mnrs,mp,nq,ri,sj->pqij", approximate, *panels, optimize=True
        )
        np.testing.assert_allclose(result.to_host(), expected, atol=1e-12)
        assert result.reference_id == snapshot.identity
        assert result.hamiltonian_id == factor.hamiltonian_id
        assert result.hamiltonian_id != snapshot.hamiltonian_id
        with pytest.raises(ValueError, match="Hamiltonian mismatch"):
            restricted_mp2(snapshot, provider)
        with pytest.raises(ValueError, match="reference and target"):
            LowRankProvider(factor, replace(snapshot, geometry_hash="another-geometry"))
        empty = provider.get(MOBlock(((), (0,), (1,), (2,))))
        assert empty.to_host().shape == (0, 1, 1, 1)
        factor.refine(1e-11)
        with pytest.raises(ValueError, match="generation changed"):
            provider.get(block)
        tight = LowRankProvider(factor, snapshot)
        np.testing.assert_allclose(
            tight.get(block).to_host(),
            arrays["conventional_mo"][np.ix_(*block.slots)],
            atol=3e-11,
            rtol=1e-10,
        )
        density = (
            2
            * snapshot.coefficients[:, : snapshot.nocc]
            @ snapshot.coefficients[:, : snapshot.nocc].T
        )
        native = tight.jk(density)
        np.testing.assert_allclose(
            native.coulomb,
            np.einsum("mnrs,rs->mn", arrays["ao"], density),
            atol=3e-11,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            native.exchange,
            np.einsum("mrns,rs->mn", arrays["ao"], density),
            atol=3e-11,
            rtol=1e-10,
        )


@pytest.mark.parametrize("charge,multiplicity", [(2, 1), (0, 3)])
def test_mo_reference_rejects_changed_electronic_ensemble_at_identical_ao_topology(
    charge, multiplicity
):
    original, arrays = source_for("water")
    metadata, _ = load_fixture("water")
    snapshot = fixture_snapshot(metadata, arrays)
    with (
        original,
        NativeSource(
            original.atoms,
            basis=original.shells,
            auxiliary_basis=original.auxiliary_shells,
            representation=original.representation,
            charge=charge,
            multiplicity=multiplicity,
        ) as changed,
    ):
        # Each case keeps every preexisting compatibility check satisfied:
        # either charge/electron count or only spin multiplicity differs.
        assert changed.geometry_hash == snapshot.geometry_hash
        assert changed.basis_hash == snapshot.basis_hash
        assert changed.nbf == snapshot.nmo
        assert changed.representation == snapshot.representation
        columns = CoulombColumns(changed)
        with IncrementalCholesky(columns, rank_capacity=0) as factor:
            before = dict(columns.statistics)
            with pytest.raises(ValueError, match="reference and target"):
                LowRankProvider(factor, snapshot)
            assert columns.statistics == before
            assert factor.rank == 0


def test_fixed_density_accuracy_evidence_cannot_certify_the_exact_relaxed_target():
    source, arrays = source_for("h2")
    metadata, _ = load_fixture("h2")
    snapshot = fixture_snapshot(metadata, arrays)
    density = 2 * snapshot.coefficients[:, :1] @ snapshot.coefficients[:, :1].T
    with source:
        factor = IncrementalCholesky(CoulombColumns(source), rank_capacity=3)
        factor.refine(0, maximum_rank=1)
        audit = audit_fixed_density(factor, density)
        tensor = dense_approximation(factor)
        difference = tensor - arrays["ao"]
        j = np.einsum("mnrs,rs->mn", difference, density)
        k = np.einsum("mrns,rs->mn", difference, density)
        assert audit.evidence.value == pytest.approx(
            abs(0.5 * np.sum(density * (j - 0.5 * k))), abs=1e-13
        )
        assert audit.evidence.scope == "fixed_density"
        assert audit.evidence.source == "integral_factorization"
        assert audit.evidence.evaluated_model_id != audit.target_model.identity
        target = TargetAccuracy(
            (ObservableTarget("energy", "absolute", "Eh", absolute=1),)
        )
        with pytest.raises(ValueError, match="model mismatch"):
            AccuracyAssessment(audit.target_model, target, (audit.evidence,), True)
        factor.refine(1e-12)
        assert audit_fixed_density(factor, density).evidence.value < 1e-11

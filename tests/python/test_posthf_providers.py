"""Native tile, bounded transformation, cache and same-Hamiltonian DF gates."""

from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_posthf.conventions import MOBlock
from tools.vibeqc_posthf.df import DFProvider, MetricFactor
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.mp2 import restricted_mp2
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource


@pytest.fixture
def source_factory():
    sources = []

    def create(name="h2", **changes):
        meta, a = load_fixture(name)
        args = source_arguments(meta)
        args.update(changes)
        try:
            s = NativeSource(**args)
        except (OSError, FileNotFoundError, AttributeError) as error:
            pytest.skip(f"native post-HF library unavailable: {error}")
        sources.append(s)
        return s, meta, a

    yield create
    for s in sources:
        s.close()


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
@pytest.mark.parametrize("axis_tile", [1, 2, 4])
def test_streamed_mo_all_elements_cache_and_mp2(source_factory, name, axis_tile):
    source, meta, a = source_factory(name)
    s = fixture_snapshot(meta, a)
    with ConventionalProvider(s, source, axis_tile=axis_tile) as provider:
        for spaces in ("oooo", "ovov", "oovv", "ovvv", "vvvv"):
            block = MOBlock.from_spaces(s, spaces)
            result = provider.get(block)
            np.testing.assert_allclose(
                result.to_host(),
                a["conventional_mo"][np.ix_(*block.slots)],
                atol=1e-11,
                rtol=1e-10,
            )
            assert (
                result.diagnostics["peak_bytes_including_cached_blocks"]
                <= provider.budget_bytes
            )
            count = provider.statistics["source_tiles"]
            cached = provider.get(block)
            assert (
                cached.values is result.values
                and provider.statistics["source_tiles"] == count
            )
        mp = restricted_mp2(s, provider)
        assert (
            abs(
                mp.correlation_energy
                - meta["records"]["conventional"]["correlation_energy"]
            )
            < 1e-9
        )
        np.testing.assert_allclose(
            mp.amplitudes, a["conventional_t2"], atol=1e-11, rtol=1e-10
        )


def test_budget_preflight_lru_empty_and_source_mismatch(source_factory, monkeypatch):
    source, meta, a = source_factory("water")
    s = fixture_snapshot(meta, a)
    block = MOBlock.from_spaces(s, "ovov")
    p = ConventionalProvider(s, source)
    required = p.plan(block).peak_bytes
    too_small = ConventionalProvider(s, source, budget_bytes=required - 1)
    with monkeypatch.context() as patch:
        patch.setattr(
            source, "tile", lambda _: pytest.fail("read before budget preflight")
        )
        with pytest.raises(MemoryError):
            too_small.get(block)
    exact = ConventionalProvider(s, source, budget_bytes=required)
    exact.get(block)
    empty = MOBlock(((), (0,), (1,), (2,)))
    assert p.get(empty).to_host().shape == (0, 1, 1, 1)
    with pytest.raises(ValueError, match="does not match"):
        ConventionalProvider(replace(s, geometry_hash="new"), source).get(block)
    with pytest.raises(ValueError, match="Hamiltonian"):
        ConventionalProvider(replace(s, hamiltonian_id="df"), source)
    lru = ConventionalProvider(s, source, budget_bytes=required, cache_policy="lru")
    lru.get(block)
    lru.get(MOBlock.from_spaces(s, "vovo"))
    assert lru.statistics["evictions"] == 1
    p.close()
    with pytest.raises(RuntimeError):
        p.get(block)


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
def test_df_raw_b_blocks_and_same_hamiltonian_mp2(source_factory, name):
    source, meta, a = source_factory(name)
    metric = MetricFactor.from_source(source)
    n = source.nbf
    P = source.naux
    raw = source._read("three_center_eri", (0, 0, 0), (n, n, P))
    np.testing.assert_allclose(raw, a["raw_three_center"], atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(
        source._read("coulomb_metric", (0, 0), (P, P)),
        a["metric"],
        atol=1e-11,
        rtol=1e-10,
    )
    s = fixture_snapshot(meta, a, label="df", metric=metric)
    with DFProvider(s, source, metric, axis_tile=2, auxiliary_tile=3) as provider:
        block = MOBlock.from_spaces(s, "ovov")
        result = provider.get(block)
        np.testing.assert_allclose(
            result.to_host(), a["df_mo"][np.ix_(*block.slots)], atol=1e-11, rtol=1e-10
        )
        mp = restricted_mp2(s, provider)
        assert (
            abs(mp.correlation_energy - meta["records"]["df"]["correlation_energy"])
            < 1e-9
        )
        np.testing.assert_allclose(mp.amplitudes, a["df_t2"], atol=1e-11, rtol=1e-10)
        left = provider.three_index(
            block.slots[0], block.slots[1], auxiliary_begin=P - 1, auxiliary_count=1
        )
        full = provider.three_index(block.slots[0], block.slots[1])
        np.testing.assert_allclose(left, full[-1:], atol=1e-12)
        assert provider.statistics["peak_bytes"] <= provider.budget_bytes
        assert meta["conventional_df_ao_max_difference"] > 1e-5
    with pytest.raises(ValueError, match="mismatch"):
        DFProvider(fixture_snapshot(meta, a), source, metric)
    with pytest.raises(MemoryError):
        MetricFactor.from_source(source, budget_bytes=1)
    with pytest.raises(MemoryError):
        DFProvider(s, source, metric, budget_bytes=1).get(block)


def test_df_rank_threshold_and_metric_invalidation(source_factory):
    _, meta, _ = source_factory()
    args = source_arguments(meta)
    source, _, _ = source_factory(
        auxiliary_basis=(*args["auxiliary_basis"], args["auxiliary_basis"][0])
    )
    metric = MetricFactor.from_source(source)
    assert metric.rank == 2 and source.naux == 3
    changed = MetricFactor.from_source(source, relative_threshold=1e-9)
    assert metric.identity != changed.identity


def test_native_hf_export_failure_isolation_and_ownership(source_factory):
    source, _, _ = source_factory()
    other, _, _ = source_factory()
    with pytest.raises(RuntimeError, match="did not converge"):
        export_rhf(source, max_iterations=1)
    s, stats = export_rhf(other, generation_id="first")
    assert stats["physical_residual"] < 1e-10
    old = s.coefficients.copy()
    other.close()
    np.testing.assert_array_equal(s.coefficients, old)
    fresh, _ = export_rhf(source, generation_id="second")
    assert fresh.identity != s.identity
    np.testing.assert_allclose(fresh.coefficients, s.coefficients, atol=1e-12)


def test_actual_f_partial_shell_tiles(source_factory):
    source, meta, a = source_factory("f_heh")
    assert 3 in meta["actual_angular_momenta"]
    # Exercise a final partial spherical-f tile with independent asymmetric
    # atoms and complete public component ordering, without an expensive ffff
    # molecular provider pass in ordinary CI.
    found = False
    for request in source.requests("four_center_eri", axis_tile=3):
        if request.shell_indices == (0, 1, 2, 0):
            begin = source.global_offsets(request)
            tile = source.tile(request)
            slices = tuple(slice(b, b + n) for b, n in zip(begin, tile.shape))
            np.testing.assert_allclose(tile, a["ao"][slices], atol=1e-11, rtol=1e-10)
            found |= request.tile.shape[1] == 1
    assert found


def test_raw_empty_overflow_wrong_rank_and_closed(source_factory):
    source, _, _ = source_factory()
    assert (
        source._read("four_center_eri", (source.nbf, 0, 0, 0), (0, 1, 1, 1)).size == 0
    )
    with pytest.raises(OverflowError):
        source._read("four_center_eri", (1 << 64, 0, 0, 0), (1, 1, 1, 1))
    with pytest.raises(ValueError, match="rank"):
        source._read("four_center_eri", (0, 0), (1, 1))
    source.close()
    with pytest.raises(RuntimeError, match="closed"):
        source.one_electron()


def test_stale_metric_geometry_is_rejected(source_factory):
    source, meta, a = source_factory()
    metric = MetricFactor.from_source(source)
    snapshot = fixture_snapshot(meta, a, label="df", metric=metric)
    with pytest.raises(ValueError, match="mismatch"):
        DFProvider(snapshot, source, replace(metric, geometry_hash="stale"))


def test_valid_but_unsupported_raw_operator_has_explicit_status(source_factory):
    from tools.vibeqc_codegen.blocks import (
        BlockRequest,
        BlockStatus,
        RawBlock,
        ShellTile,
        TensorLayout,
    )
    from tools.vibeqc_codegen.ir import IntegralIR, OperatorSpec
    from tools.vibeqc_codegen.shell_signature import (
        BasisShell,
        CenterBinding,
        ShellSignature,
    )

    source, _, _ = source_factory()
    signature = ShellSignature(
        (BasisShell(0, 0, 0), BasisShell(1, 1, 0)),
        (CenterBinding(0, 0), CenterBinding(1, 1)),
    )
    request = BlockRequest(
        "kinetic-test",
        IntegralIR(
            signature,
            OperatorSpec("kinetic", (0, 1)),
            None,
            (RawBlock(TensorLayout(signature.tensor_indices, (1, 1)), 16),),
        ),
        ShellTile((0, 0), (1, 1)),
        shell_indices=(0, 1),
    )
    response = source.execute(request)
    assert (
        response.status == BlockStatus.UNSUPPORTED
        and response.reason
        and not response.values
    )


def test_source_metadata_cannot_silently_change_owned_geometry(source_factory):
    source, _, _ = source_factory()
    for name in ("atoms", "basis_hash", "geometry_hash", "identity", "numeric_bytes"):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(source, name, None)
    # A rejected metadata edit leaves the native source and neighboring state
    # usable, rather than labelling old native integrals with new geometry.
    assert np.isfinite(source.one_electron()[0]).all()

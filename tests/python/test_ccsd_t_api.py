"""End-to-end RCCSD(T) composition and batch-state tests for #150 C."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info

from tools.cc_endpoint_fixtures import load, snapshot_from_fixture
from tools.vibeqc_cc import (
    PreparedRCCSDTBatch,
    SolverOptions,
    rccsd_t_batch_energy,
    rccsd_t_energy,
    rccsd_t_method_capabilities,
)
from tools.vibeqc_cc import ccsd_t_api as api
from tools.vibeqc_cc.triples_cuda import CudaTriplesResult
from tools.vibeqc_posthf.providers import BlockResult, ConventionalProvider

TRIPLES_REFERENCE = {
    "h2": 8.392021714075268e-49,
    "he": 0.0,
    "h2o": -6.731393342463869e-05,
    "nh3": -1.122922812723691e-04,
    "ch4": -1.555665872715297e-04,
}


class FixtureProvider(ConventionalProvider):
    """Exact saved MO integrals without a native library dependency."""

    def __init__(self, snapshot, g):
        self.snapshot = snapshot
        self.g = g
        self.backend = "cpu"
        self.source = SimpleNamespace(_check_open=lambda: None)

    def get(self, block):
        return BlockResult(
            block,
            self.g[np.ix_(*block.slots)],
            self.snapshot.identity,
            self.snapshot.hamiltonian_id,
            {},
        )


def fixture_problem(name="h2"):
    meta, arrays = load(name)
    source = SimpleNamespace(
        electron_count=int(arrays["occ"].sum()),
        geometry_hash="fixture-" + name,
        basis_hash="fixture-basis-" + name,
    )
    snapshot = snapshot_from_fixture(source, meta, arrays)
    return snapshot, FixtureProvider(snapshot, arrays["g"]), meta, arrays


def test_capabilities_are_energy_only_and_homogeneous_batch():
    caps = rccsd_t_method_capabilities("rccsd(t)")
    assert caps.method == "rccsd(t)"
    assert caps.family == "coupled_cluster"
    assert caps.available is True
    assert caps.supports_batch is True
    assert caps.supported_properties == frozenset({"energy"})
    assert caps.batch_shape_policy == "homogeneous"
    assert rccsd_t_method_capabilities("ccsd(t)") == caps
    with pytest.raises(ValueError, match="unknown method"):
        rccsd_t_method_capabilities("rccsd")


@pytest.mark.parametrize("name", ["h2", "he", "h2o", "nh3", "ch4"])
def test_cpu_endpoint_composes_converged_ccsd_and_standard_triples(name):
    snapshot, provider, meta, _arrays = fixture_problem(name)
    result = rccsd_t_energy(snapshot, provider, backend="cpu", vir_chunk_size=1)

    assert result.converged, result.reason
    assert result.ccsd.converged
    assert result.triples is not None
    assert result.triples.peak_device_bytes == 0
    assert result.provenance["triples_evaluated"] is True
    assert result.provenance["ccsd_state_identity"]
    assert result.provenance["result_identity"]
    np.testing.assert_allclose(
        result.triples_energy,
        TRIPLES_REFERENCE[name],
        atol=1e-11,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.correlation_energy,
        result.ccsd_correlation_energy + result.triples_energy,
        atol=2e-13,
        rtol=0,
    )
    assert result.reference_energy == meta["hf_energy"]
    assert abs(result.ccsd_correlation_energy - meta["correlation_energy"]) <= 1e-8
    assert result.state.history[-1]["independent_r1_max"] <= 1e-9
    assert result.state.history[-1]["independent_r2_max"] <= 1e-9
    assert result.total_energy == result.reference_energy + result.correlation_energy
    nvir = result.state.t1.shape[1]
    assert (
        result.provenance["virtual_triple_count"] == nvir * (nvir + 1) * (nvir + 2) // 6
    )
    assert result.provenance["peak_device_bytes"] == 0
    assert result.provenance["memory"]["triples_input_bytes"] > 0
    assert result.provenance["memory"]["triples_cpu_workspace_peak_bytes"] is None
    timing = result.provenance["timing"]
    assert timing["ccsd_s"] > 0 and timing["triples_s"] > 0
    assert timing["endpoint_s"] >= timing["ccsd_s"] + timing["triples_s"]
    np.testing.assert_allclose(
        result.total_energy,
        meta["total_energy"] + TRIPLES_REFERENCE[name],
        atol=2e-8,
        rtol=0,
    )


def test_nonconverged_ccsd_never_publishes_a_triples_or_total_energy():
    snapshot, provider, _meta, _arrays = fixture_problem("h2")
    result = rccsd_t_energy(
        snapshot,
        provider,
        options=SolverOptions(max_iterations=1),
    )
    assert not result.converged
    assert result.status == "not_converged"
    assert result.ccsd_correlation_energy is not None
    assert result.triples_energy is None
    assert result.correlation_energy is None
    assert result.total_energy is None
    assert result.triples is None
    assert result.provenance["triples_evaluated"] is False


def test_force_and_nonproduction_cuda_backend_are_rejected_before_execution():
    snapshot, provider, _meta, _arrays = fixture_problem("h2")
    with pytest.raises(NotImplementedError, match="energy only"):
        rccsd_t_energy(snapshot, provider, compute_forces=True)
    with pytest.raises(ValueError, match="backend"):
        rccsd_t_energy(snapshot, provider, backend="cuda")
    with pytest.raises(ValueError, match="CudaCompilerAdapter"):
        rccsd_t_energy(snapshot, provider, backend="cuda-resident")


def test_endpoint_artifact_records_components_and_identity(tmp_path):
    snapshot, provider, _meta, _arrays = fixture_problem("h2o")
    result = rccsd_t_energy(snapshot, provider)
    path = tmp_path / "h2o-rccsd-t.json"
    record = result.write(path)
    assert record["schema"] == "vibeqc.rccsd-t.endpoint/1"
    assert record["record_hash"]
    assert record["provenance"]["ccsd_state_identity"]
    assert record["triples"]["tile_count"] == result.triples.tile_count
    assert path.read_text().endswith("\n")


def test_batch_force_request_is_rejected_before_item_execution():
    snapshot, provider, _meta, _arrays = fixture_problem("h2")
    with pytest.raises(NotImplementedError, match="energy only"):
        rccsd_t_batch_energy([(snapshot, provider)], compute_forces=True)


def test_homogeneous_batch_isolates_one_invalid_provider_and_keeps_order():
    s0, p0, _meta0, _arrays0 = fixture_problem("h2")
    s1, _p1, _meta1, arrays1 = fixture_problem("h2")
    result = rccsd_t_batch_energy(
        [
            (s0, p0),
            (s0, object()),
            (s1, FixtureProvider(s1, arrays1["g"])),
        ]
    )
    assert result.shape == (1, 1)
    assert [item.index for item in result.items] == [0, 1, 2]
    assert result.items[0].converged
    assert result.items[0].triples_energy is not None
    assert result.items[1].status == "error"
    assert result.items[1].result is None
    assert result.items[2].converged


def test_prepared_batch_rejects_ragged_cc_shapes_before_any_item_execution():
    h2 = fixture_problem("h2")
    h2o = fixture_problem("h2o")
    with pytest.raises(ValueError, match="homogeneous"):
        PreparedRCCSDTBatch([(h2[0], h2[1]), (h2o[0], h2o[1])])


def test_empty_prepared_batch_has_an_explicit_empty_shape():
    result = PreparedRCCSDTBatch([]).execute()
    assert result.shape is None
    assert result.items == ()
    assert rccsd_t_batch_energy(iter(())).items == ()


def test_prepared_batch_rejects_shared_amplitude_or_warm_start_state():
    snapshot, provider, _meta, arrays = fixture_problem("h2")
    with pytest.raises(ValueError, match="shared t1/t2 or warm_start"):
        PreparedRCCSDTBatch([(snapshot, provider)], t1=arrays["t1"], t2=arrays["t2"])
    with pytest.raises(ValueError, match="shared t1/t2 or warm_start"):
        PreparedRCCSDTBatch([(snapshot, provider)], warm_start=object())


def test_deterministic_identity_and_tile_partition():
    first = rccsd_t_energy(*fixture_problem("h2o")[:2])
    repeat = rccsd_t_energy(*fixture_problem("h2o")[:2])
    for key in ("ccsd_state_identity", "result_identity"):
        assert repeat.provenance[key] == first.provenance[key]
        assert len(repeat.provenance[key]) == 64
    different_tiles = rccsd_t_energy(*fixture_problem("h2o")[:2], vir_chunk_size=2)
    assert different_tiles.provenance["triples_tile_count"] == 1
    assert (
        different_tiles.provenance["ccsd_state_identity"]
        == first.provenance["ccsd_state_identity"]
    )
    assert (
        different_tiles.provenance["result_identity"]
        != first.provenance["result_identity"]
    )
    assert abs(different_tiles.triples_energy - first.triples_energy) <= 1e-13


def test_nonconvergence_does_not_execute_tiles(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api, "cpu_triples_tiles", lambda *a, **kw: pytest.fail("triples ran")
    )
    result = rccsd_t_energy(
        *fixture_problem()[:2], options=SolverOptions(max_iterations=1)
    )
    assert result.state.total_energy is not None
    assert (
        result.total_energy
        is result.correlation_energy
        is result.triples_energy
        is None
    )
    assert result.provenance["triples_tile_count"] == 0
    assert result.provenance["timing"]["triples_s"] == 0
    record = result.write(tmp_path / "failed.json")
    assert record["total_energy"] is None
    assert record["provenance"]["result_identity"]


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"compute_forces": True}, NotImplementedError, "energy only"),
        ({"backend": "cuda"}, ValueError, "backend"),
        ({"backend": "graph"}, ValueError, "backend"),
        ({"backend": "cuda-resident"}, ValueError, "CudaCompilerAdapter"),
        ({"vir_chunk_size": 0}, ValueError, "vir_chunk_size"),
        ({"triples_max_bytes": 0}, ValueError, "max_bytes"),
    ],
)
def test_request_rejection_precedes_ccsd_including_empty_batch(
    monkeypatch, kwargs, error, match
):
    monkeypatch.setattr(
        api, "rccsd_energy", lambda *a, **kw: pytest.fail("CCSD started")
    )
    with pytest.raises(error, match=match):
        rccsd_t_energy(None, None, **kwargs)
    with pytest.raises(error, match=match):
        rccsd_t_batch_energy([], **kwargs)
    with pytest.raises(error, match=match):
        PreparedRCCSDTBatch([], **kwargs)


def test_cuda_requires_path_cache():
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    with pytest.raises(ValueError, match="pathlib.Path"):
        rccsd_t_energy(
            None, None, backend="cuda-resident", compiler=compiler, cache="cache"
        )


def test_exact_replay_feeds_and_final_amplitudes(monkeypatch):
    snapshot, provider, _, _ = fixture_problem("h2o")
    cc = api.rccsd_energy(snapshot, provider)
    monkeypatch.setattr(provider, "get", lambda *a: pytest.fail("provider re-read"))
    monkeypatch.setattr(api, "rccsd_energy", lambda *a, **kw: cc)
    original = api.cpu_triples_tiles

    def check_inputs(nocc, nvir, arrays, **kwargs):
        assert (nocc, nvir) == (5, 2)
        assert arrays["t1"] is cc.t1 and arrays["t2"] is cc.t2
        for key in ("ovvv", "ovoo", "ovov", "fov"):
            np.testing.assert_array_equal(arrays[key], cc.state.replay_inputs[key])
        np.testing.assert_array_equal(
            np.r_[arrays["eps_o"], arrays["eps_v"]],
            cc.state.replay_inputs["orbital_energies"],
        )
        return original(nocc, nvir, arrays, **kwargs)

    monkeypatch.setattr(api, "cpu_triples_tiles", check_inputs)
    # A post-solve caller view cannot replace accepted replay orbital energies.
    caller_view = SimpleNamespace(
        nocc=snapshot.nocc,
        nmo=snapshot.nmo,
        orbital_energies=np.full(snapshot.nmo, np.nan),
    )
    result = rccsd_t_energy(caller_view, provider)
    assert abs(result.triples_energy - TRIPLES_REFERENCE["h2o"]) <= 1e-11


@pytest.mark.parametrize("accepted", [False, True])
def test_resident_handoff_requires_solved_identity(monkeypatch, tmp_path, accepted):
    """CPU-only dispatch test: a fake tile owner does not qualify GPU numerics."""
    snapshot, provider, _, _ = fixture_problem()
    cc = api.rccsd_energy(snapshot, provider)
    provenance = dict(cc.provenance, combined_peak_bytes=1024)
    if accepted:
        provenance["resident_solved_state_identity"] = "qualified-test-state"
    cc = replace(
        cc, backend="cuda-resident", state=replace(cc.state, provenance=provenance)
    )
    calls = []

    def cc_energy(*args, **kwargs):
        assert kwargs["backend"] == "cuda-resident"
        calls.append("ccsd")
        return cc

    class TileOwner:
        def __init__(self, config, compiler, cache):
            assert accepted, "unqualified CCSD reached CUDA tiles"
            assert config.nocc == config.nvir == 1
            assert config.max_bytes == 4096 and cache == tmp_path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def run_tiles(self, arrays, *, oracle, profile):
            assert oracle is False and profile is False
            assert arrays["t1"] is cc.t1 and arrays["t2"] is cc.t2
            return CudaTriplesResult(
                0.0,
                [0.0],
                1,
                1,
                1,
                1,
                512,
                artifact_keys=["tile-artifact"],
                peak_bytes_per_tile=[512],
                timing={"run_s": 0.01},
            )

    monkeypatch.setattr(api, "rccsd_energy", cc_energy)
    monkeypatch.setattr(api, "CudaTriplesTiles", TileOwner)
    monkeypatch.setattr(
        api, "cpu_triples_tiles", lambda *a, **kw: pytest.fail("CPU fallback")
    )
    kwargs = {
        "backend": "cuda-resident",
        "compiler": CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120")),
        "cache": tmp_path,
        "triples_max_bytes": 4096,
    }
    if not accepted:
        with pytest.raises(RuntimeError, match="resident_solved_state_identity"):
            rccsd_t_energy(snapshot, provider, **kwargs)
        assert calls == ["ccsd"]
    else:
        result = rccsd_t_energy(snapshot, provider, **kwargs)
        assert calls == ["ccsd", "closed"]
        assert (
            result.provenance["resident_solved_state_identity"]
            == provenance["resident_solved_state_identity"]
        )
        assert result.provenance["triples_artifact_keys"] == ("tile-artifact",)
        assert result.provenance["peak_device_bytes"] == 1024
        assert result.provenance["triples_peak_device_bytes"] == 512
        assert result.provenance["timing"]["triples_detail"] == {"run_s": 0.01}


def test_artifact_hash_and_nonfinite_rejection(tmp_path):
    result = rccsd_t_energy(*fixture_problem("h2o")[:2])
    path = tmp_path / "energy.json"
    result.write(path)
    text = path.read_text()
    record = json.loads(text)
    assert text.count("\n") == 1 and len(text) < 12000
    assert record.pop("record_hash") == canonical_hash(record)
    assert record["total_energy"] == result.total_energy
    assert not {"inputs", "t1", "t2", "history"} & record.keys()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(result, triples_energy=bad).write(path)
        assert path.read_text() == text


def test_homogeneous_batch_independent_state_and_repeatability():
    h2, hp, _, _ = fixture_problem("h2")
    he, ep, _, _ = fixture_problem("he")
    prepared = PreparedRCCSDTBatch([(h2, hp), (h2, object()), (he, ep), (h2, hp)])
    first = prepared.execute()
    assert [item.index for item in first.items] == list(range(4))
    assert [item.status for item in first.items] == [
        "converged",
        "error",
        "converged",
        "converged",
    ]
    assert first.items[0].result.reference_energy == h2.reference_energy
    assert first.items[2].result.reference_energy == he.reference_energy
    a, b = first.items[0].result.state, first.items[3].result.state
    assert a is not b and a.history is not b.history
    assert not np.shares_memory(a.t1, b.t1) and not np.shares_memory(a.t2, b.t2)
    repeat = prepared.execute()
    assert (
        first.items[0].result.provenance["result_identity"]
        == repeat.items[0].result.provenance["result_identity"]
    )
    assert a is not repeat.items[0].result.state
    with pytest.raises(NotImplementedError, match="energy only"):
        prepared.execute(compute_forces=True)


def test_batch_isolates_triples_failure_and_nonconvergence(monkeypatch):
    snapshot, provider, _, _ = fixture_problem()
    original = api.cpu_triples_tiles
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("tile budget exhausted")
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "cpu_triples_tiles", fail_second)
    results = rccsd_t_batch_energy([(snapshot, provider)] * 3)
    assert [item.status for item in results.items] == [
        "converged",
        "error",
        "converged",
    ]
    assert "tile budget" in results.items[1].reason
    failed = rccsd_t_batch_energy(
        [(snapshot, provider)] * 2, options=SolverOptions(max_iterations=1)
    )
    assert calls == 3
    assert all(
        item.status == "not_converged" and item.total_energy is None
        for item in failed.items
    )


def test_ragged_preflight_never_starts_a_solver(monkeypatch):
    monkeypatch.setattr(
        api, "rccsd_energy", lambda *a, **kw: pytest.fail("ragged batch ran CCSD")
    )
    with pytest.raises(ValueError, match="homogeneous"):
        PreparedRCCSDTBatch([fixture_problem("h2")[:2], fixture_problem("h2o")[:2]])

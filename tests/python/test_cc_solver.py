"""CPU CCSD convergence, independent endpoints and explicit failure states."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tools.cc_endpoint_fixtures import load, snapshot_from_fixture, source_arguments
from tools.vibeqc_cc.solver import PreparedCCSD, SolverOptions, solve
from tools.vibeqc_posthf import MOBlock
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.providers import BlockResult, ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource


class FixtureProvider(ConventionalProvider):
    """Test-only exact MO inputs: exercises the solver without native libraries."""

    def __init__(self, snapshot, g):
        self.snapshot = snapshot
        self.g = g
        self.backend = "cpu"
        self.calls = 0
        self.source = SimpleNamespace(_check_open=lambda: None)

    def get(self, block):
        self.calls += 1
        return BlockResult(
            block,
            self.g[np.ix_(*block.slots)],
            self.snapshot.identity,
            self.snapshot.hamiltonian_id,
            {},
        )


def fixture_problem(name="h2"):
    meta, a = load(name)
    # Geometry/basis identity is irrelevant to this supplied-integral unit test;
    # the real native-provider tests below verify that boundary independently.
    source = SimpleNamespace(
        electron_count=int(a["occ"].sum()),
        geometry_hash="fixture-" + name,
        basis_hash="fixture-basis-" + name,
    )
    s = snapshot_from_fixture(source, meta, a)
    return s, FixtureProvider(s, a["g"]), meta, a


@pytest.mark.parametrize("name", ["h2", "he", "h2o", "nh3", "ch4"])
def test_same_C_solver_against_pinned_ccsd_and_two_electron_fci(name):
    s, p, meta, a = fixture_problem(name)
    result = solve(
        s, p, options=SolverOptions(residual_tolerance=1e-10, energy_tolerance=1e-12)
    )
    assert result.converged, (result.reason, result.history[-1])
    assert abs(result.total_energy - meta["total_energy"]) <= 1e-8
    np.testing.assert_allclose(result.t1, a["t1"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(result.t2, a["t2"], atol=1e-8, rtol=1e-8)
    assert (
        max(
            result.history[-1]["independent_r1_max"],
            result.history[-1]["independent_r2_max"],
        )
        <= 1e-10
    )
    if "fci_total_energy" in meta:
        assert abs(result.total_energy - meta["fci_total_energy"]) <= 1e-8


@pytest.mark.parametrize("shift,damping,diis", [(0.4, 0.15, 6), (0.0, 0.2, 0)])
def test_shift_damping_and_diis_do_not_change_target_root(shift, damping, diis):
    s, p, meta, _ = fixture_problem()
    result = solve(
        s, p, options=SolverOptions(level_shift=shift, damping=damping, diis_size=diis)
    )
    assert result.converged
    assert abs(result.total_energy - meta["total_energy"]) <= 1e-8


def test_preflight_failure_happens_before_integrals_and_iteration_failures_replay(
    tmp_path, monkeypatch
):
    s, p, _meta, a = fixture_problem()
    from tools.vibeqc_cc import evaluate

    with pytest.raises(ValueError):
        evaluate(s, p, a["t1"] * np.nan, a["t2"])
    with pytest.raises(ValueError):
        evaluate(s, p, a["t1"], a["t2"], max_bytes=1)
    assert p.calls == 0
    for kw in (
        {"t1": a["t1"].astype(complex), "t2": a["t2"]},
        {"t1": a["t1"] * np.nan, "t2": a["t2"]},
        {"t1": a["t1"]},
        {"options": SolverOptions(max_bytes=1)},
    ):
        with pytest.raises(ValueError):
            solve(s, p, **kw)
        assert p.calls == 0
    near = replace(
        s,
        orbital_energies=np.zeros_like(s.orbital_energies),
        fock=np.zeros_like(s.fock),
    )
    with pytest.raises(ValueError, match="denominator"):
        solve(near, FixtureProvider(near, a["g"]))
    with pytest.raises(ValueError, match="conventional CPU"):
        solve(s, object())
    for bad_options in (False, 0, {}):
        with pytest.raises(TypeError, match="options"):
            solve(s, p, options=bad_options)
    with pytest.raises(ValueError, match="identity"):
        solve(replace(s, generation_id="different"), p)
    with pytest.raises(ValueError, match="frozen"):
        replace(s, frozen_mask=(0,))
    with pytest.raises(ValueError):
        replace(s, algorithm="UHF")
    with pytest.raises(ValueError):
        replace(s, converged=False)
    with pytest.raises(ValueError):
        replace(s, electron_count=2 * s.nmo)
    failed = solve(s, p, options=SolverOptions(max_iterations=1))
    assert failed.status == "not_converged"
    failed.write(tmp_path / "failure.json")
    saved = json.loads((tmp_path / "failure.json").read_text())
    assert len(saved["history"]) == 2 and saved["inputs"]["initial_t2"]
    from tools.replay_ccsd import replay

    reproduced = replay(tmp_path / "failure.json")
    assert reproduced.status == failed.status
    np.testing.assert_array_equal(reproduced.t1, failed.t1)
    np.testing.assert_array_equal(reproduced.t2, failed.t2)
    saved["provenance"]["options"]["max_iterations"] = 2
    (tmp_path / "tampered.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="record identity"):
        replay(tmp_path / "tampered.json")
    original = PreparedCCSD.evaluate
    calls = 0

    def bad(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FloatingPointError("synthetic numerical overflow")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PreparedCCSD, "evaluate", bad)
    nonfinite = solve(s, p)
    assert nonfinite.status == "nonfinite" and not nonfinite.converged
    nonfinite.write(tmp_path / "nonfinite.json")


def test_false_shared_residual_cannot_bypass_expanded_acceptance(monkeypatch):
    s, p, _meta, _ = fixture_problem()
    original = PreparedCCSD.evaluate

    def wrong_shared(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not kwargs.get("independent", False):
            result["singles_residual"] *= 0
            result["doubles_residual"] *= 0
        return result

    monkeypatch.setattr(PreparedCCSD, "evaluate", wrong_shared)
    result = solve(s, p, options=SolverOptions(max_iterations=2))
    assert not result.converged
    assert result.history[-1]["independent_r2_max"] > 1e-9


def test_nonfinite_initial_equation_has_no_fabricated_energy_and_replays(tmp_path):
    s, p, _meta, a = fixture_problem()
    result = solve(
        s, p, t1=np.full_like(a["t1"], 1e150), t2=np.full_like(a["t2"], 1e150)
    )
    assert result.status == "nonfinite" and result.correlation_energy is None
    result.write(tmp_path / "overflow.json")
    from tools.replay_ccsd import replay

    assert replay(tmp_path / "overflow.json").status == "nonfinite"


def test_collective_provider_budget_rejects_before_any_read_and_accepts_cache_hits():
    s, _p, _meta, a = fixture_problem()
    source = SimpleNamespace(
        nbf=s.nmo,
        shell_sizes=(1,) * s.nmo,
        numeric_bytes=0,
        geometry_hash=s.geometry_hash,
        basis_hash=s.basis_hash,
        representation=s.representation,
        identity="budget-test-source",
        _check_open=lambda: None,
    )
    source.requests = lambda *args, **kwargs: pytest.fail(
        "AO read before complete budget preflight"
    )
    provider = ConventionalProvider(s, source)
    names = ("ovov", "ovvo", "oovv", "ovvv", "ovoo", "oooo", "vvvv")
    blocks = [MOBlock.from_spaces(s, name) for name in names]
    # Every individual block fits, but their pinned collection cannot fit.
    provider.budget_bytes = max(provider.plan(b).peak_bytes for b in blocks)
    with pytest.raises(MemoryError, match="complete provider block set"):
        PreparedCCSD(s, provider)
    assert (
        provider.statistics["transformations"] == 0
        and provider.statistics["source_tiles"] == 0
    )
    assert provider._retained == 0 and not provider._cache
    # An already warm cache fits the same budget and must not be rejected by
    # a cold-cache worst-case estimate. Actual provider.get serves all hits.
    for block in blocks:
        values = a["g"][np.ix_(*block.slots)]
        provider._cache[(s.identity, source.identity, block.slots)] = (
            BlockResult(block, values, s.identity, s.hamiltonian_id, {}),
            values.nbytes,
        )
        provider._retained += values.nbytes
    prepared = PreparedCCSD(s, provider)
    assert provider.statistics["hits"] == 7
    assert np.isfinite(prepared.evaluate(*prepared.initial)["correlation_energy"])


@pytest.mark.parametrize("name", ["h2", "he", "h2o", "nh3", "ch4"])
def test_fresh_native_HF_to_converged_CCSD(name):
    meta, a = load(name)
    try:
        source = NativeSource(**source_arguments(meta["inputs"]))
    except (OSError, FileNotFoundError) as error:
        pytest.skip(str(error))
    except RuntimeError as error:
        if "native library was not found" not in str(error):
            raise
        pytest.skip(str(error))
    with source:
        s, _ = export_rhf(source, tolerance=1e-12, max_iterations=150)
        with ConventionalProvider(s, source) as provider:
            result = solve(
                s,
                provider,
                options=SolverOptions(residual_tolerance=1e-10, energy_tolerance=1e-12),
            )
        assert result.converged, (name, result.history[-1])
        assert abs(result.total_energy - meta["total_energy"]) <= 1e-8
        # Align occupied/virtual subspaces, not individual orbital signs only.
        U = a["C"].T @ a["S"] @ s.coefficients
        o = s.nocc
        assert np.max(np.abs(U[:o, o:])) < 1e-7
        t1 = np.einsum("ki,kc,ca->ia", U[:o, :o], a["t1"], U[o:, o:])
        t2 = np.einsum(
            "ki,lj,klcd,ca,db->ijab",
            U[:o, :o],
            U[:o, :o],
            a["t2"],
            U[o:, o:],
            U[o:, o:],
        )
        np.testing.assert_allclose(result.t1, t1, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(result.t2, t2, atol=1e-8, rtol=1e-8)

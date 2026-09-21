# ruff: noqa: ANN001, ANN201, ANN202
"""Factorized DF-RCCSD residual and convergence tests for #157 slice B."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.tensor import execute

from tools.cc_endpoint_fixtures import load, snapshot_from_fixture
from tools.vibeqc_cc.df_ccsdt_oracle import (
    correlation_df_reference,
    dense_df_oracle_from_three_index,
    run_dense_df_ccsdt_oracle,
)
from tools.vibeqc_cc.df_factorized import (
    solve_df_ccsd,
    virtual_corrections,
)
from tools.vibeqc_cc.doubles import build_ccsd_program
from tools.vibeqc_cc.solver import SolverOptions
from tools.vibeqc_posthf.df import DFProvider, MetricFactor
from tools.vibeqc_posthf.providers import BlockResult
from tools.vibeqc_posthf.reference import immutable


def _factor_eri(eri: np.ndarray) -> np.ndarray:
    nmo = eri.shape[0]
    matrix = eri.reshape(nmo * nmo, nmo * nmo)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    assert values.min() > -1e-11
    keep = values > 1e-12
    return (np.sqrt(values[keep])[:, None] * vectors[:, keep].T).reshape(
        int(np.sum(keep)), nmo, nmo
    )


def _method_problem(name="h2o"):
    meta, arrays = load(name)
    nmo = arrays["g"].shape[0]
    identity = SimpleNamespace(
        electron_count=int(arrays["occ"].sum()),
        geometry_hash="df-factorized-geometry-" + name,
        basis_hash="df-factorized-basis-" + name,
    )
    reference = snapshot_from_fixture(identity, meta, arrays)
    b = _factor_eri(arrays["g"])
    source = SimpleNamespace(
        geometry_hash=reference.geometry_hash,
        basis_hash=reference.basis_hash,
        auxiliary_hash="df-factorized-aux-" + name,
        representation=reference.representation,
        nbf=nmo,
        naux=b.shape[0],
    )
    metric = MetricFactor(
        np.eye(source.naux),
        source.auxiliary_hash,
        source.geometry_hash,
        1e-12,
        source.naux,
        1e-12,
        1.0,
        "synthetic-factorized-metric-hash-" + name,
        "synthetic-factorized-metric-" + name,
        0.0,
        8 * source.naux**2,
        0,
    )
    correlated, contract = correlation_df_reference(reference, source, metric)
    return correlated, contract, b, arrays


class FixtureDFProvider(DFProvider):
    """Same-Hamiltonian fixture provider with real DF budget reservations."""

    def __init__(self, snapshot, b, eri, *, budget_bytes=256 << 20) -> None:
        self.snapshot = snapshot
        self.b = immutable(b)
        self.g = immutable(eri)
        self.budget_bytes = budget_bytes
        self._cache = {}
        self._retained = 0
        self._external_reserved = 0
        self._lock = threading.RLock()
        self._closed = False
        self.calls = []
        self.source = SimpleNamespace(_check_open=lambda: None)
        self.statistics = {
            "peak_bytes": 0,
            "external_reserved_bytes": 0,
            "transformations": 0,
            "hits": 0,
        }

    def three_index(
        self,
        p,
        q,
        *,
        auxiliary_begin=0,
        auxiliary_count=None,
    ):
        self._check()
        count = (
            self.b.shape[0] - auxiliary_begin
            if auxiliary_count is None
            else auxiliary_count
        )
        result = self.b[auxiliary_begin : auxiliary_begin + count][:, tuple(p)][
            :, :, tuple(q)
        ]
        peak = self._retained + self._external_reserved + result.nbytes
        if peak > self.budget_bytes:
            raise MemoryError("fixture DF B budget exceeded")
        self.statistics["peak_bytes"] = max(self.statistics["peak_bytes"], peak)
        return result

    def get(self, block):
        self._check()
        block.validate(self.snapshot)
        self.calls.append(block.slots)
        if block.slots in self._cache:
            self.statistics["hits"] += 1
            return self._cache[block.slots]
        values = immutable(self.g[np.ix_(*block.slots)])
        peak = self._retained + self._external_reserved + values.nbytes
        if peak > self.budget_bytes:
            raise MemoryError("fixture DF block budget exceeded")
        result = BlockResult(
            block,
            values,
            self.snapshot.identity,
            self.snapshot.hamiltonian_id,
            {"backend": "fixture-df"},
        )
        self._cache[block.slots] = result
        self._retained += values.nbytes
        self.statistics["peak_bytes"] = max(self.statistics["peak_bytes"], peak)
        self.statistics["transformations"] += 1
        return result


def test_factorized_virtual_corrections_reproduce_full_expanded_tensorir():
    rng = np.random.default_rng(157)
    nocc, nvir, naux = 2, 3, 5
    nmo = nocc + nvir
    raw = rng.normal(size=(naux, nmo, nmo))
    b = 0.5 * (raw + raw.transpose(0, 2, 1))
    eri = np.einsum("Qpq,Qrs->pqrs", b, b, optimize=True)
    fock = rng.normal(size=(nmo, nmo))
    fock = 0.5 * (fock + fock.T)
    t1 = rng.normal(scale=0.03, size=(nocc, nvir))
    raw_t2 = rng.normal(scale=0.02, size=(nocc, nocc, nvir, nvir))
    t2 = 0.5 * (raw_t2 + raw_t2.transpose(1, 0, 3, 2))
    spaces = {
        "o": tuple(range(nocc)),
        "v": tuple(range(nocc, nmo)),
    }

    def block(name):
        return eri[np.ix_(*(spaces[axis] for axis in name))]

    full_feeds = {
        "foo": fock[:nocc, :nocc],
        "fov": fock[:nocc, nocc:],
        "fvv": fock[nocc:, nocc:],
        "ovov": block("ovov"),
        "ovvo": block("ovvo"),
        "oovv": block("oovv"),
        "ovvv": block("ovvv"),
        "ovoo": block("ovoo"),
        "oooo": block("oooo"),
        "vvvv": block("vvvv"),
        "t1": t1,
        "t2": t2,
    }
    full = build_ccsd_program(nocc, nvir, form="expanded", diagnostics=False)
    expected = execute(full, full_feeds, max_bytes=1 << 28).outputs

    r1, r2 = virtual_corrections(
        b[:, :nocc, nocc:],
        b[:, nocc:, nocc:],
        t1,
        t2,
    )
    factorized = build_ccsd_program(
        nocc,
        nvir,
        form="expanded",
        diagnostics=False,
        external_virtual_correction=True,
    )
    feeds = {
        key: value for key, value in full_feeds.items() if key not in ("ovvv", "vvvv")
    }
    feeds.update(df_virtual_singles=r1, df_virtual_doubles=r2)
    actual = execute(factorized, feeds, max_bytes=1 << 28).outputs

    assert "ovvv" not in {
        node.attrs["name"] for node in factorized.live_nodes if node.op == "input"
    }
    assert "vvvv" not in {
        node.attrs["name"] for node in factorized.live_nodes if node.op == "input"
    }
    np.testing.assert_allclose(
        actual["correlation_energy"], expected["correlation_energy"], atol=2e-13, rtol=0
    )
    np.testing.assert_allclose(
        actual["singles_residual"], expected["singles_residual"], atol=2e-12, rtol=0
    )
    np.testing.assert_allclose(
        actual["doubles_residual"], expected["doubles_residual"], atol=3e-12, rtol=0
    )


@pytest.mark.parametrize("name", ["h2", "h2o"])
def test_factorized_df_ccsd_converges_to_same_hamiltonian_dense_oracle(name):
    snapshot, contract, b, arrays = _method_problem(name)
    dense = run_dense_df_ccsdt_oracle(
        dense_df_oracle_from_three_index(snapshot, contract, b)
    )
    provider = FixtureDFProvider(snapshot, b, arrays["g"])
    result = solve_df_ccsd(
        snapshot,
        provider,
        contract,
        options=SolverOptions(
            residual_tolerance=1e-10,
            energy_tolerance=1e-12,
        ),
    )

    assert result.converged, (result.reason, result.history[-1])
    np.testing.assert_allclose(
        result.correlation_energy,
        dense.ccsd_correlation_energy,
        atol=2e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.total_energy,
        dense.result.ccsd.total_energy,
        atol=2e-10,
        rtol=0,
    )
    np.testing.assert_allclose(result.t1, dense.result.ccsd.t1, atol=2e-9, rtol=0)
    np.testing.assert_allclose(result.t2, dense.result.ccsd.t2, atol=2e-9, rtol=0)
    assert result.provenance["factorized_blocks"] == ("ovvv", "vvvv")
    assert result.provenance["fock_policy"] == "preserve-conventional-rhf"
    assert provider.statistics["external_reserved_bytes"] == 0
    assert not provider._cache

    occupied = set(range(snapshot.nocc))
    virtual = set(range(snapshot.nocc, snapshot.nmo))
    for slots in provider.calls:
        spaces = tuple(
            "v"
            if set(slot).issubset(virtual) and slot
            else "o"
            if set(slot).issubset(occupied) and slot
            else "?"
            for slot in slots
        )
        assert spaces not in (("o", "v", "v", "v"), ("v", "v", "v", "v"))


def test_df_external_reservation_is_budgeted_and_released():
    snapshot, _contract, b, arrays = _method_problem("h2")
    provider = FixtureDFProvider(snapshot, b, arrays["g"], budget_bytes=4096)
    provider.reserve_external(1024)
    assert provider.statistics["external_reserved_bytes"] == 1024
    with pytest.raises(MemoryError, match="external reservation"):
        provider.reserve_external(4096)
    provider.release_external(1024)
    assert provider.statistics["external_reserved_bytes"] == 0
    with pytest.raises(ValueError, match="release"):
        provider.release_external(1)

"""Experimental RCCSD facade, capability record and isolated item batches.

These tests exercise the CPU facade end to end from committed HF/CCSD endpoints,
with no native libraries and no GPU. The CUDA backend of the same facade is
covered by the real-device tests in ``test_cc_gpu_solver.py``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tools.cc_endpoint_fixtures import load, snapshot_from_fixture
from tools.vibeqc_cc import (
    batch_energy,
    energy,
    method_capabilities,
)
from tools.vibeqc_posthf.providers import BlockResult, ConventionalProvider


class FixtureProvider(ConventionalProvider):
    """Test-only exact MO inputs: exercises the API without native libraries."""

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
    meta, a = load(name)
    source = SimpleNamespace(
        electron_count=int(a["occ"].sum()),
        geometry_hash="fixture-" + name,
        basis_hash="fixture-basis-" + name,
    )
    s = snapshot_from_fixture(source, meta, a)
    return s, FixtureProvider(s, a["g"]), meta, a


def test_capabilities_are_energy_only_and_not_a_native_batch():
    caps = method_capabilities("rccsd")
    assert caps.method == "rccsd"
    assert caps.family == "coupled_cluster"
    assert caps.available is True
    assert caps.supports_batch is False
    assert caps.supported_properties == frozenset({"energy"})
    with pytest.raises(ValueError, match="unknown method"):
        method_capabilities("rhf")


@pytest.mark.parametrize("name", ["h2", "he", "h2o", "nh3", "ch4"])
def test_energy_cpu_converges_to_pinned_endpoint(name):
    s, p, meta, a = fixture_problem(name)
    result = energy(s, p)
    assert result.backend == "cpu"
    assert result.converged, (result.reason, result.history[-1])
    assert abs(result.total_energy - meta["total_energy"]) <= 1e-8
    assert result.correlation_energy is not None
    assert (
        abs(result.reference_energy + result.correlation_energy - result.total_energy)
        <= 1e-11
    )
    assert result.final_r1_max <= 1e-9
    assert result.final_r2_max <= 1e-9
    np.testing.assert_allclose(result.t1, a["t1"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(result.t2, a["t2"], atol=1e-8, rtol=1e-8)


def test_energy_rejects_forces_and_unknown_backend():
    s, p, _meta, _ = fixture_problem()
    with pytest.raises(NotImplementedError, match="energy only"):
        energy(s, p, compute_forces=True)
    with pytest.raises(ValueError, match="backend"):
        energy(s, p, backend="graph")
    with pytest.raises(ValueError, match="CudaCompilerAdapter"):
        energy(s, p, backend="cuda")


def test_nonconvergence_is_a_failed_energy_result_not_an_exception():
    s, p, _meta, _ = fixture_problem()
    from tools.vibeqc_cc import SolverOptions

    result = energy(s, p, options=SolverOptions(max_iterations=1))
    assert result.status == "not_converged"
    assert not result.converged
    assert result.total_energy is not None  # last finite state retained


def test_batch_isolates_failure_and_preserves_input_order():
    s, p, _meta, _ = fixture_problem("h2")
    problems = [
        (s, p),
        (s, object()),  # invalid provider must not corrupt neighbors
        (s, FixtureProvider(s, _load_g("h2"))),
    ]
    result = batch_energy(problems)
    assert [item.index for item in result.items] == [0, 1, 2]
    assert result.items[0].converged
    assert result.items[1].status == "error" and result.items[1].state is None
    assert result.items[2].converged


def _load_g(name):
    _, a = load(name)
    return a["g"]

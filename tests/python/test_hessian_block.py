"""Bounded multi-RHS and full conventional RHF Hessian assembly gates."""

from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_hessian import (
    NativeRHFState,
    analytic_hessian,
    rhf_hessian,
    rhf_hvp_many,
)
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.hessian_fixtures import fixture_inputs


@pytest.fixture(scope="module")
def h2_case():
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        dense = analytic_hessian(state)["total"]
        rng = np.random.default_rng(1804)
        directions = rng.normal(size=(3, state.nat, 3))
        directions /= np.linalg.norm(directions.reshape(3, -1), axis=1)[:, None, None]
        yield state, dense, directions


def test_hvp_many_recycled_matches_dense_columns_and_reports_shared_solve(h2_case):
    state, dense, directions = h2_case
    result = rhf_hvp_many(state, directions, strategy="recycled")
    expected = np.stack(
        [np.einsum("abxy,by->ax", dense, vector) for vector in directions]
    )
    np.testing.assert_allclose(result.values, expected, atol=1e-9, rtol=4e-10)
    diag = result.diagnostics
    assert diag["nrhs"] == len(directions)
    assert diag["strategy"] == "recycled"
    assert diag["complete_numeric_peak_bound_bytes"] <= diag["total_budget_bytes"]
    assert diag["response_peak_workspace_bytes"] > 0
    assert result.response_batch.solve_result.converged
    assert len(result.response_batch.responses) == len(directions)
    for value in (result.directions, result.values, *result.components.values()):
        assert not value.flags.writeable


def test_hvp_many_uses_solve_many_not_single_solver(h2_case, monkeypatch):
    from tools.vibeqc_hessian import perturbation

    state, dense, directions = h2_case

    def forbidden(*args, **kwargs):
        raise AssertionError("block Hessian response used the scalar solve path")

    monkeypatch.setattr(perturbation, "solve", forbidden)
    result = rhf_hvp_many(state, directions[:2], strategy="blocked")
    expected = np.stack(
        [np.einsum("abxy,by->ax", dense, vector) for vector in directions[:2]]
    )
    np.testing.assert_allclose(result.values, expected, atol=1e-9, rtol=4e-10)
    assert result.response_batch.solve_result.strategy == "blocked"


def test_full_hessian_blocked_matches_independent_dense_reference(h2_case):
    state, dense, _ = h2_case
    result = rhf_hessian(state, block_size=2, strategy="recycled")
    expected = dense.transpose(0, 2, 1, 3).reshape(6, 6)
    np.testing.assert_allclose(result.matrix, expected, atol=1e-9, rtol=4e-10)
    diag = result.diagnostics
    assert diag["block_size"] == 2
    assert diag["block_count"] == 3
    assert diag["raw_symmetry_error"] < 2e-9
    assert not diag["posthoc_symmetrization"]
    assert not result.matrix.flags.writeable
    assert diag["complete_numeric_peak_bound_bytes"] <= diag["total_budget_bytes"]
    assert diag["output_peak_bound_bytes"] == 3 * diag["output_bytes"]


def test_block_budget_rejects_before_first_integral_work(h2_case, monkeypatch):
    from tools.vibeqc_hessian import block

    state, _, directions = h2_case

    def forbidden(*args, **kwargs):
        raise AssertionError("insufficient block budget reached provider work")

    monkeypatch.setattr(block, "generated_directional_first_order", forbidden)
    with pytest.raises(ValueError, match="persistent numeric storage"):
        rhf_hvp_many(state, directions[:1], total_budget_bytes=1)


def test_full_output_budget_rejects_without_partial_hessian(h2_case, monkeypatch):
    from tools.vibeqc_hessian import block

    state, _, _ = h2_case

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized full Hessian request started a block")

    monkeypatch.setattr(block, "rhf_hvp_many", forbidden)
    output_bytes = (3 * state.nat) ** 2 * 8
    with pytest.raises(ValueError, match="full Hessian output"):
        rhf_hessian(state, total_budget_bytes=output_bytes)


@pytest.fixture
def assembly_only_state(monkeypatch):
    """Exercise assembly ownership without allocating a native RHF source."""
    from types import SimpleNamespace

    from tools.vibeqc_hessian import block

    class AssemblyState:
        nat = 2
        nbf = 2
        nocc = 1
        source = SimpleNamespace(identity="test-source")
        reference = SimpleNamespace(identity="test-reference")

        def validate(self):
            pass

    monkeypatch.setattr(block, "NativeRHFState", AssemblyState)
    return AssemblyState()


def test_full_hessian_releases_previous_block_before_next_call(
    assembly_only_state, monkeypatch
):
    import weakref

    from tools.vibeqc_hessian import block

    previous = []

    class BlockResult:
        identity = "test-block"

        def __init__(self, directions):
            self.diagnostics = {"complete_numeric_peak_bound_bytes": 0}
            self.values = directions.copy()

    def evaluate(state, directions, **kwargs):
        assert all(ref() is None for ref in previous), "previous block is still live"
        result = BlockResult(directions)
        previous.append(weakref.ref(result))
        return result

    monkeypatch.setattr(block, "rhf_hvp_many", evaluate)
    result = block.rhf_hessian(assembly_only_state, block_size=2)
    np.testing.assert_array_equal(result.matrix, np.eye(6))
    assert all(ref() is None for ref in previous)


def test_full_hessian_reserves_output_publication_before_any_block(
    assembly_only_state, monkeypatch
):
    from tools.vibeqc_hessian import block

    def forbidden(*args, **kwargs):
        pytest.fail("output publication budget was not preflighted")

    monkeypatch.setattr(block, "rhf_hvp_many", forbidden)
    output_bytes = (3 * assembly_only_state.nat) ** 2 * 8
    with pytest.raises(ValueError, match="full Hessian output"):
        block.rhf_hessian(
            assembly_only_state, block_size=2, total_budget_bytes=output_bytes + 1
        )


@pytest.mark.parametrize("natoms", [1, 2])
def test_full_hessian_default_block_size_adapts_to_coordinate_count(
    assembly_only_state, monkeypatch, natoms
):
    from types import SimpleNamespace

    from tools.vibeqc_hessian import block

    state = assembly_only_state
    state.nat = natoms
    coordinates = 3 * natoms
    block_sizes = []

    def evaluate(state, directions, **kwargs):
        block_sizes.append(len(directions))
        return SimpleNamespace(
            values=directions.copy(),
            identity="test-block",
            diagnostics={"complete_numeric_peak_bound_bytes": 0},
        )

    monkeypatch.setattr(block, "rhf_hvp_many", evaluate)
    result = block.rhf_hessian(state)
    np.testing.assert_array_equal(result.matrix, np.eye(coordinates))
    assert result.diagnostics["block_size"] == min(4, coordinates)
    assert block_sizes[0] == min(4, coordinates)
    with pytest.raises(ValueError, match="block_size"):
        block.rhf_hessian(state, block_size=coordinates + 1)


def test_cuda_relaxation_budget_rejects_before_response_work(h2_case, monkeypatch):
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info

    from tools.vibeqc_hessian import block

    state, _, directions = h2_case
    compiler = CudaCompilerAdapter(Path("/bin/false"), cuda_target_info("sm_80"))

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA relaxation budget failure reached response work")

    monkeypatch.setattr(block, "NativeJKBackend", forbidden)
    with pytest.raises(MemoryError, match="relaxation numeric storage"):
        rhf_hvp_many(
            state,
            directions[:1],
            relaxation_backend="cuda",
            relaxation_compiler=compiler,
            relaxation_budget_bytes=1,
        )

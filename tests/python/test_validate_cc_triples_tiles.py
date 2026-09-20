"""Qualification gates tested without compiling or executing CUDA code."""

import json
import typing
from types import SimpleNamespace

import numpy as np
import pytest

from tools import validate_cc_triples_tiles as validator


@pytest.fixture
def qualification_args(tmp_path: typing.Any, monkeypatch: typing.Any) -> typing.Any:
    """Keep provenance/compiler setup local while exercising the real driver."""
    monkeypatch.setattr(
        validator,
        "CudaCompilerAdapter",
        lambda *a, **kw: SimpleNamespace(
            target=SimpleNamespace(to_payload=lambda: {"test_only": True})
        ),
    )
    monkeypatch.setattr(validator, "tensor_source_identity", dict)
    monkeypatch.setattr(
        validator,
        "_qualification_source_identity",
        lambda: {"worktree_dirty": False, "test_only": True},
    )
    return SimpleNamespace(
        output=tmp_path / "results",
        cache=tmp_path / "cache",
        nvcc=tmp_path / "nvcc",
        architecture="sm_120",
        compile_timeout=1,
        budget="256,512",
        molecules="h2",
        compile_only=False,
    )


def _manifest(args: typing.Any) -> typing.Any:
    return json.loads((args.output / "manifest.json").read_text())


@pytest.mark.parametrize("compile_only", [False, True])
def test_all_infeasible_budgets_fail_qualification(
    qualification_args: typing.Any, monkeypatch: typing.Any, compile_only: typing.Any
) -> None:
    """An empty set of executed plans cannot satisfy the requested gates."""
    from vibeqc_compiler.tensor import cuda_plan

    def reject(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise ValueError("infeasible: test budget is below the plan peak")

    qualification_args.compile_only = compile_only
    qualification_args.budget = "1,2"
    monkeypatch.setattr(validator.CudaTriplesTiles, "run_tiles", reject)
    monkeypatch.setattr(cuda_plan, "plan_cuda", reject)
    with pytest.raises(SystemExit) as exc:
        validator.run(qualification_args)
    assert exc.value.code == 1
    manifest = _manifest(qualification_args)
    results = manifest["molecules"][0]["tile_results"]
    assert len(results) == 2
    assert all(not r["compiled"] and r["gpu_run"] is None for r in results)


def _mock_gpu_result(
    owner: typing.Any, arrays: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """Return a CPU scalar only inside these driver tests, never qualification."""
    if owner.config.max_bytes < 256 * (1 << 20):
        raise ValueError("infeasible: test budget is below the plan peak")
    et = validator.triples_energy(
        owner.config.nocc, owner.config.nvir, *arrays.values()
    )
    return SimpleNamespace(
        et=float(et),
        per_tile=[float(et)],
        per_tile_masked_cpu=[np.float64(et)],
        peak_bytes_per_tile=[1024],
        peak_device_bytes=1024,
        artifact_keys=["test-only"],
        runtime_device={"test_only": True},
        tile_count=1,
        timing={},
    )


def test_partial_budget_failure_fails_qualification(
    qualification_args: typing.Any, monkeypatch: typing.Any
) -> None:
    """A passing large budget must not hide a failed constrained budget."""
    qualification_args.budget = "1,256"
    monkeypatch.setattr(validator.CudaTriplesTiles, "run_tiles", _mock_gpu_result)
    with pytest.raises(SystemExit) as exc:
        validator.run(qualification_args)
    assert exc.value.code == 1
    results = _manifest(qualification_args)["molecules"][0]["tile_results"]
    assert [r["compiled"] for r in results] == [False, True]


def test_successful_gates_are_json_booleans(
    qualification_args: typing.Any, monkeypatch: typing.Any
) -> None:
    """NumPy comparisons must serialize as booleans, not the string 'True'."""
    monkeypatch.setattr(validator.CudaTriplesTiles, "run_tiles", _mock_gpu_result)
    validator.run(qualification_args)
    manifest = _manifest(qualification_args)
    assert manifest["selected_molecules"] == ["h2"]
    results = manifest["molecules"][0]["tile_results"]
    assert len(results) == 2  # nvir=1's repeated chunk choice is deduplicated.
    for result in results:
        for key in (
            "de_total_ok",
            "per_tile_ok",
            "budget_ok",
            "determinism_ok",
            "per_tile_bitwise_equal",
            "total_bitwise_equal",
        ):
            assert result["gpu_run"][key] is True


@pytest.mark.parametrize("molecules", ["", "h2,typo"])
def test_invalid_endpoint_selection_fails(
    qualification_args: typing.Any, molecules: typing.Any
) -> None:
    qualification_args.molecules = molecules
    with pytest.raises(ValueError, match="molecules must be"):
        validator.run(qualification_args)

"""Qualified CUDA meta-GGA must survive parent-branch integration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_qualified_mgga_reaches_native_state_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vibeqc import _stationary_cuda as runtime

    contract = SimpleNamespace(family="mgga", validate=lambda state: None)
    monkeypatch.setattr(
        runtime, "StationaryDerivativeContract", lambda identity: contract
    )
    state = SimpleNamespace(identity=object(), _source=SimpleNamespace(backend="cpu"))
    with pytest.raises(NotImplementedError, match="requires a native CUDA KS state"):
        runtime.complete_rks_cuda_gradient_diagnostic(
            state, None, compiler=None, cache=tmp_path / "not-created"
        )
    assert not (tmp_path / "not-created").exists()


def test_source_owner_validates_spin_storage_and_packs_ao_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vibeqc import _stationary_cuda as runtime

    names = (
        "stationary_create",
        "stationary_topology",
        "stationary_reset",
        "stationary_tasks",
        "stationary_nuclear",
        "stationary_geometry",
        "stationary_finish",
        "stationary_finish_reduced",
        "stationary_metrics",
        "stationary_destroy",
    )
    library = SimpleNamespace(**{name: MagicMock(return_value=0) for name in names})
    monkeypatch.setattr(runtime, "file_hash", lambda _path: "binary")
    monkeypatch.setattr(runtime.ct, "CDLL", lambda _path: library)
    monkeypatch.setattr(runtime, "_native_ao_atoms", lambda _basis: np.array([0, 0]))
    primitives = np.array([[1.5, 2.5], [2.0, 3.0]])
    aos = np.array(
        [[0, 0, 1, 1, 0, 0, 0, 0.5], [0, 1, 1, 1, 0, 0, 0, 0.75]],
        dtype=float,
    )
    monkeypatch.setattr(
        runtime,
        "_layout",
        lambda _basis: (
            primitives,
            aos,
            ("", ""),
            (("four_center_eri", ("", "", "", "")),),
        ),
    )
    artifact = SimpleNamespace(
        library=tmp_path / "runtime.so", metadata={"binary_sha256": "binary"}
    )
    compiler = SimpleNamespace(target=SimpleNamespace(compute_capability=(12, 0)))
    basis = SimpleNamespace(
        natom=1,
        nprimitive=2,
        nao=2,
        representation="cartesian",
        charge=0,
        multiplicity=1,
        atoms=(SimpleNamespace(atomic_number=1),),
        identity="basis-id",
        packed=np.zeros(3),
    )

    with pytest.raises(ValueError, match="one or two density spin blocks"):
        runtime._CudaSources(basis, artifact, compiler, 0, 4, 2, 4096, spin_blocks=3)

    owner = runtime._CudaSources(
        basis, artifact, compiler, 0, 4, 2, 4096, spin_blocks=2
    )
    assert owner.tasks.shape == (2, 9)
    density = np.stack((np.eye(2), 2 * np.eye(2)))
    owner.reset(1.0e-12, density, 3 * density)
    owner.integral(0, "four_center_eri", (0, 1, 0, 1), charge=2.5)

    assert owner.spin_blocks == 2
    assert owner.used == 1
    assert owner.charges[0] == pytest.approx(2.5)
    np.testing.assert_array_equal(owner.tasks[0, :9], [0, 0, 4, -1, 0, 1, 0, 1, 1])


@pytest.mark.parametrize("aot", (False, True))
def test_weight_fusion_orchestration_runs_without_a_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aot: bool
) -> None:
    from vibeqc import _stationary_cuda as runtime

    contract = SimpleNamespace(
        family="lda", spin="unpolarized", validate=lambda state: state
    )
    monkeypatch.setattr(
        runtime, "StationaryDerivativeContract", lambda _identity: contract
    )

    class FakeCompiler:
        def __init__(self) -> None:
            from vibeqc_compiler.common.cuda_target import cuda_target_info

            self.target = cuda_target_info("sm_120")

    class FakePlan:
        source_names = runtime._SOURCE_NAMES
        spin_blocks = 1
        identity = "plan-id"

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def reduction_program(self, *, atoms: int, sources: object = None) -> object:
            return SimpleNamespace(atoms=atoms, sources=sources)

        def integral_block(self, name: str, **_kwargs: object) -> object:
            return SimpleNamespace(
                weights=SimpleNamespace(logical_hash=f"{name}-weight-hash")
            )

    monkeypatch.setattr(runtime, "CudaCompilerAdapter", FakeCompiler)
    monkeypatch.setattr(runtime, "StationaryGradientPlan", FakePlan)
    monkeypatch.setattr(runtime, "StationaryMeanField", lambda *_a, **_k: object())
    monkeypatch.setattr(runtime, "native_ao_geometry_identity", lambda _basis: "geom")
    monkeypatch.setattr(runtime, "resolve_ks_method", lambda _method: (object(), None))
    monkeypatch.setattr(
        runtime,
        "_layout",
        lambda _basis: (
            np.array([[1.0, 1.0]]),
            np.array([[0, 0, 1, 1, 0, 0, 0, 1.0]], dtype=float),
            ("",),
            (),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "plan_tiles",
        lambda *_a, **_k: SimpleNamespace(peak_bytes=1024, host_bytes=256),
    )
    tensor_plan = SimpleNamespace(peak_bytes=128, host_bytes=64)
    monkeypatch.setattr(runtime, "plan_cuda", lambda *_a, **_k: tensor_plan)

    def artifact(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            library=tmp_path / f"{name}.so",
            metadata={"binary_sha256": f"{name}-sha", "key": f"{name}-key"},
        )

    monkeypatch.setattr(runtime, "emit_first_derivative_cuda", lambda _requests: "cuda")
    monkeypatch.setattr(
        runtime, "compile_stationary_cuda", lambda *_a, **_k: artifact("stationary")
    )
    monkeypatch.setattr(runtime, "compile_grid", lambda *_a, **_k: artifact("grid"))
    monkeypatch.setattr(runtime, "compile_cuda", lambda *_a, **_k: artifact("tensor"))

    if aot:
        for name in ("compile_stationary_cuda", "compile_grid", "compile_cuda"):
            monkeypatch.setattr(
                runtime, name, lambda *a, **k: pytest.fail("compiler used by AOT")
            )
        monkeypatch.setattr(
            runtime,
            "load_stationary_aot_artifact",
            lambda *a, **k: artifact("stationary"),
        )
        monkeypatch.setattr(
            runtime, "_native_grid_artifact", lambda *a, **k: artifact("grid")
        )

    reduction = MagicMock()
    reduction.__enter__.return_value = reduction
    reduction.execute.return_value = SimpleNamespace(
        outputs={"gradient": np.zeros((1, 3))},
        metrics={"owned_device_bytes": 128, "endpoint_ms": 0.1, "device_ms": 0.05},
    )
    monkeypatch.setattr(runtime, "PreparedCuda", lambda *_a, **_k: reduction)

    owner = MagicMock()
    owner.__enter__.return_value = owner
    owner.borrowed_streams = set()
    owner.finish.return_value = {
        name: np.zeros((1, 3)) for name in runtime._SOURCE_NAMES
    }
    owner.reduced.return_value = np.zeros((1, 3))
    admitted: dict[str, int] = {}

    def make_owner(
        _basis: object,
        _artifact: object,
        _compiler: object,
        _device: int,
        _points: int,
        _records: int,
        budget: int,
        *,
        spin_blocks: int = 1,
        target: object = None,
        work_budget: int = 2_000_000,
        timeline: object = None,
        profile_device: bool = False,
    ) -> MagicMock:
        assert timeline is not None
        assert profile_device is False
        admitted["budget"] = budget
        admitted["spin_blocks"] = spin_blocks
        admitted["work_budget"] = work_budget
        return owner

    monkeypatch.setattr(runtime, "_CudaSources", make_owner)
    owner.metrics.side_effect = lambda: {
        "owned_device_bytes": admitted["budget"],
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "launches": 1,
        "primitive_records": owner.integral.call_count,
        "xc_points": 0,
        "grid_pair_visits": 0,
        "stream": 0,
        "task_descriptors": owner.integral.call_count,
        "task_batches": owner.integral.call_count,
    }

    grid_owner = MagicMock()
    grid_owner.__enter__.return_value = grid_owner
    grid_owner.metrics.return_value = {"owned_device_bytes": 0}
    monkeypatch.setattr(runtime, "CudaGrid", lambda *_a, **_k: grid_owner)
    basis = SimpleNamespace(
        identity="basis",
        natom=1,
        nao=1,
        nprimitive=1,
        charge=0,
        atoms=[SimpleNamespace(atomic_number=1)],
    )
    grid = SimpleNamespace(
        points=np.empty((0, 3)),
        owners=np.empty(0, dtype=np.int64),
        weights=np.empty(0),
    )
    source = SimpleNamespace(
        backend="cuda",
        metadata=(3,) + (0,) * 12,
        grid_spec=SimpleNamespace(partition_iterations=3, coincident_tolerance=1.0e-12),
        hamiltonian="all-electron",
        ecp_cores=(0,),
        atomic_weights=np.empty(0),
        values=np.zeros(1),
        export_work={"reads": 1},
    )
    state = SimpleNamespace(
        identity=SimpleNamespace(
            basis_identity="basis", geometry_identity="geom", method="lda-rks"
        ),
        occupations=np.array([1.0]),
        density=np.ones((1, 1, 1)),
        weighted_density=2 * np.ones((1, 1, 1)),
        grid=grid,
        _source=source,
    )
    result = runtime.complete_rks_cuda_gradient_diagnostic(
        state,
        basis,
        compiler=None if aot else FakeCompiler(),
        target=FakeCompiler().target if aot else None,
        aot_directory=tmp_path if aot else None,
        native_grid_library=tmp_path / "native.so" if aot else None,
        cache=tmp_path / "cache",
        tile_points=4,
        integral_terms=2,
        primitive_tile=4,
    )

    owner.reset.assert_called_once_with(1.0e-12, state.density, state.weighted_density)
    assert admitted["spin_blocks"] == 1
    assert owner.integral.call_args_list == [
        call(0, "kinetic", (0, 0)),
        call(0, "nuclear_attraction", (0, 0), 0, 1),
        call(5, "overlap", (0, 0)),
        call(1, "four_center_eri", (0, 0, 0, 0)),
    ]
    owner.reduced.assert_called_once_with()
    assert result.work["tensor_executions"] == 0
    assert (
        result.work["stationary_final_reduction"] == "native-seven-source-device-sum-v1"
    )
    assert result.work["stationary_weight_tensor_executions"] == 0
    assert result.work["stationary_weight_roundtrip_bytes"] == 0
    assert result.work["stationary_state_dw_upload_bytes"] == 16
    assert result.execution.endswith("/generated-device-stationary-weights-v1")


def test_stationary_cuda_production_task_page_default() -> None:
    """Keep the qualified large-page schedule explicit and bounded."""
    import inspect

    from vibeqc._stationary_cuda import (
        _complete_rks_cuda_gradient_diagnostic,
        complete_rks_cuda_gradient_diagnostic,
    )

    for function in (
        _complete_rks_cuda_gradient_diagnostic,
        complete_rks_cuda_gradient_diagnostic,
    ):
        parameter = inspect.signature(function).parameters["primitive_tile"]
        assert parameter.default == 4096

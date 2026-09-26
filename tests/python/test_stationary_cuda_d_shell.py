"""Device-free closure gates for d-shell stationary CUDA task lowering."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _d_shell_basis() -> SimpleNamespace:
    centers = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    primitives = np.asarray([[1.0, 0.7], [0.8, 0.6]])
    aos = np.zeros((2, 16))
    aos[0, :4] = [0, 0, 1, 2]
    aos[0, 4:8] = [2, 0, 0, 0.5]
    aos[0, 8:12] = [0, 2, 0, -0.5]
    aos[1, :4] = [1, 1, 1, 1]
    aos[1, 4:8] = [0, 0, 0, 1.0]
    return SimpleNamespace(
        natom=2,
        nprimitive=2,
        nao=2,
        shells=(
            SimpleNamespace(angular_momentum=2),
            SimpleNamespace(angular_momentum=0),
        ),
        packed=np.concatenate((centers.ravel(), primitives.ravel(), aos.ravel())),
    )


def test_stationary_cuda_d_shell_layout_uses_canonical_component_inventory() -> None:
    from vibeqc._stationary_cuda import _component_domain, _component_mode, _layout
    from vibeqc_compiler.integral.first_derivative_schedule import derivative_requests

    _, _, expansions, requests = _layout(_d_shell_basis())

    assert expansions == ((("xx", 0.5), ("yy", -0.5)), (("", 1.0),))
    assert _component_mode(expansions)
    assert _component_domain(expansions) == ("", "xx", "yy")
    assert requests == derivative_requests(("", "xx", "yy"))


def test_stationary_cuda_d_shell_layout_rejects_unsupported_records() -> None:
    from vibeqc._stationary_cuda import _component_domain, _layout

    basis = _d_shell_basis()
    high_l = SimpleNamespace(
        **{
            **vars(basis),
            "shells": (SimpleNamespace(angular_momentum=3),),
        }
    )
    with pytest.raises(NotImplementedError, match="s/p/d"):
        _layout(high_l)

    packed = basis.packed.copy()
    ao_start = 3 * basis.natom + 2 * basis.nprimitive
    packed[ao_start + 3] = 4
    invalid_components = SimpleNamespace(**{**vars(basis), "packed": packed})
    with pytest.raises(ValueError, match="one to three"):
        _layout(invalid_components)

    with pytest.raises(ValueError, match="exceeds s/p/d"):
        _component_domain(((("xxx", 1.0),),))


def test_stationary_cuda_component_tasks_preserve_public_ao_indices_and_weights() -> (
    None
):
    from vibeqc._stationary_cuda import _CudaSources, _layout
    from vibeqc_compiler.integral.first_derivative_schedule import derivative_binding
    from vibeqc_compiler.method.stationary_cuda import encode_stationary_derivative_kind

    _, aos, expansions, requests = _layout(_d_shell_basis())
    owner = object.__new__(_CudaSources)
    owner.aos = aos
    owner.expansions = expansions
    owner.component_mode = True
    owner.components = tuple(expansion[0][0] for expansion in expansions)
    owner.kinds = {request: kind for kind, request in enumerate(requests)}
    owner.tasks = np.empty((8, 9), dtype=np.int64)
    owner.charges = np.empty(8)
    owner.used = 0

    owner.integral(0, "overlap", (0, 1), charge=2.0)

    assert owner.used == 2
    np.testing.assert_array_equal(owner.tasks[:2, 4:6], [[0, 1], [0, 1]])
    np.testing.assert_allclose(owner.charges[:2], [1.0, -1.0])
    np.testing.assert_array_equal(owner.tasks[:2, 8], [1, 1])
    expected = []
    for component in ("xx", "yy"):
        binding = derivative_binding("overlap", (component, ""))
        expected.append(
            encode_stationary_derivative_kind(
                owner.kinds[binding.request],
                binding,
                rank=2,
                has_nucleus=False,
            )
        )
    np.testing.assert_array_equal(owner.tasks[:2, 0], expected)


def test_stationary_cuda_component_nuclear_task_encodes_binding() -> None:
    from vibeqc._stationary_cuda import _CudaSources
    from vibeqc_compiler.integral.first_derivative_schedule import derivative_binding
    from vibeqc_compiler.method.stationary_cuda import encode_stationary_derivative_kind

    calls = []
    owner = object.__new__(_CudaSources)
    owner.used = 0
    owner.component_mode = True
    owner.kinds = {("nuclear", ()): 7}
    owner.handle = None
    owner._call = lambda *args: calls.append(args)

    owner.nuclear(0, 1, np.asarray([1.0, 2.0]))

    expected = encode_stationary_derivative_kind(
        7,
        derivative_binding("nuclear", ()),
        rank=2,
        has_nucleus=False,
    )
    assert len(calls) == 1
    assert calls[0][0] == "stationary_nuclear"
    assert calls[0][2:] == (expected, 0, 1, 1.0, 2.0)


def test_stationary_cuda_derivative_shards_are_bounded_and_uniquely_named() -> None:
    from vibeqc_compiler.integral.first_derivative_schedule import (
        CUDA_REQUESTS_PER_UNIT,
        MAX_UNIT_BYTES,
        derivative_cuda_sources,
        derivative_requests,
    )

    domain = ("", "x", "xx")
    units = derivative_cuda_sources(domain)
    flattened = tuple(request for requests, _ in units for request in requests)

    assert flattened == derivative_requests(domain)
    assert len(units) > 1
    assert all(len(requests) <= CUDA_REQUESTS_PER_UNIT for requests, _ in units)
    assert all(len(source.encode("utf-8")) <= MAX_UNIT_BYTES for _, source in units)
    for unit, (_, source) in enumerate(units):
        assert f"first_derivative_shard_{unit}" in source
        assert f"first_derivative_shard_{unit}_primitive_0" in source
        assert f"return first_derivative_shard_{unit}_primitive_0" in source


def test_stationary_cuda_shard_adapter_uses_explicit_schedule_width() -> None:
    from vibeqc_compiler.method.stationary_cuda import (
        _sharded_first_derivative_adapter,
    )

    source = _sharded_first_derivative_adapter(3, 16)

    assert "switch (kind / 16u)" in source
    assert "kind % 16u" in source
    assert "case 2: ok = first_derivative_shard_2" in source
    with pytest.raises(ValueError, match="dimensions"):
        _sharded_first_derivative_adapter(3, 0)


def test_stationary_cuda_d_shell_work_cap_is_explicit() -> None:
    import inspect

    from vibeqc._stationary_cuda import (
        _complete_rks_cuda_gradient_diagnostic,
        complete_rks_cuda_gradient_diagnostic,
    )

    assert (
        inspect.signature(_complete_rks_cuda_gradient_diagnostic)
        .parameters["max_primitive_records"]
        .default
        == 16_000_000
    )
    assert (
        inspect.signature(complete_rks_cuda_gradient_diagnostic)
        .parameters["max_primitive_records"]
        .default
        == 16_000_000
    )


def test_prepared_d_shell_execution_selects_component_aot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vibeqc import _stationary_cuda
    from vibeqc._stationary_cuda import PreparedStationaryCudaExecution
    from vibeqc_compiler.method.stationary_cuda import QUALIFIED_SPD_COMPONENTS

    class StopAfterStationaryLoad(Exception):
        pass

    captured = {}
    owner = PreparedStationaryCudaExecution()
    monkeypatch.setattr(owner, "_request", lambda **kwargs: object())
    monkeypatch.setattr(
        _stationary_cuda,
        "compile_stationary_cuda",
        lambda *args, **kwargs: pytest.fail(
            "prepared component AOT path must not JIT-compile stationary CUDA"
        ),
    )

    def load_stationary(aot_directory: object, **kwargs: object) -> object:
        captured["aot_directory"] = aot_directory
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        _stationary_cuda, "load_stationary_aot_artifact", load_stationary
    )

    def stop(*args: object, **kwargs: object) -> None:
        raise StopAfterStationaryLoad

    monkeypatch.setattr(_stationary_cuda, "compile_grid", stop)
    target = SimpleNamespace(architecture="sm_90")
    plan = SimpleNamespace(spin_blocks=1)

    with pytest.raises(StopAfterStationaryLoad):
        owner.ensure(
            state=object(),
            basis=_d_shell_basis(),
            contract=SimpleNamespace(spin=0),
            plan=plan,
            tensor_plans={},
            compiler=object(),
            cache=tmp_path,
            aot_directory=tmp_path,
            target=target,
            requests=(),
            functional=0,
            ecp=False,
            device=0,
            spec=SimpleNamespace(partition_iterations=3),
            grid_plan=SimpleNamespace(peak_bytes=0),
            source_bytes=0,
            tile_points=1,
            primitive_tile=1,
            integral_terms=1,
            work_budget=1,
            max_device_bytes=1,
            max_host_bytes=1,
            host_bound=0,
        )

    assert captured == {
        "aot_directory": tmp_path,
        "functional": 0,
        "spin": 0,
        "plan": plan,
        "architecture": "sm_90",
        "iterations": 3,
        "component_domain": QUALIFIED_SPD_COMPONENTS,
    }


def test_component_subset_loads_real_aot_contract_and_uses_packaged_kind_indices(
    tmp_path: Path,
) -> None:
    """Opaque binary bytes exercise loading/ABI mapping without executing a GPU."""
    import json

    from vibeqc._stationary_cuda import _artifact_derivative_requests, _layout
    from vibeqc_compiler.common.provenance import file_hash
    from vibeqc_compiler.integral.first_derivative_schedule import derivative_requests
    from vibeqc_compiler.method.stationary_cuda import (
        QUALIFIED_SPD_COMPONENTS,
        _qualified_aot_plan,
        load_stationary_aot_artifact,
        stationary_aot_contract_identity,
    )

    plan = _qualified_aot_plan(0, "unpolarized")
    library = tmp_path / "libvibeqc_stationary_lda_rks_spd.so"
    library.write_bytes(b"opaque component inventory fixture; never dlopen")
    metadata = {
        "schema": "vibeqc.stationary-cuda-aot.v3",
        "functional": 0,
        "spin": "unpolarized",
        "plan_identity": plan.identity,
        "partition_iterations": 3,
        "architectures": ["sm_120"],
        "code_objects": [{"architecture": "sm_120", "kind": "cubin"}],
        "contract_identity": stationary_aot_contract_identity(
            0, spin="unpolarized", component_domain=QUALIFIED_SPD_COMPONENTS
        ),
        "component_domain": list(QUALIFIED_SPD_COMPONENTS),
        "primitive_shard_width": 16,
        "primitive_shards": 23,
        "source_identity": "opaque-component-fixture",
        "binary_sha256": file_hash(library),
        "binary_bytes": library.stat().st_size,
        "compile_contract": {"fp64": True, "fmad": False},
    }
    (tmp_path / "vibeqc_stationary_lda_rks_spd.json").write_text(json.dumps(metadata))
    artifact = load_stationary_aot_artifact(
        tmp_path,
        functional=0,
        spin="unpolarized",
        plan=plan,
        architecture="sm_120",
        component_domain=QUALIFIED_SPD_COMPONENTS,
    )
    _, _, _, subset = _layout(_d_shell_basis())
    inventory = _artifact_derivative_requests(subset, artifact)
    packaged = derivative_requests(QUALIFIED_SPD_COMPONENTS)
    assert inventory == packaged
    kinds = {request: index for index, request in enumerate(inventory)}
    assert any(kinds[request] != index for index, request in enumerate(subset))
    for request in subset:
        assert packaged[kinds[request]] == request

    # JIT keeps its compact ABI; a foreign package cannot silently drop kinds.
    assert _artifact_derivative_requests(subset, SimpleNamespace(metadata={})) == subset
    with pytest.raises(ValueError, match="omits required"):
        _artifact_derivative_requests(
            subset, SimpleNamespace(metadata={"component_domain": [""]})
        )

"""Device-free closure gates for d-shell stationary CUDA task lowering."""

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

"""Run production inventory/ABI decisions without loading a native CUDA library."""

from __future__ import annotations

import ast
import ctypes
import math
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _tree() -> ast.Module:
    return ast.parse((ROOT / "python/vibeqc/resources_ks.py").read_text())


def _function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _environment() -> dict[str, typing.Any]:
    # Dimensions are deliberately small: overflow/allocations are not mocked as
    # qualified. These tests exercise the actual semantic and lifetime decisions.
    return {
        "typing": typing,
        "ctypes": ctypes,
        "byte_product": lambda *values: math.prod(values),
        "checked_bytes": lambda value, *args: value,
        "_ecp_workspace": lambda *args, **kwargs: 0,
    }


def _load(name: str) -> typing.Any:
    namespace = _environment()
    module = ast.Module(body=[_function(name)], type_ignores=[])
    exec(compile(module, "inventory", "exec"), namespace)  # noqa: S102
    return namespace[name]


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        grid=SimpleNamespace(radial_points=4, angular_polar=3),
        tile_points=16,
        xc_schedule="fused",
        has_nonlocal_correlation=False,
        has_nondefault_composition=False,
    )


def _item(n: int) -> dict[str, typing.Any]:
    return {
        "atoms": 3,
        "spins": 1,
        "grid_points": 64,
        "orbital": {
            "nbf": n,
            "cartesian_nbf": n,
            "shells": 4,
            "primitives": 8,
            "maximum_angular": 1,
        },
    }


@pytest.mark.parametrize(
    "backend,n", [("cpu", 17), ("cuda", 16), ("cuda", 17), ("cuda", 32)]
)
def test_host_inventory_preserves_current_model_and_solver_charge(
    backend: str, n: int
) -> None:
    row = _load("_item_host_inventory")(
        _item(n),
        diis_history=8,
        max_iterations=100,
        pbe=True,
        backend=backend,
        model=_model(),
    )
    expected = (1 << 20) + 16 * 8 * n * n if backend == "cuda" and n > 16 else 0
    assert row["solver_host"] == expected
    retained_fields = (
        "metadata",
        "grid",
        "basis",
        "warm_and_matrices",
        "history",
        "provider",
        "solver_host",
        "xc_schedule_staging",
        "nonlocal_provider",
    )
    assert row["retained"] == sum(row[key] for key in retained_fields)


def _library(quadrature_status: int = 0) -> SimpleNamespace:
    def query(*args: typing.Any) -> int:
        output = args[-2]
        output[:] = [8 << 20] * 3
        return 0

    def quadrature(atoms: int, points: int, output: typing.Any) -> int:
        assert (atoms, points) == (3, 64)
        output._obj.value = 320 << 20
        return quadrature_status

    return SimpleNamespace(
        vibeqc_resource_ks_cuda_v1=query,
        vibeqc_resource_quadrature_cuda_v1=quadrature,
    )


def test_quadrature_setup_coexists_with_prepared_coulomb() -> None:
    result = _load("_cuda_item_inventory")(
        _library(), _item(32), diis_history=8, pbe=True, tile=16
    )
    assert result["quadrature_setup"] == 320 << 20
    assert result["setup"] == 328 << 20


@pytest.mark.parametrize("missing", [False, True])
def test_unavailable_quadrature_inventory_fails_closed(missing: bool) -> None:
    library = _library(quadrature_status=1)
    if missing:
        del library.vibeqc_resource_quadrature_cuda_v1
    with pytest.raises(NotImplementedError, match="quadrature"):
        _load("_cuda_item_inventory")(
            library, _item(32), diis_history=8, pbe=True, tile=16
        )


@pytest.mark.parametrize("version", [None, 0, 1, 3, 5])
def test_native_semantic_abi_version_is_checked_even_for_default_model(
    version: int | None,
) -> None:
    function = _function("ks_resource_request")
    guard = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "library is not None"
    )
    library = SimpleNamespace(vibeqc_ks_resource_inventory_version_v1=lambda: 1)
    if version is not None:
        library.vibeqc_ks_options_version = lambda: version
    namespace = _environment()
    namespace.update(library=library, model=_model(), method="pbe-rks")
    module = ast.Module(body=[guard], type_ignores=[])
    code = compile(module, "inventory ABI", "exec")
    if version == 1:
        exec(code, namespace)  # noqa: S102
    else:
        with pytest.raises(NotImplementedError, match="semantic KS execution-plan ABI"):
            exec(code, namespace)  # noqa: S102


def test_request_does_not_read_retired_version_properties() -> None:
    attributes = {
        node.attr
        for node in ast.walk(_function("ks_resource_request"))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "model"
    }
    assert not attributes & {
        "requires_nonlocal_v5",
        "requires_composition_v2",
        "requires_schedule_v3",
    }
    assert {"has_nonlocal_correlation", "has_nondefault_composition"} <= attributes

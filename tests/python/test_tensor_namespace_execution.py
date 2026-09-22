"""Backend-generic TensorIR reference execution for issue #633 B2."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from vibeqc_compiler.array_api import namespace as frontend_xp
from vibeqc_compiler.array_api import trace
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    execute,
    input_tensor,
)


class _NumpyArrayNamespace:
    """Small Array API-shaped namespace that forces the non-legacy dispatch."""

    __name__ = "test_numpy_array_namespace"
    float32 = np.float32
    float64 = np.float64
    int64 = np.int64
    asarray = staticmethod(np.asarray)
    zeros = staticmethod(np.zeros)
    reshape = staticmethod(np.reshape)
    permute_dims = staticmethod(np.transpose)
    broadcast_to = staticmethod(np.broadcast_to)
    take = staticmethod(np.take)
    sum = staticmethod(np.sum)
    isfinite = staticmethod(np.isfinite)
    all = staticmethod(np.all)
    any = staticmethod(np.any)
    abs = staticmethod(np.abs)
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    pow = staticmethod(np.power)

    @staticmethod
    def astype(
        value: np.ndarray, dtype: typing.Any, *, copy: bool = True
    ) -> np.ndarray:
        return value.astype(dtype, copy=copy)


def _captured_program() -> tuple[Program, dict[str, np.ndarray]]:
    ao = IndexSpace("ao", "ao", 2)
    batch = IndexSpace("batch", "batch", 3)
    flat = IndexSpace("flat", "matrix", 4)
    p, q = Index("p", ao), Index("q", ao)
    spec = TensorSpec((p, q), role="input")
    flat_index = Index("flat", flat)
    batch_index = Index("batch", batch)
    program = trace(
        lambda x, y: {
            "norm": frontend_xp.sum(
                frontend_xp.sqrt(x * x + y * y),
                axis=1,
            ),
            "ratio": (x + y) / (x * y),
            "power": frontend_xp.pow(x, 2),
            "exp_log": frontend_xp.log(frontend_xp.exp(x)),
            "permuted": frontend_xp.permute_dims(x, (1, 0)),
            "reshaped": frontend_xp.reshape(x, (4,), indices=(flat_index,)),
            "sliced": frontend_xp.slice(x, ((0, 2), (0, 1))),
            "taken": frontend_xp.take(x, (1, 0), axis=0),
            "broadcast": frontend_xp.broadcast_to(
                frontend_xp.sum(x, axis=0),
                (3, 2),
                indices=(batch_index, q),
                axes=(1,),
            ),
        },
        {"x": spec, "y": spec},
    )
    feeds = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "y": np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float64),
    }
    return program, feeds


def test_declared_namespace_matches_numpy_without_changing_program_identity() -> None:
    program, feeds = _captured_program()
    before = program.logical_hash
    expected = execute(program, feeds, debug=True)
    actual = execute(program, feeds, namespace=_NumpyArrayNamespace, debug=True)

    assert program.logical_hash == before
    assert expected.backend == "numpy-cpu-interpreter"
    assert actual.backend == "array-namespace:_NumpyArrayNamespace"
    assert actual.logical_retained_bytes == expected.logical_retained_bytes
    assert actual.intermediates.keys() == expected.intermediates.keys()
    for name in expected.outputs:
        np.testing.assert_allclose(
            actual.outputs[name], expected.outputs[name], rtol=0, atol=0
        )

    original = feeds["x"].copy()
    actual.outputs["ratio"][:] = -1
    np.testing.assert_array_equal(feeds["x"], original)


def test_explicit_numpy_namespace_keeps_independent_legacy_oracle() -> None:
    program, feeds = _captured_program()
    result = execute(program, feeds, namespace=np)
    assert result.backend == "numpy-cpu-interpreter"


def test_namespace_missing_einsum_extension_fails_closed() -> None:
    ao = IndexSpace("ao", "ao", 2)
    p, q = Index("p", ao), Index("q", ao)
    x = input_tensor("x", TensorSpec((p, q), role="input"))
    program = Program({"gram": einsum("pi,qi->pq", x, x)})
    feed = {"x": np.eye(2, dtype=np.float64)}

    with pytest.raises(
        RuntimeError, match="einsum.*does not provide an einsum extension"
    ):
        execute(program, feed, namespace=_NumpyArrayNamespace)


def test_array_api_strict_preserves_dtype_device_and_values_when_available() -> None:
    strict = pytest.importorskip("array_api_strict")
    program, numpy_feeds = _captured_program()
    feeds = {
        name: strict.asarray(value, dtype=strict.float64)
        for name, value in numpy_feeds.items()
    }
    expected = execute(program, numpy_feeds).outputs
    result = execute(program, feeds, namespace=strict, debug=True)

    assert result.backend == "array-namespace:array_api_strict"
    for name, value in result.outputs.items():
        assert value.dtype == strict.float64
        assert value.device == feeds["x"].device
        np.testing.assert_allclose(np.asarray(value), expected[name], rtol=0, atol=0)
    assert all(
        value.device == feeds["x"].device for value in result.intermediates.values()
    )

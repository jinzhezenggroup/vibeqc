"""DLPack copy requirements cannot weaken after a consumer has been invoked."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.array_api.interop import (
    DLPackDevice,
    DLPackInteropError,
    import_dlpack,
)


def test_consumer_typeerror_is_not_retried_without_copy_control() -> None:
    source = np.arange(4.0)
    calls = []

    def consume(value: object, *, copy: bool | None = None) -> object:
        calls.append(copy)
        if copy is False:
            raise TypeError("producer export failed after starting the handoff")
        return np.array(value, copy=True)

    with pytest.raises(DLPackInteropError) as error:
        import_dlpack(source, SimpleNamespace(from_dlpack=consume))
    assert isinstance(error.value.__cause__, TypeError)
    assert calls == [False]


def test_legacy_signature_is_selected_before_invocation() -> None:
    source = np.arange(4.0)
    calls = []

    def consume(value: object) -> object:
        calls.append(value)
        return np.from_dlpack(value)

    result = import_dlpack(source, SimpleNamespace(from_dlpack=consume))
    assert len(calls) == 1
    assert result.copy_control == "legacy-dlpack-zero-copy"
    assert np.shares_memory(source, result.array)


def test_legacy_consumer_internal_typeerror_is_not_retried() -> None:
    calls = []

    def consume(value: object) -> object:
        calls.append(value)
        raise TypeError("legacy export failed")

    with pytest.raises(DLPackInteropError):
        import_dlpack(np.arange(2.0), SimpleNamespace(from_dlpack=consume))
    assert len(calls) == 1


@pytest.mark.parametrize("device", [DLPackDevice(True, 0), DLPackDevice(1, False)])
def test_typed_expected_device_does_not_bypass_integer_validation(
    device: DLPackDevice,
) -> None:
    calls = []

    def consume(value: object, *, copy: bool = False) -> object:
        calls.append(value)
        return value

    with pytest.raises(DLPackInteropError, match="invalid DLPack device"):
        import_dlpack(
            np.arange(2.0), SimpleNamespace(from_dlpack=consume), expected_device=device
        )
    assert calls == []


def test_opaque_consumer_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    def no_signature(value: object) -> object:
        raise ValueError("opaque builtin")

    monkeypatch.setattr(inspect, "signature", no_signature)
    calls = []

    def consume(value: object, *, copy: bool | None = None) -> object:
        calls.append(copy)
        raise TypeError("opaque handoff failed")

    with pytest.raises(DLPackInteropError):
        import_dlpack(np.arange(2.0), SimpleNamespace(from_dlpack=consume))
    assert calls == [False]


def test_numpy_roundtrip_keeps_shared_storage() -> None:
    source = np.arange(12.0).reshape(3, 4)
    result = import_dlpack(source, np, expected_device=(1, 0))
    assert np.shares_memory(source, result.array)
    np.testing.assert_array_equal(result.array, source)
    assert result.copy_control == "explicit-copy-false"


def test_unsupported_required_consumer_argument_is_rejected_before_call() -> None:
    calls = []

    def consume(value: object, required: object) -> object:
        calls.append(value)
        return required

    with pytest.raises(DLPackInteropError):
        import_dlpack(np.arange(2.0), SimpleNamespace(from_dlpack=consume))
    assert calls == []


def test_positional_only_copy_control_is_not_treated_as_legacy() -> None:
    calls = []

    def consume(value: object, copy: bool | None = None, /) -> object:
        calls.append(copy)
        return value

    with pytest.raises(DLPackInteropError, match="copy keyword"):
        import_dlpack(np.arange(2.0), SimpleNamespace(from_dlpack=consume))
    assert calls == []

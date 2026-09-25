"""DLPack interoperability contract for issue #633 B3."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc_compiler.array_api import (
    DLPACK_INTEROP_VERSION,
    DLPackDevice,
    DLPackInteropError,
    capabilities,
    import_dlpack,
)


def test_numpy_dlpack_round_trip_is_same_device_and_zero_copy() -> None:
    source = np.arange(12, dtype=np.float64).reshape(3, 4)
    first = import_dlpack(source, np, expected_device=(1, 0))

    assert first.zero_copy is True
    assert first.source_device == DLPackDevice(1, 0)
    assert first.target_device == first.source_device
    assert first.copy_control in {"explicit-copy-false", "legacy-dlpack-zero-copy"}
    assert np.shares_memory(source, first.array)

    second = import_dlpack(first.array, np)
    assert second.zero_copy is True
    assert np.shares_memory(first.array, second.array)
    np.testing.assert_array_equal(second.array, source)


def test_capabilities_describe_bounded_dlpack_contract() -> None:
    assert capabilities()["dlpack_interop"] == {
        "version": DLPACK_INTEROP_VERSION,
        "import": "same-device-zero-copy",
        "device_transfer": False,
        "stream_handoff": "consumer-owned-protocol",
        "raw_capsule_ownership": "not-retained",
    }


def test_missing_producer_protocol_fails_closed() -> None:
    with pytest.raises(DLPackInteropError, match="__dlpack__"):
        import_dlpack(object(), np)


def test_requested_device_relocation_requires_explicit_copy() -> None:
    source = np.arange(4, dtype=np.float64)
    with pytest.raises(DLPackInteropError, match="explicit copy"):
        import_dlpack(source, np, expected_device=(2, 0))


def test_missing_consumer_protocol_fails_closed() -> None:
    source = np.arange(4, dtype=np.float64)
    with pytest.raises(DLPackInteropError, match="from_dlpack"):
        import_dlpack(source, object())


class _FakeProducer:
    def __init__(self, device: tuple[int, int] = (1, 0)) -> None:
        self.device = device

    def __dlpack__(self, *args: object, **kwargs: object) -> object:
        return object()

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.device


class _WrongDeviceNamespace:
    @staticmethod
    def from_dlpack(source: object, *, copy: bool = False) -> object:
        del source, copy
        return _FakeProducer((2, 0))


class _LegacyNamespace:
    @staticmethod
    def from_dlpack(source: object) -> object:
        return source


class _BrokenNamespace:
    @staticmethod
    def from_dlpack(source: object, *args: object, **kwargs: object) -> object:
        del source, args, kwargs
        raise ValueError("layout unsupported")


def test_consumer_device_change_requires_explicit_copy() -> None:
    with pytest.raises(DLPackInteropError, match="changed device"):
        import_dlpack(_FakeProducer(), _WrongDeviceNamespace)


def test_invalid_device_identity_fails_closed() -> None:
    with pytest.raises(DLPackInteropError, match="invalid DLPack device"):
        import_dlpack(_FakeProducer((True, 0)), _WrongDeviceNamespace)


def test_legacy_consumer_signature_retains_zero_copy_contract() -> None:
    source = _FakeProducer()
    result = import_dlpack(source, _LegacyNamespace)
    assert result.array is source
    assert result.zero_copy is True
    assert result.copy_control == "legacy-dlpack-zero-copy"


def test_consumer_layout_failure_diagnoses_copy_boundary() -> None:
    with pytest.raises(DLPackInteropError, match="explicit copy"):
        import_dlpack(_FakeProducer(), _BrokenNamespace)

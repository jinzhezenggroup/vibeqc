import typing

import pytest
from vibeqc_compiler.common.native_call import checked_native_call


def test_checked_native_call_success() -> None:
    observed = {}

    def native(value: typing.Any, error: typing.Any, size: int) -> int:
        observed["value"] = value
        observed["size"] = size
        observed["error_empty"] = error.value == b""
        return 0

    assert checked_native_call(native, 7) is None
    assert observed == {"value": 7, "size": 2048, "error_empty": True}


def test_checked_native_call_raises_native_message() -> None:
    def native(error: typing.Any, size: int) -> int:
        assert size == 2048
        error.value = b"native failure"
        return 5

    with pytest.raises(RuntimeError, match="native failure"):
        checked_native_call(native)


def test_checked_native_call_preserves_requested_error_type() -> None:
    def native(error: typing.Any, size: int) -> int:
        error.value = b"bad input"
        return 1

    with pytest.raises(ValueError, match="bad input"):
        checked_native_call(native, error_type=ValueError)

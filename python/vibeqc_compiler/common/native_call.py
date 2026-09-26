"""Small ctypes helpers for status-returning native ABIs."""

from __future__ import annotations

import ctypes
import typing


def checked_native_call(
    function: typing.Callable[..., int],
    *args: typing.Any,
    error_type: type[Exception] = RuntimeError,
    buffer_bytes: int = 2048,
) -> None:
    """Call a native function that appends an error buffer and raises on status."""
    if type(buffer_bytes) is not int or buffer_bytes <= 0:
        raise ValueError("native error buffer size must be a positive integer")
    error = ctypes.create_string_buffer(buffer_bytes)
    status = function(*args, error, len(error))
    if status:
        raise error_type(error.value.decode())

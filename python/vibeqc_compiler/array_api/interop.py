"""Fail-closed DLPack interoperability for issue #633 B3.

This module keeps external array ownership at the protocol boundary. It never
extracts or stores a raw DLPack capsule itself: the destination namespace owns
stream handoff and capsule consumption through ``from_dlpack``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

DLPACK_INTEROP_VERSION = 1


class DLPackInteropError(RuntimeError):
    """Raised when a requested DLPack handoff cannot be verified safely."""


@dataclass(frozen=True, slots=True)
class DLPackDevice:
    """Normalized DLPack device identity (device type, device id)."""

    device_type: int
    device_id: int

    def as_tuple(self) -> tuple[int, int]:
        """Return the protocol device tuple."""
        return (self.device_type, self.device_id)


@dataclass(frozen=True, slots=True)
class DLPackImport:
    """Verified same-device DLPack import result."""

    array: object
    source_device: DLPackDevice
    target_device: DLPackDevice
    copy_control: str

    @property
    def zero_copy(self) -> bool:
        """Whether this handoff used the zero-copy DLPack contract."""
        return self.source_device == self.target_device


def _normalize_device(value: object, *, owner: str) -> DLPackDevice:
    if not isinstance(value, tuple) or len(value) != 2:
        raise DLPackInteropError(f"{owner} returned an invalid DLPack device tuple")
    device_type, device_id = value
    if (
        isinstance(device_type, bool)
        or isinstance(device_id, bool)
        or not isinstance(device_type, int)
        or not isinstance(device_id, int)
        or device_type <= 0
        or device_id < 0
    ):
        raise DLPackInteropError(f"{owner} returned an invalid DLPack device tuple")
    return DLPackDevice(device_type, device_id)


def dlpack_device(value: object) -> DLPackDevice:
    """Return and validate the DLPack device declared by *value*."""
    device_fn = getattr(value, "__dlpack_device__", None)
    dlpack_fn = getattr(value, "__dlpack__", None)
    if not callable(device_fn) or not callable(dlpack_fn):
        raise DLPackInteropError(
            "value does not implement both __dlpack__ and __dlpack_device__"
        )
    return _normalize_device(device_fn(), owner="DLPack producer")


def _expected_device(
    value: DLPackDevice | tuple[int, int] | None,
) -> DLPackDevice | None:
    if value is None:
        return None
    if isinstance(value, DLPackDevice):
        value = value.as_tuple()
    return _normalize_device(value, owner="expected device")


def import_dlpack(
    source: object,
    namespace: object,
    *,
    expected_device: DLPackDevice | tuple[int, int] | None = None,
) -> DLPackImport:
    """Import *source* through a namespace-owned same-device DLPack handoff.

    Request ``copy=False`` unless signature binding establishes a legacy
    one-argument consumer before invocation. An opaque consumer is called once
    with ``copy=False``; wrap an older opaque consumer with an explicit legacy
    signature. Never retry a failed handoff with weaker copy requirements.
    Device relocation requires a separate copy/synchronization contract.
    """
    source_device = dlpack_device(source)
    wanted = _expected_device(expected_device)
    if wanted is not None and wanted != source_device:
        raise DLPackInteropError(
            "requested DLPack device differs from the producer; "
            "an explicit copy is required"
        )

    consumer = getattr(namespace, "from_dlpack", None)
    if not callable(consumer):
        raise DLPackInteropError("array namespace does not provide from_dlpack")

    # Decide compatibility without invoking the consumer or exporting a capsule.
    # TypeError may originate after consumption, not only from keyword binding.
    legacy = False
    try:
        signature = inspect.signature(consumer)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        try:
            signature.bind(source, copy=False)
        except TypeError as exc:
            if "copy" in signature.parameters:
                raise DLPackInteropError(
                    "from_dlpack must accept the copy keyword"
                ) from exc
            try:
                signature.bind(source)
            except TypeError as exc:
                raise DLPackInteropError("unsupported from_dlpack signature") from exc
            legacy = True

    try:
        if legacy:
            imported = consumer(source)
            copy_control = "legacy-dlpack-zero-copy"
        else:
            imported = consumer(source, copy=False)
            copy_control = "explicit-copy-false"
    except Exception as exc:
        raise DLPackInteropError(
            "zero-copy DLPack import failed; the consumer may require an explicit copy"
        ) from exc

    target_device = dlpack_device(imported)
    if target_device != source_device:
        raise DLPackInteropError(
            "DLPack consumer changed device; an explicit copy is required"
        )
    return DLPackImport(
        array=imported,
        source_device=source_device,
        target_device=target_device,
        copy_control=copy_control,
    )
